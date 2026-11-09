"""Adapt the official consumer view to contiguous CP and 910C paged attention."""

from itertools import accumulate, groupby
from dataclasses import dataclass

import torch

from deepinfer.npu_ops.cp910c import QuerySlice
from deepinfer.npu_ops.dsa910c import DSABatch, sparse_mla
from .empty_swa910c import compressed_only as run_c1_only
from .swa_store910c import prepare_store, store
from .yoco_exact_compact910c import prepare_compaction_plan


KEEP, WINDOW = 140, 128


def compressed_only(query,batch,sink,cache,table,selected,*,window=WINDOW):
    """Keep explicit C1-only operator calls within the verified query width."""
    from deepinfer import envs
    if envs.DEEPINFER_BATCH_INVARIANT or window != WINDOW:
        from deepinfer.npu_ops.dsa910c import paged_mla
        return paged_mla(query,batch,sink=sink,window=window,ratio=1,
            compressed_cache=cache,compressed_table=table,compressed_indices=selected,
            batch_invariant=envs.DEEPINFER_BATCH_INVARIANT)
    limit = 32
    if query.shape[0]<=limit:
        return run_c1_only(query,batch,sink,cache,table,selected)
    pieces = []
    offset = 0
    for request,(width,end) in enumerate(zip(batch.query_lengths,batch.seq_lengths,strict=True)):
        for first in range(0,width,limit):
            last = min(first+limit,width)
            start_row,end_row = offset+first,offset+last
            sub = DSABatch.build([last-first],[end-width+last],query.device)
            pieces.append(run_c1_only(query[start_row:end_row],sub,sink,cache,
                                      table[request:request+1],selected[start_row:end_row]))
        offset += width
    return torch.cat(pieces,0)


@dataclass(frozen=True)
class ChunkView:
    lengths: tuple
    selected: tuple
    positions: tuple
    official_store: tuple
    bases: tuple
    keep_tokens: int = KEEP
    window_size: int = WINDOW

    @classmethod
    def from_metadata(cls, metadata, starts, ends):
        """Validate full/consumer coordinates before any producer cache writes."""
        if len(metadata) != len(starts) or len(starts) != len(ends):
            raise ValueError("official YOCO metadata must cover the packed requests")
        geometries = {(entry.get('keep_tokens', KEEP), entry.get('window_size', WINDOW)) for entry in metadata}
        if len(geometries) != 1:
            raise ValueError('official YOCO batch requires one configured window and retained length')
        keep_tokens, window_size = next(iter(geometries))
        if (type(window_size) is not int or type(keep_tokens) is not int
                or not 0 < window_size <= keep_tokens):
            raise ValueError('invalid configured YOCO window or retained length')
        lengths, indices, positions, stores, bases = [], [], [], [], []
        offset = 0
        for request, (entry, start, end) in enumerate(zip(metadata, starts, ends, strict=True)):
            full, valid = entry["full_length"], entry["valid_start"]
            if not 0 <= valid <= start < end <= full:
                raise ValueError("unsupported official YOCO chunk bounds")
            if entry["start"] != start or entry["end"] != end:
                raise ValueError("official YOCO chunk does not match the native cursor")
            rows = entry["rows"]
            layout = entry.get("layout","compact")
            if layout not in ("compact","mask"):
                raise ValueError("unknown YOCO consumer layout")
            store_layout = entry.get("store_layout", "consumer")
            if store_layout not in ("consumer", "official"):
                raise ValueError("unknown official YOCO persistent layout")
            state_id = entry.get("state_id", request)
            stored_width = entry.get('swa_block_size', window_size)
            if type(stored_width) is not int or stored_width < window_size:
                raise ValueError('configured SWA storage is smaller than its visible window')
            if type(state_id) is not int or state_id < 0:
                raise ValueError("invalid official YOCO state owner")
            first = start if layout=="mask" else min(max(start, full-keep_tokens), end-1)
            if [row["position"] for row in rows] != list(range(first,end)):
                raise ValueError("official YOCO consumer selection does not match the declared layout")
            base = None
            for row in rows:
                pos = row["position"]
                lo = max(0, valid, full-keep_tokens, pos-window_size+1)
                count = max(0,pos-lo+1)
                # Read the actual C++ pivots. Validate that the CANN window
                # adapter can express them without silently changing keys.
                if row["visible_count"] != count or row["visible_start"] != (lo if count else None):
                    raise ValueError("unsupported official YOCO SWA visibility")
                if count and base is None:
                    base = lo
                persistent = pos>=full-keep_tokens or layout=="mask" and store_layout=="official"
                expected_store = state_id*stored_width+pos%stored_width if persistent and end-pos<=stored_width else -1
                if row["store_slot"] != expected_store:
                    raise ValueError("invalid official YOCO persistent store")
                indices.append(offset+pos-start)
                positions.append(pos)
                stores.append(row["store_slot"])
            if base is None:
                if layout=="compact" and len(rows)!=1:
                    raise ValueError("empty official SWA must be a single placeholder")
            elif base < 0 or end-base > keep_tokens:
                raise ValueError("official SWA does not fit the retained native window")
            bases.append(base)
            lengths.append(len(rows))
            offset += end-start
        return cls(tuple(lengths),tuple(indices),tuple(positions),tuple(stores),tuple(bases),keep_tokens,window_size)


def gather_kept(model, state, trunk, previous, view):
    """Reuse the validated native CP gather plan with the original row indices."""
    plan = prepare_compaction_plan(model,state,view.keep_tokens,official_selection=(view.lengths,view.selected))
    cp = state.context_query
    if plan is None:
        return trunk, previous, state.selected, state.candidates
    _, _, _, rows, width, local_indices, gathered_indices = plan

    def compact(value, fill):
        if value is None:
            return None
        if value.shape[0] < cp.end-cp.start:
            raise ValueError("query-aligned tensor lacks its CP-owned rows")
        padded = value.new_full((width,*value.shape[1:]),fill)
        padded[:len(rows)].copy_(value.index_select(0,local_indices))
        return model.sp.gather(padded,width*model.sp.world).index_select(0,gathered_indices)

    new = QuerySlice.build(view.lengths,state.batch.seq_lengths,model.sp.world,model.sp.rank)
    selected, candidates = compact(state.selected,-1), compact(state.candidates,-1)
    selected = selected[new.start:new.end].contiguous()
    candidates = None if candidates is None else candidates[new.start:new.end].contiguous()
    return model.sp.split(compact(trunk,0)), model.sp.split(compact(previous,0)), selected, candidates


class ConsumerCache:
    """Hold one forward's temporary read pages and original persistent write plan."""

    def __init__(self, plane, slots, view):
        device = plane.cache.device
        self.bases = dict(zip(slots,view.bases,strict=True))
        self.slots = tuple(slots)
        requests = [slot for slot,q in zip(slots,view.lengths,strict=True) for _ in range(q)]
        self.store_plan = prepare_store(plane,view.positions,requests,view.official_store)
        self.stores = view.official_store
        self.row_count = len(view.positions)
        self.window_size = view.window_size
        required_rows = max((view.positions[end-1] + 1 - base
            for base, end in zip(view.bases, accumulate(view.lengths), strict=True)
            if base is not None), default=1)
        blocks = (required_rows + plane.block_size - 1) // plane.block_size
        stride = blocks * plane.block_size
        history_positions, history_slots, history_dest = [], [], []
        new_source, new_dest = [], []
        offset = 0
        for i,(slot,base,q) in enumerate(zip(slots,view.bases,view.lengths,strict=True)):
            if base is not None:
                first = max(view.positions[offset],base)
                first_row = offset+first-view.positions[offset]
                for pos in range(base,first):
                    history_positions.append(pos)
                    history_slots.append(slot)
                    history_dest.append(i*stride+pos-base)
                new_source.extend(range(first_row,offset+q))
                new_dest.extend(i*stride+view.positions[j]-base for j in range(first_row,offset+q))
            offset += q
        # History is read before any persistent update. The separate temporary
        # pages keep all consumer rows, including those with official slot -1.
        self.history_plan = plane.plan(history_positions,history_slots) if history_positions else None
        self.history_dest = torch.tensor(history_dest,dtype=torch.int64,device=device)
        self.new_source = torch.tensor(new_source,dtype=torch.int64,device=device)
        self.new_dest = torch.tensor(new_dest,dtype=torch.int64,device=device)
        self.pages = plane.cache.new_zeros((blocks*len(slots),plane.block_size,1,plane.width)) if new_source else None
        self.table = torch.arange(blocks*len(slots),dtype=torch.int32,device=device).view(-1,blocks)
        self.local_groups = None

    def write(self, plane, values):
        if values.shape[0] != self.row_count:
            raise ValueError("official YOCO KV payload does not match its consumer rows")
        values = values[:,None]
        if self.pages is not None:
            flat = self.pages.flatten(0,1)
            if self.history_plan is not None:
                history = plane.cache.flatten(0,1).index_select(0,self.history_plan.physical_rows)
                flat.index_copy_(0,self.history_dest,history)
            flat.index_copy_(0,self.new_dest,values.index_select(0,self.new_source))
        store(plane,values,self.store_plan)

    def attention(self, query, batch, slots, sink, cache, table, selected):
        if self.local_groups is None:
            segments = []
            for request,(slot,width,end) in enumerate(zip(slots,batch.query_lengths,batch.seq_lengths,strict=True)):
                base = self.bases[slot]
                start = end-width
                empty = width if base is None else min(width,max(0,base-start))
                if empty:
                    segments.append((None,request,slot,empty,start+empty))
                if empty<width:
                    segments.append((base,request,slot,width-empty,end))
            groups = []
            row = 0
            for base,entries in groupby(segments,lambda item:item[0]):
                _,requests,group_slots,lengths,ends = zip(*entries,strict=True)
                last = row+sum(lengths)
                request,end_request = requests[0],requests[-1]+1
                if tuple(range(request,end_request))!=requests:
                    raise ValueError("noncontiguous local YOCO request slice")
                sub = batch if row==0 and last==query.shape[0] else DSABatch.build(lengths,ends,query.device)
                ids = [self.slots.index(slot) for slot in group_slots]
                tmp_table = self.table[ids].contiguous() if base is not None else None
                groups.append((base,row,last,request,end_request,sub,tmp_table))
                row = last
            if row!=query.shape[0]:
                raise ValueError("native YOCO attention lost query rows")
            self.local_groups = groups
        pieces = []
        for base,a,b,c,d,sub,tmp_table in self.local_groups:
            q, chosen, cmp_table = query[a:b], selected[a:b], table[c:d].contiguous()
            if base is None:
                out = compressed_only(q,sub,sink,cache,cmp_table,chosen,window=self.window_size)
            else:
                out = sparse_mla(q,self.pages,sub,tmp_table,sink=sink,ratio=1,
                                 window=self.window_size,
                                 compressed_cache=cache,compressed_table=cmp_table,
                                 compressed_indices=chosen,swa_base=base,swa_repacked=True)
            pieces.append(out)
        return pieces[0] if len(pieces)==1 else torch.cat(pieces,0)


def enter(model, state, trunk, previous, view):
    if state.pre_stored_source != 20 or state.context_query is None:
        raise ValueError("full C1 and original CP state are required before YOCO consumer entry")
    trunk, previous, selected, candidates = gather_kept(model,state,trunk,previous,view)
    batch = DSABatch.build(view.lengths,state.batch.seq_lengths,model.device)
    cache = model.cache.swa[20]
    cp = QuerySlice.build(batch.query_lengths,batch.seq_lengths,model.sp.world,model.sp.rank)
    state.batch, state.positions, state.context_query = batch,batch.positions(),cp
    state.bind_row_controls(model.sp,state.full_images_seq_mask,
                            state.full_gate_force_random,selected=view.selected)
    state.query_slots = [state.slots[i] for i in cp.request_indices]
    state.query_batch = DSABatch.build(cp.query_lengths,cp.seq_lengths,model.device) if cp.query_lengths else None
    state.query_swa_table = cache.table(state.query_slots)
    state.selected, state.candidates = selected,candidates
    state.valid_query_lengths = view.lengths
    state.tail_ffn_indices = None
    state.official_consumer = ConsumerCache(cache,state.slots,view)
    if model.dspark is not None:
        model.dspark.prepare_capture(state,model.sp)
    return trunk, previous
