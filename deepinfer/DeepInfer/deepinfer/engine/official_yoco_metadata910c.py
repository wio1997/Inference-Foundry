"""Bridge original prepare_prompt_v4 metadata to the verified 910C consumer ABI.

Selection, visibility pivots and persistent stores come from the original C++
preparer. This module only decodes its coordinates and tracks native snapshots.
"""
import hashlib
import importlib
import importlib.util
import os
from pathlib import Path
import sys

KEEP = 140
WINDOW = 128


def load_bindings():
    """Permit an isolated official build during integration; never replace a loaded build."""
    artifact = os.environ.get("DEEPINFER_910C_METADATA_BINDINGS")
    if artifact:
        path = Path(artifact).resolve()
        expected = os.environ.get("DEEPINFER_910C_METADATA_BINDINGS_SHA256")
        if not expected or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("official metadata binding identity is not pinned")
        import deepinfer
        if "deepinfer._C" not in sys.modules:
            spec = importlib.util.spec_from_file_location("deepinfer._C", path)
            if spec is None or spec.loader is None:
                raise RuntimeError("official metadata binding cannot be loaded")
            module = importlib.util.module_from_spec(spec)
            sys.modules["deepinfer._C"] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                sys.modules.pop("deepinfer._C", None)
                raise
            deepinfer._C = module
        module = importlib.import_module("deepinfer._C")
        if Path(module.__file__).resolve() != path:
            raise RuntimeError("another official metadata binding is already loaded")
    api = importlib.import_module("deepinfer._C.engine")
    for name in ("PrepareArgs", "SequenceMetadata", "SequenceData", "YocoPrefillTruncationMode", "prepare_prompt_v4"):
        if not hasattr(api, name):
            raise RuntimeError("original metadata binding lacks " + name)
    if not hasattr(api.SequenceMetadata,"swa_valid_start"):
        raise RuntimeError("official metadata binding does not expose the SWA valid bound")
    return api


def decode(prepared, specifications, layout, *, window_size=WINDOW, keep_tokens=KEEP,
           swa_block_size=None):
    """Decode original virtual SWA sections; do not recompute selection or visibility."""
    if layout not in ("mask", "compact"):
        raise ValueError("unknown original YOCO layout")
    total = len(specifications)
    if layout == "compact":
        cu = prepared["yoco_consumer_cu_seqlens_q"].tolist()
        positions = prepared["yoco_consumer_positions"].tolist()
        indices = prepared["yoco_consumer_query_indices"].tolist()
        stores = prepared["yoco_consumer_slot_mapping_store_swa"].tolist()
        pivots = prepared["yoco_consumer_swa_index_pivots"].tolist()
        if cu[-1] != prepared["yoco_consumer_total_q"]:
            raise ValueError("original consumer total differs from its coordinates")
    else:
        cu = [0]
        positions = []
        for full, start, end, valid in specifications:
            cu.append(cu[-1] + end - start)
            positions.extend(range(start, end))
        indices = list(range(cu[-1]))
        stores = prepared["slot_mapping_store_swa"].tolist()
        pivots = prepared["yoco_mask_swa_index_pivots"].tolist()
    if (len(cu) != total + 1 or cu[0] != 0 or
            not len(positions) == len(indices) == len(stores) == len(pivots) == cu[-1]):
        raise ValueError("inconsistent original consumer arrays")
    result = []
    full_offset = 0
    for request, (full, start, end, valid) in enumerate(specifications):
        count = cu[request + 1] - cu[request]
        clip = end - start - count
        if not 1 <= count <= end - start or not 0 <= valid <= start < end <= full:
            raise ValueError("invalid original consumer request bounds")
        entries = []
        for row in range(cu[request], cu[request + 1]):
            pos = positions[row]
            if indices[row] != full_offset + pos - start:
                raise ValueError("original query index differs from its absolute position")
            p0, p1, p2, p3 = pivots[row]
            if p1 < 0 or p3 < 0 or p1 + p3 > window_size:
                raise ValueError("invalid original visibility section lengths")
            old_lo = start - window_size + p0 - request * window_size
            new_lo = start + clip + p2 - (total * window_size + cu[request])
            if p1 and p3 and old_lo + p1 != new_lo:
                raise ValueError("910C contiguous view cannot express original visibility")
            lo = old_lo if p1 else new_lo if p3 else None
            if lo is not None and (lo < valid or lo + p1 + p3 != pos + 1):
                raise ValueError("original visibility is not a causal contiguous suffix")
            entries.append(dict(position=pos, visible_start=lo, visible_count=p1+p3, store_slot=stores[row]))
        result.append(dict(full_length=full, start=start, end=end, valid_start=valid,
                           layout=layout, store_layout="official", rows=entries,
                           window_size=window_size, keep_tokens=keep_tokens,
                           swa_block_size=window_size if swa_block_size is None else swa_block_size))
        full_offset += end - start
    return result


def snapshot_valid(value, length):
    if type(value) is not int or not 0 <= value <= length:
        raise ValueError("invalid prefix consumer SWA valid start")
    return value


def valid_after(entry, previous):
    """Track the actual contiguous stored suffix, including all-clipped chunks."""
    start, end = entry["start"], entry["end"]
    snapshot_valid(previous, start)
    written = [row["position"] for row in entry["rows"] if row["store_slot"] >= 0]
    if not written:
        return end
    if written != list(range(written[0], end)) or written[0] < start:
        raise ValueError("original persistent stores are not a contiguous chunk suffix")
    stored_width = entry.get('swa_block_size', entry.get('window_size', WINDOW))
    return written[0] if written[0] > start else max(previous, end - stored_width, 0)


def prepare_original(metadata, *, cp_size, block_size=128, layout='compact',
                     keep_tokens=KEEP, window_size=WINDOW, swa_block_size=None,
                     max_backtracking_tokens=6, api=None):
    """Use actual original metadata owners with the existing consumer decoder."""
    api = load_bindings() if api is None else api
    swa_block_size = window_size if swa_block_size is None else swa_block_size
    if (not metadata or cp_size != 8 or block_size not in (128, 256)
            or type(window_size) is not int or window_size <= 0
            or type(swa_block_size) is not int or swa_block_size < window_size
            or layout not in ('mask', 'compact') or keep_tokens < window_size + max_backtracking_tokens):
        raise ValueError('unsupported original YOCO metadata configuration')
    if any(not row.is_prompt or row.is_placeholder for row in metadata):
        raise ValueError('original YOCO preparation requires real prompt metadata')
    args = api.PrepareArgs()
    args.seq_metadata_list = metadata
    args.block_size = block_size
    args.swa_block_size = swa_block_size
    args.max_ngram_size = 0
    args.engram_compressed_pad_token = 0
    args.image_sentinel = -1
    args.cp_size = cp_size
    args.window_size = window_size
    args.dense_compress_ratios = []
    args.sparse_compress_ratios = [1, 2]
    args.yoco_prefill_keep_tokens = keep_tokens
    args.yoco_prefill_truncation_mode = (api.YocoPrefillTruncationMode.COMPACT
        if layout == 'compact' else api.YocoPrefillTruncationMode.MASK)
    args.max_backtracking_tokens = max_backtracking_tokens
    specs = [(row.full_len, row.seqlen_k-row.seqlen_q, row.seqlen_k, row.swa_valid_start) for row in metadata]
    result = decode(api.prepare_prompt_v4(args), specs, layout,
                    window_size=window_size, keep_tokens=keep_tokens, swa_block_size=swa_block_size)
    for entry, row in zip(result, metadata, strict=True):
        entry["state_id"] = row.state_id
    return result


class OfficialYocoMetadata:
    def __init__(self, *, cp_size, block_size=128, window_size=WINDOW,
                 keep_tokens=KEEP, max_backtracking_tokens=6):
        if cp_size != 8 or block_size not in (128, 256):
            raise ValueError("unsupported verified 910C metadata geometry")
        self.api = load_bindings()
        self.cp_size, self.block_size = cp_size, block_size
        self.window_size, self.keep_tokens = window_size, keep_tokens
        self.max_backtracking_tokens = max_backtracking_tokens
        self.sources = {}
        self.full_lengths = {}
        self.prepares = 0

    def begin_job(self, requests):
        # SequenceMetadata stores borrowed C++ pointers. Keep both SequenceData
        # owners alive through every PrepareArgs call and discard only at job end.
        self.sources = {slot: (self.api.SequenceData(list(r["token_ids"])),
                               self.api.SequenceData(list(r["token_ids"])))
                        for slot, r in enumerate(requests)}
        self.full_lengths = {slot: len(r["token_ids"]) for slot, r in enumerate(requests)}

    def end_job(self):
        self.sources.clear()
        self.full_lengths.clear()

    def bind_slot(self, slot, request):
        """Replace a borrowed source only after the old Worker generation drains."""
        self.sources[slot] = (self.api.SequenceData(list(request["token_ids"])),
                              self.api.SequenceData(list(request["token_ids"])))
        self.full_lengths[slot] = len(request["token_ids"])

    def prepare(self, slots, starts, ends, valid_starts, *, layout="compact"):
        from deepinfer.sampling_params import SamplingParams
        if not (len(slots) == len(starts) == len(ends) == len(valid_starts)) or not slots:
            raise ValueError("metadata must cover the active requests")
        if len(set(slots)) != len(slots) or layout not in ("mask", "compact"):
            raise ValueError("duplicate slots or invalid YOCO layout")
        specs = []
        metadata = []
        for request, (slot, start, end, valid) in enumerate(zip(slots, starts, ends, valid_starts, strict=True)):
            if slot not in self.sources:
                raise ValueError("metadata has no live request owner")
            full = self.full_lengths[slot]
            if not 0 <= valid <= start < end <= full:
                raise ValueError("invalid live prefill cursor")
            seq_data, compressed = self.sources[slot]
            seq = self.api.SequenceMetadata(
                request_id=str(slot), seq_id=slot, is_prompt=True,
                seq_data=seq_data, compressed_seq_data=compressed, multi_modal_data=None,
                sampling_params=SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=False),
                block_table=list(range((full+self.block_size-1)//self.block_size+8)),
                seqlen_q=end-start, seqlen_k=end, last_output_token=0,
                last_accepted_token=[], last_next_n_token=[], state_id=request)
            seq.swa_valid_start = valid
            metadata.append(seq)
            specs.append((full, start, end, valid))
        result = prepare_original(metadata, cp_size=self.cp_size, block_size=self.block_size,
            layout=layout, window_size=self.window_size, keep_tokens=self.keep_tokens,
            max_backtracking_tokens=self.max_backtracking_tokens, api=self.api)
        self.prepares += 1
        return result


def configure_consumer(model):
    """Retain the previously verified local ABI prerequisites for all consumer layers."""
    if (not model.sequence_parallel or model.sp.world != 8 or model.tail_ffn_source != 20 or
            any(layer.ratio != 1 for layer in model.layers[20:])):
        raise ValueError("unsupported official first-consumer topology")
    if any(not layer.cp_o_available for layer in model.layers[20:]):
        raise ValueError("official consumer projection pool was not loaded")
    for layer in model.layers[20:]:
        layer.cp_local_o = True
    for feature in ("MHC_PRE", "Q_A", "Q_B", "WKV", "FFN_MHC_PRE", "ROUTER_LINEAR"):
        os.environ["DEEPINFER_910C_YOCO_FIXED_"+feature] = "1"
        os.environ["DEEPINFER_910C_YOCO_EXTEND_"+feature] = "1"
    os.environ["DEEPINFER_910C_YOCO_FP32_TP_REFERENCE"] = "1"



def official_mode(mode):
    """Validate opt-in request controls before applying any of their environment."""
    names=("yoco_official_prefill","yoco_official_mask","yoco_merged_qb","yoco_dp_local")
    values={name:mode.get(name,False) for name in names}
    if any(type(value) is not bool for value in values.values()):
        raise ValueError("official YOCO controls must be boolean")
    enabled=values["yoco_official_prefill"]
    if not enabled and (values["yoco_official_mask"] or values["yoco_merged_qb"] or values["yoco_dp_local"]):
        raise ValueError("official MASK or merged Q_b requires official prefill")
    if values["yoco_dp_local"]:
        if not (mode.get("zos_official_scheduler",False) and mode.get("zos_continuous_admission",False)):
            raise ValueError("per-DP YOCO requires original continuous Scheduler")
        if os.environ.get("DEEPINFER_910C_PREFILL_QUERY_PADDING")=="1":
            raise ValueError("per-DP YOCO requires native global query padding disabled")
    if enabled:
        if not mode.get("tail_ffn",False) or any(mode.get(k,False) for k in
                ("yoco_exact_compact","yoco_meta_fuse","yoco_preplan")):
            raise ValueError("official prefill requires its exclusive consumer path")
        if mode.get("yoco_bounded",False) and not mode.get("yoco_bounded_140",False):
            raise ValueError("official prefill retains140;disable it to select256 fallback")
    return {"DEEPINFER_910C_YOCO_OFFICIAL_PREFILL":"1" if enabled else "0",
            "DEEPINFER_910C_YOCO_LAYOUT":"mask" if values["yoco_official_mask"] else "compact",
            "DEEPINFER_910C_YOCO_MERGED_QB":"1" if values["yoco_merged_qb"] else "0",
            "DEEPINFER_910C_YOCO_DP_LOCAL":"1" if values["yoco_dp_local"] else "0"}


def prepare_native_prefill(runner, owned, slots, chunks, hits, *, internal=False):
    """Select original rank-local metadata; keep EP fallback votes uniform."""
    self = runner
    official_requested = os.environ.get('DEEPINFER_910C_YOCO_OFFICIAL_PREFILL') == '1'
    dp_local = os.environ.get('DEEPINFER_910C_YOCO_DP_LOCAL') == '1'
    query_padding = os.environ.get('DEEPINFER_910C_PREFILL_QUERY_PADDING') == '1'
    local_selection = dp_local or not query_padding
    if dp_local and (not official_requested or query_padding):
        # forward_tokens has a global padding collective only on the native
        # branch. Mixed official/scratch calls must never enter that branch.
        raise ValueError('per-DP YOCO requires official prefill and no global query padding')
    official_metadata = None
    official_eligible = False
    global_rejected = False
    if official_requested:
        official_eligible = (not internal and bool(slots) and all(s < self.slots for s in slots)
                            and self.model.sequence_parallel
                            and all(len(owned[s][1]['token_ids']) > 140 for s in slots))
        # All ranks still vote in the same order. Local selection does not
        # change layer/MC2 calls; CP/TP gathers belong to that DP's group.
        global_rejected = self._global_busy(not official_eligible)
        # With global query padding disabled, TP/CP metadata belongs to this
        # DP. An idle peer must not discard its live consumer view: earlier
        # compact chunks may have intentionally stored no SWA history.
        use_official = official_eligible if local_selection else not global_rejected
        if use_official:
            starts = [self.model.next_positions[s] for s in slots]
            ends = [p+len(c) for p,c in zip(starts,chunks,strict=True)]
            valid = [self.model.yoco_swa_valid_start[s] for s in slots]
            layout = os.environ.get('DEEPINFER_910C_YOCO_LAYOUT','compact')
            official_metadata = self.official_yoco_metadata.prepare(slots,starts,ends,valid,layout=layout)
            self.model.yoco_official_metadata_calls += 1
        else:
            self.model.yoco_official_fallback_calls += 1
        if global_rejected and query_padding:
            # This branch has a global predicate, so every EP rank votes in
            # the same order before either any cache write or an exception.
            unavailable = not internal and any(
                slot < self.slots and self.model.yoco_swa_valid_start[slot] >
                max(0,self.model.next_positions[slot]-WINDOW+1) for slot in slots)
            if self._global_busy(unavailable):
                raise ValueError('native YOCO fallback lacks retained SWA history; disable global query padding for per-DP continuation')
    bounded_keep = 0
    # Predicate is global, including ranks already holding official metadata.
    # Testing local metadata here would strand the scratch rank's all_reduce.
    if os.environ.get('DEEPINFER_910C_YOCO_BOUNDED') == '1' and (not official_requested or global_rejected):
        from .bounded_replay910c import eligible
        keep = 140 if os.environ.get('DEEPINFER_910C_YOCO_BOUNDED_140') == '1' else 256
        local = official_metadata is None and eligible(self.model,owned,slots,chunks,hits,internal=internal,keep=keep)
        if (os.environ.get('DEEPINFER_910C_PREFILL_QUERY_PADDING') == '1'
                and len({len(x) for x in chunks}) != 1):
            local = False
        if not self._global_busy(not local):
            bounded_keep = keep
    decisions = getattr(self, '_official_yoco_decisions', None)
    if decisions is not None and getattr(self, '_scheduled_official_yoco', False):
        decisions.append(dict(internal=internal, slots=list(slots), query_rows=sum(map(len,chunks)),
            local_eligible=official_eligible, global_rejected=global_rejected,
            dp_local=dp_local, local_selection=local_selection,
            official_applied=official_metadata is not None, bounded_keep=bounded_keep))
    _record_functional_route(self,owned,slots,chunks,internal,official_eligible,global_rejected,
        local_selection,official_metadata,bounded_keep)
    return bounded_keep, official_metadata


def _record_functional_route(runner,owned,slots,chunks,internal,eligible,rejected,local_selection,metadata,bounded):
    """Optional integer-only route provenance for completeness testing."""
    root=os.environ.get('DEEPINFER_910C_FUNCTIONAL_ROUTE_TRACE')
    if not root:return
    from pathlib import Path
    import json
    import torch.distributed as dist
    root=Path(root).resolve()
    if not root.is_relative_to('/data/tiankuan/wio/910c-deepinfer-172/diagnostics/module-validation-20261107'):
        raise ValueError('functional route trace outside owned diagnostics')
    root.mkdir(parents=True,exist_ok=True)
    number=getattr(runner,'_functional_route_records',0);runner._functional_route_records=number+1
    real=[s<runner.slots and not internal for s in slots]
    record=dict(number=number,rank=dist.get_rank(),slots=list(slots),internal=internal,
        full_lengths=[len(owned[s][1]['token_ids']) if yes else None for s,yes in zip(slots,real,strict=True)],
        starts=[runner.model.next_positions[s] for s in slots],
        valid_starts=[runner.model.yoco_swa_valid_start[s] for s in slots],query_lengths=list(map(len,chunks)),
        local_eligible=eligible,global_rejected=rejected,local_selection=local_selection,
        official_applied=metadata is not None,bounded_keep=bounded,
        retained_lengths=None if metadata is None else [len(e['rows']) for e in metadata],
        first_visible_counts=None if metadata is None else [e['rows'][0]['visible_count'] for e in metadata])
    with (root/f'rank-{dist.get_rank()}.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
