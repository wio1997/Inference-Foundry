"""Optional hashes of actual NPU rows for a bounded prefill diagnostic.

No reference model is executed. With no enable marker, existing compute
probes remain untouched. The diagnostic client enables tracing only after an
ordinary untraced comparison fails, so fences cannot qualify a normal pass.
"""
import hashlib
import json
import os
from pathlib import Path
import torch


def select_rows(lengths, rows, rank, world, target, request=None):
    if request is None and target not in lengths:
        return None
    request = lengths.index(target) if request is None else request
    if not 0 <= request < len(lengths):
        return None
    width_for_request = lengths[request]
    logical = target-width_for_request
    first = sum(lengths[:request])
    total = sum(lengths)
    if rows == total:
        return first, first+width_for_request, logical, rank
    width = (total+world-1)//world
    if rows != width:
        return None
    left, right = max(first, rank*width), min(first+width_for_request, (rank+1)*width)
    return (left-rank*width, right-rank*width, logical+left-first, None) if left < right else None


class Trace:
    def __init__(self, model, root):
        from . import ascend_w8a8_packed as packed
        self.packed, self.model, self.root = packed, model, Path(root)
        self.rank = int(os.environ['RANK'])
        config=json.loads((self.root/'enable').read_text())
        self.target=int(config['target_length'])
        if self.target<1:raise ValueError('positive diagnostic target length required')
        self.dump_stages=set(config.get('dump_stages',()))
        self.dump_layers=set(config.get('dump_layers',()))
        self.dump_limit=int(config.get('max_dump_bytes',0))
        self.dump_bytes=0
        self.cp_rank, self.world = model.sp.rank, model.sp.world
        path = self.root/f'rank{self.rank}.jsonl'
        self.phase = (sum(json.loads(line).get('type') == 'phase' for line in path.read_text().splitlines())
                      if path.exists() else 0)
        self.active, self.state, self.request = False, None, None
        self.old = packed._layer20_probe, packed._layer21_probe, dict(packed._late_consumer_probes)
        self.root.mkdir(parents=True, exist_ok=True)
        self.stream = path.open('a', buffering=1)
        for layer in range(len(model.layers)):
            packed._late_consumer_probes[layer] = self.callback(layer)
        packed._layer20_probe = packed._late_consumer_probes[20]
        packed._layer21_probe = packed._late_consumer_probes[21]

    def emit(self, value):
        self.stream.write(json.dumps(value, separators=(',', ':'))+'\n')

    def callback(self, layer):
        def observe(stage, scope, *values, state=None):
            if state is not None:
                self.state = state
            if layer == 0 and stage == 'input':
                state = self.state
                self.active = self.target in state.batch.query_lengths
                if self.active:
                    self.request = state.batch.query_lengths.index(self.target)
                    self.phase += 1
                    self.emit(dict(type='phase', phase=self.phase, rank=self.rank,
                        lengths=list(state.batch.query_lengths), seq_lengths=list(state.batch.seq_lengths)))
            if not self.active:
                return
            lengths = self.state.batch.query_lengths
            for index, value in enumerate(values):
                if value is None or value.ndim == 0:
                    continue
                # Original compact YOCO keeps the suffix of the same request.
                # Its query length changes; its request index does not.
                selected = select_rows(lengths, value.shape[0], self.cp_rank,
                                       self.world,self.target,self.request)
                if selected is None:
                    continue
                first, last, logical, axis = selected
                # CPU inspects opaque bytes only; no CPU numerical golden.
                payload = value[first:last].detach().contiguous().cpu().view(torch.uint8)
                payload = payload.reshape(last-first, -1).numpy()
                hashes = [hashlib.sha256(row.tobytes()).hexdigest() for row in payload]
                record=dict(type='rows',phase=self.phase,rank=self.rank,layer=layer,
                    stage=stage,scope=scope,tensor=index,dtype=str(value.dtype),
                    shape=list(value.shape[1:]),axis=axis,first=logical,hashes=hashes)
                if layer in self.dump_layers and stage in self.dump_stages and self.dump_bytes+payload.nbytes<=self.dump_limit:
                    name=f'rows/p{self.phase}-r{self.rank}-l{layer}-{stage}-t{index}-{logical}.bin'
                    path=self.root/name;path.parent.mkdir(exist_ok=True)
                    raw=payload.tobytes();path.write_bytes(raw);self.dump_bytes+=len(raw)
                    record.update(payload=name,payload_sha256=hashlib.sha256(raw).hexdigest())
                self.emit(record)
        return observe

    def close(self):
        self.packed._layer20_probe, self.packed._layer21_probe, previous = self.old
        self.packed._late_consumer_probes.clear()
        self.packed._late_consumer_probes.update(previous)
        self.stream.close()


def update(model, current, root):
    enabled = (Path(root)/'enable').exists()
    if current is not None and not enabled:
        current.close()
        return None
    if current is None and enabled:
        from deepinfer import envs
        if not envs.DEEPINFER_BATCH_INVARIANT:
            raise ValueError('row comparison requires the invariant diagnostic profile')
        return Trace(model, root)
    return current
