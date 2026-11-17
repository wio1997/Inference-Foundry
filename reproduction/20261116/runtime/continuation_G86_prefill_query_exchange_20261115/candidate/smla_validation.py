"""Separate G48 finite actual-model shadow, off unless explicit directory+ARMED.

Exactly ends245760/253952, same request, every admitted compressed layer.
Terminal receipts are required on all16 actual workers before root scores.
"""
import json
import os
from pathlib import Path

DIRECTORY = os.environ.get('SGLANG_G48_VALIDATE_DIR', '')
ENDS = (245760, 253952)


class SmlaShadow:
    def __init__(self, rank, layers):
        self.rank, self.layers = rank, sorted(layers)
        self.path = Path(DIRECTORY) / ('global_tp_rank_%s.json' % rank)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise RuntimeError('G48 shadow receipt already exists: %s' % self.path)
        self.windows, self.active, self.disabled = [], None, False
        self.write('armed')

    def write(self, status, **extra):
        record = dict(schema=1, status=status, disabled=self.disabled,
                      shadow_enabled=not self.disabled, pid=os.getpid(),
                      global_tp_rank=self.rank, expected_layers=self.layers,
                      expected_ends=ENDS, windows=self.windows,
                      all_checks_passed=self.disabled and all(
                          c['strict_pass'] for w in self.windows for c in w['checks']),
                      scope='SparseMLA restored output only, not full-model logits', **extra)
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(record, indent=2) + '\n')
        tmp.replace(self.path)

    def begin(self, context, fb):
        if self.disabled or not (Path(DIRECTORY) / 'ARMED').is_file():
            return
        end = context['end']
        if self.active is not None:
            raise RuntimeError('G48 previous shadow window did not complete')
        if end not in ENDS or end in [w['global_end'] for w in self.windows]:
            return
        if end != ENDS[len(self.windows)]:
            raise RuntimeError('G48 shadow expected selected late ends in order')
        reqs = getattr(fb, 'reqs', None)
        ids = [str(r.rid) for r in reqs] if reqs else fb.req_pool_indices.cpu().tolist()
        if self.windows and ids != self.windows[0]['request_ids']:
            raise RuntimeError('G48 shadow requires same request in both chunks')
        import torch
        expected = torch.arange(end-8192, end, device=fb.positions.device, dtype=fb.positions.dtype)
        if not torch.equal(fb.positions, expected):
            raise AssertionError('G48 actual positions do not match consecutive suffix geometry')
        self.active = dict(global_end=end, local_end=context['local_end'],
                           request_ids=ids, checks=[], layers=[])
        self.windows.append(self.active)
        self.write('active')

    def compare(self, bindings, layer_id, ratio, q, candidate, args, context):
        if self.active is None or self.disabled:
            return
        import torch
        import torch.distributed as dist
        if layer_id not in self.layers or layer_id in self.active['layers']:
            raise RuntimeError('G48 unexpected/duplicate layer in shadow')
        # Read-only operator repeat with EXACT same current model inputs/cache.
        # No second model forward, cache write, compressor or QLI execution.
        reference, _ = bindings.sparse(q, **args)
        finite, different, maxabs = True, 0, 0.0
        for lo in range(0, q.shape[0], 256):
            a, b = reference[lo:lo+256].float(), candidate[lo:lo+256].float()
            finite = finite and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
            delta = (a-b).abs()
            different += int(torch.count_nonzero(delta))
            maxabs = max(maxabs, float(delta.max()))
        passed = finite and different == 0
        ok = torch.tensor([int(passed)], dtype=torch.int32, device=q.device)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=context['group'])
        passed_group = bool(ok.item())
        self.active['checks'].append(dict(layer_id=layer_id, ratio=ratio,
            finite=finite, different_elements=different, max_abs=maxabs,
            strict_pass=passed and passed_group, same_current_model_inputs=True))
        self.active['layers'].append(layer_id)
        if not passed_group:
            self.write('failed')
            raise AssertionError('G48 strict actual-model SparseMLA shadow failed')
        if set(self.active['layers']) == set(self.layers):
            self.active['status'] = 'passed_exact'
            self.active = None
            if len(self.windows) == 2:
                self.disabled = True
                self.write('disabled', terminal=True,
                           score_permitted_only_after_all_16_workers_disabled=True)
            else:
                self.write('armed')
        else:
            self.write('active')
