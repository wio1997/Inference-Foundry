#!/usr/bin/env python3
"""Run401 reversible sparse markers. check is read-only for borrowed sources.
Install/restore only with service stopped; use a new --state-dir on install.
"""
import argparse
import hashlib
import json
from pathlib import Path
ROOT = Path('/data/wio/Inference_Foundry')
ASCEND = Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend')
SOURCES = dict(runtime=ROOT/'runtime/extreme_decode.py', state=ROOT/'runtime/fixed_decode.py',
    serving=ROOT/'runtime/fixed_serving.py', adapter=ROOT/'bootstrap/vllm_dspark_handoff.py',
    utils=ASCEND/'spec_decode/utils.py', dsa=ASCEND/'attention/dsa_v1.py')
MARK = '# EXTREME_RUN401_COUNT_MARKERS'


def insert(src, anchor, block, before=False):
    if src.count(anchor) != 1:
        raise ValueError(f'expected one anchor, found {src.count(anchor)}: {anchor!r}')
    return src.replace(anchor, block+anchor if before else anchor+block)


def scoped(src, function, transform):
    a = src.index('    def '+function+'(')
    b = src.find('\n    def ', a+1)
    if b < 0:
        b = len(src)
    return src[:a]+transform(src[a:b])+src[b:]


def patch(key, src):
    if MARK in src:
        raise ValueError('already patched')
    src = insert(src, 'import torch\n', MARK+'\nfrom scripts import loop078_count_markers as _count401\n')
    if key == 'runtime':
        return insert(src, '        self._profile_cycle_begin()\n', '        _count401.step(self)\n', before=True)
    if key == 'state':
        anchor = '''        state.num_sampled.copy_(
            torch.where(state.active_mask, acceptance.num_sampled, 0).to(torch.int32)
        )
'''
        # The original compound expression is preserved byte-for-byte.
        src = insert(src, anchor, '        _count401.event("W_PRE_65", state.num_sampled)\n', before=True)
        return insert(src, anchor, '        _count401.event("W_POST_65", state.num_sampled)\n')
    if key == 'serving':
        src = insert(src, '        cycles = 0\n', '        _count401.start(self)\n', before=True)
        src = insert(src, '            return [int(value) for value in provider().tolist()]\n',
                     '            _count401.progress()\n', before=True)
        return insert(src, '        counts_cpu = count_history[:cycles].cpu()\n',
                      '        _count401.export(self, counts_cpu, cycles)\n')
    if key == 'utils':
        anchor = '    out = seq_lens_cpu.clone()\n'
        src = insert(src, anchor, '    _run401_clone = _count401.clone_before(seq_lens_cpu, "spec_decode.utils.build_parallel_draft_seq_lens_cpu.clone")\n', before=True)
        src = insert(src, anchor, '    _count401.clone_after(_run401_clone, out)\n')
        anchor = '    out[:num_reqs].add_(query_len)\n'
        src = insert(src, anchor, '    _run401_read = _count401.downstream_before(out, "spec_decode.utils.build_parallel_draft_seq_lens_cpu.add_")\n', before=True)
        return insert(src, anchor, '    _count401.downstream_after(_run401_read)\n')
    if key == 'dsa':
        for expression, method in [('_seq_lens_cpu[reqs_start:].max().item()', 'build_prefill_metadata'),
                                   ('_seq_lens_cpu[: self.num_decodes].max().item()', 'build_decode_metadata')]:
            anchor = '            max_seq_lens = '+expression+'\n'
            src = insert(src, anchor, f'            _run401_read = _count401.downstream_before(_seq_lens_cpu, "dsa_v1.{method}.max_item")\n', before=True)
            src = insert(src, anchor, '            _count401.downstream_after(_run401_read)\n')
        return src
    src = insert(src, '        current = torch.npu.current_stream(counts.device)\n',
                 '        _count401.launch(self, counts)\n', before=True)
    src = insert(src, '            self._host_count_copy.copy_(counts, non_blocking=True)\n',
                 '            _count401.event("R_BEGIN_64", counts)\n', before=True)
    src = insert(src, '            self._host_copy_event.record()\n',
                 '            _count401.event("R_DONE_64", counts)\n')
    src = src.replace('    def _commit_host_mirrors(self) -> None:\n',
        '    def _commit_host_mirrors(self, _run401_reason="unspecified") -> None:\n        _count401.commit(self, _run401_reason)\n')
    def commit_body(body):
        body = insert(body, '        self._host_copy_event.synchronize()\n',
                      '        _count401.host("sync_pre")\n', before=True)
        body = insert(body, '        self._host_copy_event.synchronize()\n',
                      '        _count401.host("sync_post")\n')
        body = insert(body, '        self._committed_emitted_count[:batch].add_(counts)\n',
                      '        _count401.host("count_add_pre", counts)\n', before=True)
        body = insert(body, '        self._committed_emitted_count[:batch].add_(counts)\n',
                      '        _count401.host("count_add_post")\n')
        first = '            mirror[:batch].add_(counts)\n'
        if body.count(first) != 2:
            raise ValueError('expected two host mirror update loops')
        body = body.replace(first, '            _count401.host("seq_add_pre", mirror)\n'+first+
                            '            _count401.host("seq_add_post", mirror)\n', 1)
        # Place after the second loop, before following method/decorator.
        last = body.rfind(first)+len(first)
        return body[:last]+'        _count401.host("mirrors_done")\n'+body[last:]
    src = scoped(src, '_commit_host_mirrors', commit_body)
    for fn, reason in [('park_completed_slots', 'park_completed_slots'),
                       ('validate_host_mirrors', 'validation_drain'), ('_execute', 'normal_proposer')]:
        src = scoped(src, fn, lambda body, reason=reason: insert(body,
            'self._commit_host_mirrors()', '').replace('self._commit_host_mirrors()',
            f'self._commit_host_mirrors("{reason}")'))
    return src

def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['check', 'install', 'restore'])
    ap.add_argument('--state-dir', type=Path)
    ap.add_argument('--record', type=Path, required=True)
    ap.add_argument('--expected-check', type=Path)
    args = ap.parse_args()
    rows = []
    if args.action in ('check', 'install'):
        prepared = []
        for key, path in SOURCES.items():
            old = path.read_bytes()
            new = patch(key, old.decode()).encode()
            compile(new, str(path), 'exec')
            prepared.append((key, path, old, new))
            rows.append(dict(key=key, source=str(path), original=digest(old), patched=digest(new)))
        for helper in ('loop078_count_markers.py', 'loop078_count_markers_analyze.py'):
            path = ROOT/'scripts'/helper
            compile(path.read_bytes(), str(path), 'exec')
        if args.action == 'install':
            if args.expected_check is None:
                raise ValueError('install requires reviewed --expected-check source hash record')
            expected = json.loads(args.expected_check.read_text())
            if expected.get('action') != 'check' or expected.get('files') != rows:
                raise ValueError('sources/generated patch differ from reviewed check')
            if args.state_dir is None or args.state_dir.exists():
                raise ValueError('install needs a new --state-dir')
            args.state_dir.mkdir(parents=True)
            for key, path, old, new in prepared:
                (args.state_dir/f'{key}.orig').write_bytes(old)
            (args.state_dir/'manifest.json').write_text(json.dumps(rows, indent=2))
            # Prevalidation completed before any borrowed-source write.
            try:
                for key, path, old, new in prepared:
                    if path.read_bytes() != old:
                        raise ValueError(f'source changed during install: {path}')
                    path.write_bytes(new)
            except BaseException:
                for key, path, old, new in prepared:
                    if path.read_bytes() == new:
                        path.write_bytes(old)
                raise
    else:
        if args.state_dir is None:
            raise ValueError('restore needs --state-dir')
        rows = json.loads((args.state_dir/'manifest.json').read_text())
        for row in rows:
            if SOURCES[row['key']] != Path(row['source']):
                raise ValueError('source manifest mismatch')
            if digest(Path(row['source']).read_bytes()) != row['patched']:
                raise ValueError('installed source changed; refusing overwrite')
            if digest((args.state_dir/f"{row['key']}.orig").read_bytes()) != row['original']:
                raise ValueError('backup changed')
        for row in rows:
            Path(row['source']).write_bytes((args.state_dir/f"{row['key']}.orig").read_bytes())
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(dict(action=args.action, files=rows), indent=2)+'\n')
    print(json.dumps(dict(action=args.action, files=len(rows), installed=args.action=='install')))


if __name__ == '__main__':
    main()
