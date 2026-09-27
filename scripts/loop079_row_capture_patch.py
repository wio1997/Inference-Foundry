#!/usr/bin/env python3
"""Reversible diagnostic patch; check compiles generated sources without installing.
Install only with service stopped. Restore verifies installed hashes first.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
ASCEND = Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend')
SOURCES = dict(graph=ASCEND/'compilation/acl_graph.py', w4a8=ASCEND/'quantization/methods/w4a8.py',
    comm=ASCEND/'ops/fused_moe/moe_comm_method.py',
    draft=ASCEND/'spec_decode/dspark_proposer.py',
    runtime=ROOT/'runtime/extreme_decode.py', serving=ROOT/'runtime/fixed_serving.py')
MARK = '# EXTREME_RUN437_ROW_IDENTITY'


def insert(src, anchor, block, before=False):
    if src.count(anchor) != 1:
        raise ValueError(f'expected one anchor, got {src.count(anchor)}: {anchor!r}')
    return src.replace(anchor, block+anchor if before else anchor+block)


def patch(key, src):
    if MARK in src:
        raise ValueError('already patched')
    if key == 'graph':
        src = insert(src, '                        output = self.runnable(*args, **kwargs)\n', '', before=True)
        src = src.replace('                        output = self.runnable(*args, **kwargs)\n', """                        # EXTREME_RUN437_ROW_IDENTITY
                        from scripts import loop079_row_capture as _route437
                        _route437.graph_begin(entry, args, kwargs, self.runtime_mode.name)
                        try:
                            output = self.runnable(*args, **kwargs)
                        finally:
                            _route437.graph_end()
""")
        return insert(src, '        entry.aclgraph.replay()\n', """        # EXTREME_RUN437_ROW_IDENTITY
        from scripts import loop079_row_capture as _route437
        _route437.graph_replay(entry, args, kwargs, self.runtime_mode.name)
""", before=True)
    if key == 'w4a8':
        return insert(src, '        topk_weights = topk_weights.to(x.dtype)\n', '''
        # EXTREME_RUN437_ROW_IDENTITY
        from scripts import loop079_row_capture as _route437
        _route437.route(layer, topk_ids, log2phy, enable_force_load_balance, self.dynamic_eplb)
''')
    if key == 'comm':
        return insert(src, '        token_dispatch_output = self.token_dispatcher.token_dispatch(token_dispatch_input=token_dispatch_input)\n', '''
        # EXTREME_RUN437_ROW_IDENTITY
        from scripts import loop079_row_capture as _route437
        _route437.dispatched(fused_experts_input.topk_ids, token_dispatch_output,
                             self.token_dispatcher, fused_experts_input.routing.expert_map)
''')
    if key == 'draft':
        return insert(src, '        return num_query_total, token_indices_to_sample, cad, None\n', '''        # EXTREME_RUN437_ROW_IDENTITY
        from scripts import loop079_row_capture as _route437
        _route437.draft_inputs(self, num_query_total, token_indices_to_sample, cad)
''', before=True)
    if key == 'runtime':
        src = insert(src, '                    target_output = self.target.execute(self.state)\n', '''                    # EXTREME_RUN437_ROW_IDENTITY
                    from scripts import loop079_row_capture as _route437
                    _route437.before_target(self)
''', before=True)
        src = insert(src, '                    target_output = self.target.execute(self.state)\n', '''                    # EXTREME_RUN437_ROW_IDENTITY
                    from scripts import loop079_row_capture as _route437
                    _route437.after_target(self)
''')
        src = insert(src, '            mark("acceptance")\n', '''            # EXTREME_RUN437_ROW_IDENTITY
            from scripts import loop079_row_capture as _route437
            _route437.after_acceptance(acceptance_output)
''')
        src = insert(src, '            with self._scope("extreme::proposer"):\n', '''                # EXTREME_RUN437_ROW_IDENTITY
                from scripts import loop079_row_capture as _route437
                _route437.draft_begin()
''')
        return insert(src, '            mark("proposer")\n', '''            # EXTREME_RUN437_ROW_IDENTITY
            _route437.draft_end(next_draft)
''', before=True)
    src = insert(src, '        cycles = 0\n', '''        # EXTREME_RUN437_ROW_IDENTITY
        from scripts import loop079_row_capture as _route437
        _route437.cohort_start(self.runtime)
''', before=True)
    return insert(src, '        counts_cpu = count_history[:cycles].cpu()\n', '''        # EXTREME_RUN437_ROW_IDENTITY
        _route437.export(self, counts_cpu, cycles)
''')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['check', 'install', 'restore'])
    ap.add_argument('--state-dir', type=Path)
    ap.add_argument('--record', type=Path, required=True)
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
        for helper in ('loop079_row_capture.py', 'loop079_row_capture_validate.py'):
            path = ROOT/'scripts'/helper
            compile(path.read_bytes(), str(path), 'exec')
        if args.action == 'install':
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
