#!/usr/bin/env python3
"""CPU synthetic checks of validator rejection paths; never touches NPU/service."""
import copy
import json
from pathlib import Path
import tempfile
from loop078_route_capture_validate import validate_cohort


def fixture():
    counts = [[2]*12 for _ in range(66)]
    data = []
    for rank in range(8):
        records = {}
        for c in (64, 65):
            positions = [10000+s*100+2*c+j for s in range(12) for j in range(8)]
            dpos = [positions[s*8]+2+j for s in range(12) for j in range(7)]
            records[str(c)] = dict(replayed_graphs=[[rank+1, 'bucket96']], target_positions=positions,
                target_input_ids=[10]*96, active_mask=[True]*12,
                raw_acceptance_counts=[2]*12, absolute_cycle=c,
                draft_output=[[10]*7 for _ in range(12)],
                draft_metadata=dict(q=7, sample_from_anchor=True, positions=dpos,
                    input_ids=[10]*84, sample_indices=list(range(84)),
                    query_start_loc=[s*7 for s in range(13)]))
            for model, layers, rows in [('target', range(43), 96), ('draft', range(43,46), 84)]:
                records[str(c)][model] = [dict(name=(f'model.layers.{i}.mlp.experts' if model=='target' else f'mtp.{i-43}.mlp.experts'),
                    ordinal=i, model=model, graph_key=([rank+1, 'bucket96'] if model=='target' else None), ids=[list(range(6)) for _ in range(rows)],
                    ep_rank=rank, group=[rows if rank==0 and e<6 else 0 for e in range(32)],
                    expert_map=[e-rank*32 if rank*32 <= e < (rank+1)*32 else -1 for e in range(256)])
                    for i in layers]
        data.append(dict(schema=1, rank=rank, cohort=1, cycles=66, start_cycle=0,
            counts=counts, remaining=[1024]*12, initial_output_counts=[1]*12,
            initial_positions=[10000+s*100 for s in range(12)],
            slot_identity='cohort-local slot; external request IDs unavailable', records=records))
    return data


def run(data):
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, row in enumerate(data):
            p = Path(tmp)/f'{i}.json'
            p.write_text(json.dumps(row))
            paths.append(p)
        return validate_cohort(paths)


def main():
    import os
    import sys
    from types import SimpleNamespace
    import loop078_route_capture as hooks
    for i in range(43):
        assert hooks.identity(SimpleNamespace(layer_name=f'model.layers.{i}.mlp.experts'))[1:] == (i, 'target')
    for i in range(3):
        assert hooks.identity(SimpleNamespace(layer_name=f'mtp.{i}.mlp.experts'))[1:] == (43+i, 'draft')
    try:
        hooks.identity(SimpleNamespace(layer_name='model.layers.43.mlp.experts'))
    except RuntimeError:
        pass
    else:
        raise AssertionError('ambiguous Target/Draft identity accepted')
    # Only a mock module is imported: this check cannot initialize an accelerator.
    sys.modules['torch'] = SimpleNamespace(npu=SimpleNamespace(is_current_stream_capturing=lambda: False))
    os.environ[hooks.ENV] = '/unused-selftest'
    hooks.route(None, None, None, False, False)
    hooks.CYCLE, hooks.PHASE = 64, 'draft'
    hooks.RECORDS = {64: {}}
    hooks.draft_end(SimpleNamespace(clone=lambda: 'snapshot'))
    assert hooks.RECORDS[64]['draft_output'] == 'snapshot' and hooks.PHASE is None
    # Two capture buckets have the same 43 layer identities; only replayed entry wins.
    class Clone:
        def clone(self):
            return 'snapshot'
    hooks.TARGET = {(bucket, i): dict(name=f'model.layers.{i}.mlp.experts', ordinal=i,
        model='target', graph_key=bucket, ids=Clone(), group=Clone(), expert_map=Clone())
        for bucket in ((1, 'bucketA'), (2, 'bucketB')) for i in range(43)}
    hooks.START = 0
    hooks.REPLAYED = [(2, 'bucketB')]
    hooks.after_target(SimpleNamespace(state=SimpleNamespace(cycle_index=64,
        target_positions=Clone(), target_input_ids=Clone(), active_mask=Clone())))
    assert len(hooks.RECORDS[64]['target']) == 43
    assert all(x['graph_key'] == (2, 'bucketB') for x in hooks.RECORDS[64]['target'])
    del os.environ[hooks.ENV]
    del sys.modules['torch']
    original = fixture()
    result = run(original)
    assert result['cycles']['64']['useful_rows'] == 24
    checks = ['explicit_target_draft_identity', 'eager_hook_no_capture', 'draft_end_clone', 'explicit_graph_replay_selection', 'valid_fixture']
    def reject(label, mutate):
        data = copy.deepcopy(original)
        mutate(data)
        try:
            run(data)
        except (ValueError, KeyError, TypeError):
            checks.append(label)
        else:
            raise AssertionError(f'accepted corrupt fixture: {label}')
    reject('wrong_graph', lambda d: d[0]['records']['64']['target'][0].__setitem__('graph_key', [999, 'unknown']))
    reject('missing_rank', lambda d: d.pop())
    reject('group_mismatch', lambda d: d[0]['records']['64']['target'][0]['group'].__setitem__(0, 0))
    reject('row_mismatch', lambda d: d[0]['records']['64']['target'][0]['ids'].pop())
    reject('layer_mismatch', lambda d: d[0]['records']['64']['target'].pop())
    reject('handoff_mismatch', lambda d: [x['records']['64']['draft_output'][0].__setitem__(0, 11) for x in d])
    reject('draft_position_mismatch', lambda d: [x['records']['64']['draft_metadata']['positions'].__setitem__(0, 0) for x in d])
    reject('masked_acceptance_mismatch', lambda d: [x['records']['64']['active_mask'].__setitem__(0, False) for x in d])
    reject('ep_mapping_mismatch', lambda d: d[0]['records']['64']['target'][0]['expert_map'].__setitem__(0, -1))
    print(json.dumps(dict(passed=True, checks=checks)))


if __name__ == '__main__':
    main()
