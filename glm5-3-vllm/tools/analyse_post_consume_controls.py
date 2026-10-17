#!/usr/bin/env python3
"""Join post-D native controls, retain content counterexamples and separate ABBA arms."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import tarfile


def archive(path, snapshot):
    rows, layouts, tests = [], {}, {}
    with tarfile.open(path) as tar:
        prefix = f'consumer-audits/{snapshot}/kv-control-audit/'
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = member.name
            if name.startswith(prefix) and name.endswith('.jsonl'):
                rows.extend(json.loads(line) for line in tar.extractfile(member) if line.strip())
            elif name.startswith(prefix) and '/layout-' in name:
                layout = json.load(tar.extractfile(member))
                layouts[layout['tp_rank']] = layout
            elif name.startswith('tests/') and name.endswith('.json'):
                tests[name] = json.load(tar.extractfile(member))
    assert set(layouts) == set(range(16)), 'Expected16 layouts'
    return rows, layouts, tests


def select(rows, request_id):
    return [r for r in rows if r['phase'] == 'native_prefix' and r['request_id'].startswith(request_id+'-')]


def ranked(rows):
    result = defaultdict(list)
    for row in rows:
        result[row['tp_rank']].append(row)
    assert set(result) == set(range(16))
    assert all(len(v) == 31 for v in result.values())
    for values in result.values():
        assert len({r['key'] for r in values}) == 31
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Never overwrite analysis evidence'
    source, layouts, tests = archive(args.source, args.snapshot)
    target, target_layouts, _ = archive(args.target, args.snapshot)
    prefix = f'tests/{args.run_id}/'
    run = tests[prefix+'result.json']
    assert run['completed']
    native = ranked(select(target, run['D_native_control_id']))
    waves, details = [], []
    for wave in run['waves']:
        p = ranked(select(source, wave['P_native_id']))
        d = ranked(select(target, wave['D_local_id']))
        p_keys = [r['key'] for r in p[0]]
        assert all([r['key'] for r in v] == p_keys for v in p.values())
        assert all([r['key'] for r in v] == p_keys for v in d.values())
        summary = dict(wave, source_replica_divergent_blocks=[], target_replica_divergent_blocks=[],
                       target_control_mismatch_blocks=[], changed_entry_rank_count=0,
                       zero_target_changed_entry_rank_count=0, zero_target_changed_bytes_per_rank={})
        for block, key in enumerate(p_keys):
            ps = {json.dumps(v[block]['checksums']) for v in p.values()}
            ds = {json.dumps(v[block]['checksums']) for v in d.values()}
            if len(ps) != 1:
                summary['source_replica_divergent_blocks'].append(block)
            if len(ds) != 1:
                summary['target_replica_divergent_blocks'].append(block)
            changed = []
            for rank in range(16):
                sr, dr, nr = p[rank][block], d[rank][block], native[rank][block]
                assert sr['sizes'] == dr['sizes'] == nr['sizes']
                assert len(sr['checksums']) == len(dr['checksums']) == len(nr['checksums']) == 115
                for entry, (sc, dc, nc) in enumerate(zip(sr['checksums'], dr['checksums'], nr['checksums'])):
                    if sc == dc:
                        continue
                    summary['changed_entry_rank_count'] += 1
                    zero = dc[0] == 0
                    if zero:
                        assert dc == [0, 0, 0, 0]
                        summary['zero_target_changed_entry_rank_count'] += 1
                        k = str(rank)
                        summary['zero_target_changed_bytes_per_rank'][k] = summary['zero_target_changed_bytes_per_rank'].get(k, 0)+sr['sizes'][entry]
                    changed.append({'rank':rank,'entry':entry,'bytes':sr['sizes'][entry],
                                    'source_control':sc,'target_control':dc,'D_native_control':nc,
                                    'source_matches_D_native':sc == nc,
                                    'target_zero':zero,'labels':target_layouts[rank]['groups']['0'][entry]['labels']})
            if changed:
                summary['target_control_mismatch_blocks'].append(block)
                details.append({'wave':wave['wave'],'source_fence_selected':wave.get('source_fence_selected'),
                                'block':block,'token_start':block*128,'token_end_inclusive':block*128+127,
                                'key':key,'changed_entries':changed})
        summary['content_controls_match'] = not summary['target_control_mismatch_blocks']
        waves.append(summary)
    arms = {}
    for value, name in [(False, 'unfenced'),(True, 'fenced')]:
        selected = [r for r in waves if r.get('source_fence_selected') is value]
        if selected:
            arms[name] = {'requests':len(selected),'content_control_matches':sum(r['content_controls_match'] for r in selected),
                          'output_matches_P_native':sum(r['pool_matches_P_native'] for r in selected),
                          'changed_entry_rank_count':sum(r['changed_entry_rank_count'] for r in selected),
                          'zero_target_changed_entry_rank_count':sum(r['zero_target_changed_entry_rank_count'] for r in selected)}
    result = {'completed':True,'run_id':args.run_id,'waves':waves,'arms':arms,'differences':details,
              'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.source,args.target]},
              'scope':'Post-D native-prefix four integer moments on logical unsigned bytes, not cryptographic payload equality. Zero unsigned-byte sum proves zero logical bytes in that observed view. D-native comparison is by documented sequential token-block order, not salt-dependent key. Cannot alone distinguish PUT, GET or later overwrite. Performance is not measured.'}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'arms':arms,'waves':[{'wave':r['wave'],'fenced':r.get('source_fence_selected'),'matching':r['content_controls_match'],'bad_blocks':r['target_control_mismatch_blocks'],'changed_entry_rank_count':r['changed_entry_rank_count'],'zero_bytes_per_rank':r['zero_target_changed_bytes_per_rank']} for r in waves]},indent=2))


if __name__ == '__main__':
    main()
