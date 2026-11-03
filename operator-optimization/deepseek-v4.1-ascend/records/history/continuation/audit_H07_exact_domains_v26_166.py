from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib, json, socket

assert socket.gethostname() == 'bm-jn-zs-zone1-910C-64G-10-105'
M = Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18')
p = M / 'H07-external-embedding-copy-and-exact-replay-chains-v26-v2.json'
b = p.read_bytes()
assert hashlib.sha256(b).hexdigest() == 'e98f77e2b4da4b42c0f085f6d87eda1ce256751da2a4423dbbff5a454f174791'
d = json.loads(b)
out = M / 'H07-exact-root-domain-and-TP-cohort-audit-v26.json'
assert not out.exists()
rows = d['prepare_root_embedding_replay_rows']
assert len(rows) == 96 and len({r['root_key'] for r in rows}) == 96
assert len(d['unbound_external']) == 0
regimes = Counter()
for r in rows:
    n = r['metadata']['total_scheduled_tokens']
    es = r['external_embedding']
    copies = r['persistent_embedding_copy_APIs']
    assert len(es) == len(copies) == (1 if n else 0)
    for e in es:
        assert e['prepare_root_candidates'] == [r['root_key']]
        assert e['communication']['count'] == n * 5120
        assert e['communication']['rankSize'] == 8
        assert e['communication']['dtype'] == 'BFP16'
    for c in copies:
        assert c['root_key'] == r['root_key']
        assert c['input_shapes'] == f'{n},5120;\r\n{n},5120;\r\n'
    for g in r['graphs']:
        assert g['prepare_root_keys'] == [r['root_key']]
        assert len(g['prefix_communication']) == 1
        co = g['prefix_communication'][0]
        assert (co['count'], co['rankSize'], co['dtype']) == (122880, 8, 'BFP16')
        gather = [t for t in g['prefix_compute'] if t['definition']['opType_text'] == 'GatherV2']
        assert len(gather) == 1
        assert gather[0]['definition']['outputShapes_text'] == '"24,5120"'
    regimes[(n, len(r['graphs']), len(es), len(copies))] += 1
cohorts = []
for c in d['exact_TP_cohorts']:
    ms = c['members']
    ranks = sorted(m['ep_rank'] for m in ms)
    assert ranks in [list(range(8)), list(range(8, 16))]
    assert len({m['scheduled_rows'] for m in ms}) == 1
    assert len({tuple(m['external_comm_groups']) for m in ms}) == 1
    assert all(m['external_embeddings'] == m['copies'] == 1 for m in ms)
    cohorts.append(dict(key=c['key'], ranks=ranks,
        scheduled_rows=ms[0]['scheduled_rows'],
        communication_groups=ms[0]['external_comm_groups'],
        members=[dict(root_key=m['root_key'], database_index=m['database_index'],
                      graph_ordinal=m['graph_ordinal'], ep_rank=m['ep_rank']) for m in ms]))
assert len(cohorts) == 8
primitive_chain = 'd0785d253009066d61f2cf846d32e49cd0d2e6e8707de40c05657da9cc9bfaea'
primitive_apis = [r for r in d['native_embedding_APIs'] if r['API']['chain_key'] == primitive_chain]
assert len(primitive_apis) == 88
primitive_roots = []
for a in primitive_apis:
    rr = a['root_links']['same_execute_marker_prepare_roots']
    assert len(rr) == 1
    primitive_roots += rr
assert len(set(primitive_roots)) == 88
summary = dict(
    created_utc=datetime.now(timezone.utc).isoformat(),
    source=dict(path=str(p), sha256=hashlib.sha256(b).hexdigest()),
    roots=len(rows), external_embedding_and_copy_roots=88,
    decode_graph_bound_roots=64, nonempty_without_bound_graph_roots=24,
    empty_roots=8, phase_unbound_graphs=16,
    root_regimes=[dict(scheduled_rows=k[0], graphs=k[1], external_embeddings=k[2], copies=k[3], roots=v)
                  for k,v in regimes.items()],
    primitive_line85_API_records=len(primitive_apis), primitive_unique_roots=len(set(primitive_roots)),
    native_embedding_task_types=dict(Counter(t['definition']['opType_text'] for t in d['native_embedding_tasks'])),
    exact_cohorts=cohorts,
    all80_graph_prefix_gather_domains=dict(Counter(t['definition']['outputShapes_text']
         for g in d['graph_prefixes'] for t in g['prefix_compute'] if t['definition']['opType_text']=='GatherV2')),
    copy_native_TASK_records=len(d['copy_native_tasks']),
    caveats=['No copy native TASK link here does not imply no device copy: selected copy_ API is a different layer from native flow IDs.',
             'Exact cohort schedules and observed collectives do not prove future cached-entry guards are rank-uniform.',
             'Same-root association is control lineage, not tensor-pointer identity, same-value proof or removable critical-path time.',
             'Nonempty no-bound-graph roots are long-input schedules; preserve eager/prefill conditions rather than label every graphless operation removable.'],
    no_new_source_DB_model_device_or_service_work=True, overall_complete=False)
blob = (json.dumps(summary, ensure_ascii=False, separators=(',', ':'))+'\n').encode()
out.write_bytes(blob)
print(json.dumps(dict(result=summary, full_result=str(out), result_sha256=hashlib.sha256(blob).hexdigest()), ensure_ascii=False))
