from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip, json, hashlib, statistics, socket

assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
A=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030')
ROOT=A/'observer-20261030T160315Z-76b79f0f/value-review'
paths=[ROOT/n for n in ['summary.json','cohorts.json','snapshot-phase-index.json','decoded-records.jsonl.gz']]
stats_before={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in paths}
source_sha={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
summary=json.loads(paths[0].read_text())
assert summary['decoded_snapshots']==4736 and summary['cohorts_exact_per_expert_matches']==296 and summary['decode_errors']==0 and summary['cohorts_mismatch']==0
cohorts=json.loads(paths[1].read_text())
phase_rows=json.loads(paths[2].read_text())
phase_by_index={r['index']:r for r in phase_rows}
assert len(phase_by_index)==len(phase_rows)==4736
groups=defaultdict(list)
cohort_phase=defaultdict(set)
with gzip.open(paths[3],'rt') as f:
    for line in f:
        r=json.loads(line)
        p=phase_by_index[r['index']]
        assert (r['role'],r['layer'],r['ordinal'],r['ep_rank'])==(p['role'],p['layer'],p['ordinal'],p['rank'])
        assert r['receiver_count_sum']==sum(r['receiver_expert_counts'])
        key=(r['role'],r['layer'],r['ordinal'])
        cohort_phase[key].add(p['phase'])
        groups[(r['role'],p['phase'])].append(r)
assert len(cohort_phase)==296 and all(len(p)==1 for p in cohort_phase.values())


def stat(v):
    if not v:return None
    vals=sorted(v)
    return dict(n=len(vals),min=vals[0],median=statistics.median(vals),mean=statistics.mean(vals),p90=vals[(9*(len(vals)-1))//10],max=vals[-1])


cohort_by_key={(c['role'],c['layer'],c['ordinal']):c for c in cohorts}
results=[]
for (role,phase),rs in sorted(groups.items()):
    keys=sorted({(r['role'],r['layer'],r['ordinal']) for r in rs})
    cs=[cohort_by_key[k] for k in keys]
    assert all(c['complete_membership_and_rank_coverage'] and not c['ambiguous_duplicate_records'] and c['per_expert_histogram_match'] for c in cs)
    counts=[n for r in rs for n in r['receiver_expert_counts']]
    ratios=[c['receiver_sum_max']/c['receiver_sum_mean'] for c in cs if c['receiver_sum_mean']>0]
    results.append(dict(role=role,phase=phase,snapshots=len(rs),cohorts=len(cs),layers=len({r['layer'] for r in rs}),
                        M_hist=dict(Counter(r['M'] for r in rs)),valid_sender_rows=stat([r['valid_rows'] for r in rs]),
                        K_values=sorted({r['K'] for r in rs}),E_local_values=sorted({r['E_local'] for r in rs}),E_global_values=sorted({r['E_global'] for r in rs}),
                        per_expert_receiver_rows=stat(counts),zero_expert_observations=sum(n==0 for n in counts),
                        receiver_rank_total_rows=stat([r['receiver_count_sum'] for r in rs]),
                        per_cohort_receiver_rank_max_over_mean=stat(ratios),
                        capacity_arguments=sorted({r['capacity_max_output_size'] for r in rs}),
                        rank_receiver_sum_above_capacity_observations=sum(r['receiver_count_sum_above_capacity_argument'] for r in rs),
                        use_mega_moe_values=sorted({r['use_mega_moe'] for r in rs}),
                        enable_fused_mc2_values=sorted({r['enable_fused_mc2'] for r in rs})))
for p in paths:assert (p.stat().st_size,p.stat().st_mtime_ns)==stats_before[str(p)]
result=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_run=str(ROOT.parent),source_sha256=source_sha,
            source_stats_unchanged=True,snapshots=4736,cohorts=296,phase_counts=dict(Counter(next(iter(p)) for p in cohort_phase.values())),groups=results,
            no_device_or_inference_work=True,old169_profile_timing_join=False,overall_complete=False,
            limits=['Counts are observed166 same-call snapshot values; cannot attach old169 task duration, native wait timing or source tiling to them.',
                    'Prefix seeds, formal C8 and captured replay phase-unknown samples remain separate.',
                    'Expert/rank row-count skew is not elapsed-time skew; zero rows do not make native synchronization protocol removable.',
                    'Repeated call/rank/expert observations are descriptive populations, not independent trials.',
                    'Capacity comparisons cover observed receiver count sums, not universal safety or an exact memory-capacity ABI proof.'])
out=A/'profile-readonly-20261031/observed-moe-counts-v7.json'
assert not out.exists(),'refuse_overwrite'
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
result['full_result']=str(out)
print(json.dumps(result,ensure_ascii=False))
