#!/usr/bin/env python3
"""Rank-local summary of already clipped inter-forward window intervals."""
import csv, hashlib, json, statistics
from pathlib import Path

T = Path(__file__).resolve().parents[2]
gap = T / 'inter-forward-gap-analysis-v2/rank_gaps.csv'
families = T / 'inter-forward-gap-analysis-v2/task_families.csv'
g = {(r['rank'],int(r['target_forward_ordinal'])):r for r in csv.DictReader(gap.open()) if r['interior_runs_1_2_3']=='True'}
assert len(g)==48 and {k[1] for k in g}=={1,6,7}
f = {(r['rank'],int(r['target_forward_ordinal']),r['type']):r for r in csv.DictReader(families.open()) if (r['rank'],int(r['target_forward_ordinal'])) in g}
rows = []
for (rank, o), r in sorted(g.items()):
    total = float(r['local_gap_us']); known = float(r['known_device_task_union_us']); missing = float(r['no_task_in_exported_intervals_us'])
    assert total>0 and abs(known+missing-total)<0.002
    row = {'rank':rank,'target_forward_ordinal':o,'gap_us':total,'known_union_us':known,'no_exported_row_us':missing,
           'known_union_fraction':known/total,'no_exported_row_fraction':missing/total,'hcom_type_unions_us':{}}
    for t in ('hcom_allGather_','hcom_allReduce_','hcom_reduceScatter_','hcom_alltoall_'):
        z=f.get((rank,o,t)); val=float(z['clipped_union_us']) if z else 0.0
        assert val <= known+0.002
        row['hcom_type_unions_us'][t]=val
    rows.append(row)
summary=[]
for o in (1,6,7):
    group=[r for r in rows if r['target_forward_ordinal']==o];assert len(group)==16
    med=lambda fn: statistics.median(fn(r) for r in group)
    summary.append({'target_forward_ordinal':o,'rank_count':len(group),'gap_us_median':med(lambda r:r['gap_us']),
        'known_union_fraction_median':med(lambda r:r['known_union_fraction']),'no_exported_row_fraction_median':med(lambda r:r['no_exported_row_fraction']),
        'hcom_allReduce_union_us_median':med(lambda r:r['hcom_type_unions_us']['hcom_allReduce_']),
        'hcom_allReduce_fraction_median':med(lambda r:r['hcom_type_unions_us']['hcom_allReduce_']/r['gap_us']),
        'hcom_allGather_union_us_median':med(lambda r:r['hcom_type_unions_us']['hcom_allGather_']),
        'hcom_allGather_fraction_median':med(lambda r:r['hcom_type_unions_us']['hcom_allGather_']/r['gap_us'])})
result={'scope':'existing 16 rank local inter-forward bracket; target forward ordinal 1,6,7; previous layer39 MC2 end to next layer0 start',
        'sources_sha256':{str(p.relative_to(T)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (gap,families)},
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'summary':summary,'rank_rows':rows,
        'limitations':['Previous layer39 MC2 end is not prior full-forward end; windows include other work and are not scheduler delay.',
            'No exported kernel interval does not mean device idle or host delay.',
            'hcom per-type union includes any relevant CSV rows, possibly N/A/synthetic or duplicate underlying events; duration is not proven actual transfer/wait.',
            'Different hcom type unions may overlap; never add them as disjoint times.',
            'Fractions are computed per rank before medians; do not assume cross-rank clock synchronization or matching request IDs.']}
out=T/'agents/diagnosis/gap-communication-summary.json';assert not out.exists()
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'COMPLETED','output':str(out),'summary':summary},ensure_ascii=False))
