#!/usr/bin/env python3
"""Analyze legal Run154 single-cohort first-token boundary and tail."""
import json,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1]
base=root/'evidence/20260925_loop045_boundary/run154'
bench=json.loads((base/'bench.json').read_text())
assert bench['summary']['n']==12 and bench['summary']['success']==12 and bench['summary']['fail']==0
assert bench['summary']['max_tokens']==1024 and bench['summary']['concurrency']==12
client_start=min(r['start'] for r in bench['requests'])
client_end=max(r['end'] for r in bench['requests'])
max_ttft=max(r['ttft_ms'] for r in bench['requests'])/1000
rows=[]
for rank in range(8):
 b=json.loads((base/f'boundary/rank{rank}_cohort1.json').read_text())
 r=json.loads((base/f'runtime/rank{rank}_cohort1.json').read_text())
 assert r['pass'] and r['generated_output_counts']==[1024]*12
 assert len(b['counts_cpu'])==r['cycles'] and all(len(x)==12 for x in b['counts_cpu'])
 parks={slot:p['cycle'] for p in b['park_log'] for slot in p['slots']}
 assert set(parks)==set(range(12))
 cycles=r['cycles']
 parked_slot_cycles=sum(max(0,cycles-p-1) for p in parks.values())
 rows.append({'rank':rank,'calls':len(b['calls']),'prefill_scheduled_tokens':[x['scheduled_tokens'] for x in b['calls'][:-1]],
  'client_to_first_execute_s':b['calls'][0]['t_ns']/1e9-client_start,
  'first_execute_to_handoff_s':(b['handoff_ns']-b['calls'][0]['t_ns'])/1e9,
  'handoff_to_build_start_s':(b['build_start_ns']-b['handoff_ns'])/1e9,
  'build_s':(b['build_end_ns']-b['build_start_ns'])/1e9,
  'build_end_to_serve_s':(b['serve_start_ns']-b['build_end_ns'])/1e9,
  'serve_s':(b['serve_end_ns']-b['serve_start_ns'])/1e9,
  'serve_end_to_publication_s':(b['publication_ns']-b['serve_end_ns'])/1e9,
  'publication_to_client_end_s':client_end-b['publication_ns']/1e9,
  'cycles':cycles,'parked_slot_cycles':parked_slot_cycles,
  'parked_fraction':parked_slot_cycles/(cycles*12),
  'first_park_cycle':min(parks.values()),'last_park_cycle':max(parks.values()),
  'counts_total':sum(sum(c) for c in b['counts_cpu'])})
def desc(v):return {'median':statistics.median(v),'min':min(v),'max':max(v)}
fields=('client_to_first_execute_s','first_execute_to_handoff_s','handoff_to_build_start_s','build_s','build_end_to_serve_s','serve_s','serve_end_to_publication_s','publication_to_client_end_s','parked_fraction')
out={'run':'run154','contract':'8x910B3 DP1TP8 DSpark7 legal 12 requests exact max_tokens1024, cold service diagnostic; not formal E2E',
 'client_envelope_s':client_end-client_start,'client_max_ttft_s':max_ttft,
 'bench_summary':bench['summary'],'rank_summary':{k:desc([r[k] for r in rows]) for k in fields},
 'rank_rows':rows,
 'patch_restore':json.loads((base/'patch_restore.json').read_text()),
 'interpretation':'Cold-cache one-cohort diagnostic, so TTFT and TPS cannot be directly compared with formal warm-cache Run99. Same-host perf_counter aligns client/worker. Build_extreme_runtime takes ~40ms, while pre-handoff execute calls account for ~6s. This rejects a multi-second per-cohort runtime-construction explanation in the cold case. Parked slot-cycles measure fixed-shape waste exposure, not achievable compaction speedup. EngineDeadError appears after 12/12 completed and service shutdown; keep teardown log.'}
p=base/'boundary_analysis.json';p.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('rank_rows','patch_restore')},indent=2))
