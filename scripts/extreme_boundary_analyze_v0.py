#!/usr/bin/env python3
"""Common-clock phase accounting for one legal warmed 48-request boundary pass."""
import argparse,json,statistics
from pathlib import Path
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True);a=ap.parse_args()
 root=a.run_dir
 bench=json.loads((root/'measured48.json').read_text())
 reqs=bench['requests'];assert len(reqs)==48 and all(r['output_tokens']==1024 and not r['error'] for r in reqs)
 waves=[]
 for i in range(4):
  cohort=i+5
  sub=reqs[i*12:(i+1)*12]
  start=min(r['start'] for r in sub);end=max(r['end'] for r in sub)
  ranks=[json.loads((root/f'boundary/rank{k}_cohort{cohort}.json').read_text()) for k in range(8)]
  runtime=[json.loads((root/f'runtime/rank{k}_cohort{cohort}.json').read_text()) for k in range(8)]
  assert all(r['rank']==k and r['cohort']==cohort for k,r in enumerate(ranks))
  assert all(r['pass'] and r['cohort']==cohort for r in runtime)
  first=[]
  for r in ranks:
   candidates=[c['t_ns']/1e9 for c in r['calls'] if c['t_ns']/1e9 >= start-0.01]
   assert candidates, (cohort,r['rank'],'no execute after client start')
   first.append(min(candidates))
  marks={'client_start':start,'first_execute':max(first),
         'handoff':max(r['handoff_ns'] for r in ranks)/1e9,
         'serve_start':max(r['serve_start_ns'] for r in ranks)/1e9,
         'serve_end':max(r['serve_end_ns'] for r in ranks)/1e9,
         'publication':max(r['publication_ns'] for r in ranks)/1e9,
         'client_end':end}
  keys=list(marks)
  phases={f'{keys[j]}_to_{keys[j+1]}_s':marks[keys[j+1]]-marks[keys[j]] for j in range(len(keys)-1)}
  assert all(v>=-0.01 for v in phases.values()), (cohort,phases)
  waves.append(dict(wave=i+1,cohort=cohort,client_wall_s=end-start,marks_s=marks,
                    phases=phases,cycles=runtime[0]['cycles'],
                    decode_rank_wall_s=[r['wall_seconds'] for r in runtime],
                    mean_ttft_s=statistics.mean(r['ttft_ms'] for r in sub)/1000,
                    mean_tpot_ms=statistics.mean(r['tpot_ms'] for r in sub)))
 result=dict(contract='warmed legal 48x32K-to-1024 c12 diagnostic; boundary patch active; not formal E2E',
             bench_summary=bench['summary'],waves=waves,
             caution='Max-rank boundaries are same-host common-clock phase envelopes. Sequential differences do not isolate removable time and may include work overlap; use them for variation localization only.')
 (root/'phase_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
 for w in waves: print(w['cohort'],round(w['client_wall_s'],3),{k:round(v,3) for k,v in w['phases'].items()},w['cycles'])
if __name__=='__main__':main()
