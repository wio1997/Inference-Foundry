#!/usr/bin/env python3
"""Audit Run99 formal client waves against preserved rank0 cohort decode wall."""
import argparse,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'evidence/20260924_loop036_metadata/run99'
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
 out=[]
 for ri in range(1,4):
  bench=json.loads((BASE/f'bench48_{ri}.json').read_text())
  reqs=bench['requests'];assert len(reqs)==48
  rows=[]
  for wi in range(4):
   cohort=4+(ri-1)*4+wi+1
   sub=reqs[wi*12:(wi+1)*12]
   row=json.loads((BASE/f'runtime/rank0_cohort{cohort}.json').read_text())
   assert row['pass'] and row['cohort']==cohort
   client_wall=max(r['end'] for r in sub)-min(r['start'] for r in sub)
   decode=row['wall_seconds']
   rows.append(dict(wave=wi+1,cohort=cohort,client_wall_s=client_wall,
                    rank0_decode_wall_s=decode,client_minus_rank0_decode_s=client_wall-decode,
                    cycles=row['cycles'],mean_ttft_s=statistics.mean(r['ttft_ms'] for r in sub)/1000,
                    max_ttft_s=max(r['ttft_ms'] for r in sub)/1000,
                    mean_tpot_ms=statistics.mean(r['tpot_ms'] for r in sub)))
  out.append(dict(formal_run=ri,duration_s=bench['summary']['duration_s'],waves=rows))
 result=dict(source='Run99 formal saved bench48 and rank0 cohort files; no new serving measurement',runs=out,
             caution='Client wave minus rank0 decode is an observed window difference, not identified prefill or removable idle; other ranks, overlap and publication are included.')
 Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
 for x in out:
  print('run',x['formal_run'],[(r['cohort'],round(r['client_minus_rank0_decode_s'],3),round(r['mean_ttft_s'],3),r['cycles']) for r in x['waves']])
if __name__=='__main__':main()
