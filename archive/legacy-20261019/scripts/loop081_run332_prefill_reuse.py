#!/usr/bin/env python3
"""Read-only exact response-ID join of historical Run332 client and worker marks."""
import json
from pathlib import Path
ROOT=Path('/data/wio/Inference_Foundry/evidence')
OLD=ROOT/'20260926_loop074_refill/run332'
OUT=ROOT/'20260929_loop081_bound/run668/historical_run332_reuse.json'
def main():
 rows=[json.loads(s) for s in (OLD/'marks/rank0.jsonl').read_text().splitlines()]
 requests=json.loads((OLD/'measured48.json').read_text())['requests']
 result=[]
 for c,ready in enumerate(r for r in rows if r['kind']=='runtime_built'):
  ids={x.rsplit('-',1)[0] for x in ready['req_ids']}
  q=[x for x in requests if x['stream_id'] in ids]
  assert len(q)==12
  ex=[r for r in rows if r['kind']=='execute_entry' and r['t_ns']<=ready['t_ns']
      and any(x['req_id'].rsplit('-',1)[0] in ids for x in r['new']+r['cached'])]
  assert ex
  first=min(x['t_ns'] for x in ex);start=min(x['start']*1e9 for x in q)
  result.append({'cohort':c+5,'client_start_to_first_execute_ms':(first-start)/1e6,
     'first_execute_to_runtime_built_ms':(ready['t_ns']-first)/1e6,
     'execute_calls':len(ex),'scheduled_tokens_sum':sum(x['total_scheduled_tokens'] for x in ex)})
 OUT.write_text(json.dumps({'scope':'historical Run332 only; not same-run Run668 attribution','cohorts':result},indent=2)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
