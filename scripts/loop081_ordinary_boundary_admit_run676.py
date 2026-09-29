"""Reject missing composition evidence, even when serving itself succeeded."""
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);boundary=root/'boundary'
read=lambda p:json.loads(p.read_text())
core=[json.loads(x) for x in (boundary/'core.jsonl').read_text().splitlines()]
client=read(root/'bench48_1.json')['requests'];assert len(client)==48
assert len({r['response_id'] for r in client})==48 and all(r['response_id'] for r in client)
ids=set()
for c in range(5,9):ids.update(read(root/'runtime'/f'rank0_cohort{c}.json')['req_ids'])
assert len(ids)==48
required=('socket_add_decoded','preprocess_begin','preprocess_end','queue_put_begin','queue_put_end')
external={}
for rid in ids:
 rows={kind:[r for r in core if r['kind']==kind and r.get('req_id')==rid] for kind in required}
 assert all(len(v)==1 for v in rows.values()),rid
 times=[rows[k][0]['t_ns'] for k in required];assert times==sorted(times)
 external[rid]=rows['socket_add_decoded'][0]['external_req_id']
assert set(external.values())=={r['response_id'] for r in client}
submits={r['submit_id']:r for r in core if r['kind']=='schedule_end_submit'}
assert len(submits)==sum(r['kind']=='schedule_end_submit' for r in core)
for rank in range(8):
 events=[]
 for c in range(5,9):
  packet=read(boundary/f'rank{rank}_cohort{c}.json');ee=packet['events'];events+=ee
  for kind in ('handoff_entry','runtime_built','runtime_begin','runtime_end'):
   rows=[r for r in ee if r['kind']==kind];assert len(rows)==1,(rank,c,kind)
  rr=[next(r for r in ee if r['kind']==kind) for kind in ('handoff_entry','runtime_built','runtime_begin','runtime_end')]
  assert [r['t_ns'] for r in rr]==sorted(r['t_ns'] for r in rr)
  assert len({r['submit_id'] for r in rr})==1
 for r in events:assert r['submit_id'] in submits
 for start,end in [('target_begin','target_end'),('seed_begin','seed_end')]:
  begins=[r for r in events if r['kind']==start];ends=[r for r in events if r['kind']==end]
  assert begins and len(begins)==len(ends),(rank,start)
  assert len({r['submit_id'] for r in begins})==len(begins),(rank,start,'duplicate')
  for b in begins:
   ee=[r for r in ends if r['submit_id']==b['submit_id']];assert len(ee)==1 and ee[0]['t_ns']>=b['t_ns']
 flush=[json.loads(x) for x in (boundary/f'rank{rank}_flush.jsonl').read_text().splitlines()]
 assert {x['cohort'] for x in flush}==set(range(5,9)) and len(flush)==4
 assert all(x['end_ns']>=x['begin_ns'] for x in flush)
print(json.dumps({'pass':True,'exact_client_core_ids':48,'all8_boundary_cohorts':4,'core_submissions':len(submits),'scope':'diagnostic composition; no performance promotion'}))
