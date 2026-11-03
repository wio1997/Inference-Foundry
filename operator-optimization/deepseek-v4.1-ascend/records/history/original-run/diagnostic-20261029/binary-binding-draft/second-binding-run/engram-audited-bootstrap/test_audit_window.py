"""Synthetic offline audit verifies losses fail closed; no NPU."""
import sys
if sys.flags.optimize:raise RuntimeError('tests require assertions enabled')
from pathlib import Path
import json,tempfile
from audit_window import audit
with tempfile.TemporaryDirectory() as d:
 p=Path(d);run='a'*32;window='b'*32;inst='c'*24
 status={'run_id':run,'process_instance':inst,'pid':123,'status':'WINDOW_CLOSED','window_id':window,'closed_window':window,'inflight':0,'installation':{'status':'verified_before_install'},'state_incomplete':False,'calls_seen':1,'begin_written':1,'end_written':1}
 (p/('observer-'+inst+'.json')).write_text(json.dumps(status))
 events=[{'event':'embed_begin','process_instance':inst,'window_id':window,'call':1},{'event':'embed_end','process_instance':inst,'window_id':window,'call':1}]
 log=p/'engram-metadata.jsonl';log.write_text(''.join(json.dumps(e)+'\n' for e in events))
 m={'run_id':run,'window_id':window,'workers':[{'pid':123,'rank':8}]}
 assert audit(p,m)['workers'][0]['status']=='NO_LOSS_DETECTED_CONDITIONAL_SNAPSHOT'
 log.write_text(json.dumps(events[0])+'\n');x=audit(p,m)['workers'][0];assert x['status']=='UNKNOWN' and 'missing_or_duplicate_call_pair' in x['reasons']
 log.write_text(''.join(json.dumps(e)+'\n' for e in events)+json.dumps({'event':'trace_cap'})+'\n');assert 'trace_cap' in audit(p,m)['workers'][0]['reasons']
 status['state_incomplete']=True;(p/('observer-'+inst+'.json')).write_text(json.dumps(status));assert 'observer_tainted' in audit(p,m)['workers'][0]['reasons']
print(json.dumps({'status':'PASS','pair_loss':'UNKNOWN','cap':'UNKNOWN','tainted':'UNKNOWN','rank_graph':'not proved'}))
