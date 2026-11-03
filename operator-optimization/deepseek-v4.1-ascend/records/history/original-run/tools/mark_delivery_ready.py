"""Record P7 delivery while preserving scientific unknowns."""
from pathlib import Path
import datetime,json
T=Path(__file__).resolve().parents[1]
a=json.loads((T/'evidence/final-scope-review.json').read_text());assert a['status']=='PARTIAL_DELIVERY_ACCEPTED_WITH_EXPLICIT_UNKNOWNS' and len(a['requirements'])==7
p=T/'STATE.json';s=json.loads(p.read_text());s['state']='DIAGNOSTIC_DELIVERABLE_COMPLETE_WITH_GAPS';s['current_phase']='P7';s['last_update']=datetime.datetime.now(datetime.timezone.utc).isoformat();s['overall_work_paused']=False;s['goal_tool_state_last_checked']=s.pop('goal_tool_state',None);s['goal_tool_state_note']='Historical snapshot only; query goal tool for current status.';s['next_read_only_action']='No further work required for this diagnosis workflow; unresolved graph, data and hardware facts remain explicitly UNKNOWN in final-scope-review.json.';p.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'state':s['state'],'phase':'P7','not_proven':a['not_proven']}))
