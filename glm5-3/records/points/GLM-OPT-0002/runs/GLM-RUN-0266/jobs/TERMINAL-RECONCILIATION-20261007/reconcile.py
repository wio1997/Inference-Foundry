"""One preplanned terminal request after cross-host pointer driver failure.

No D reload, source edit, candidate retry or change to original failed state.
"""
import hashlib,importlib.util,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
JOB=Path(__file__).resolve().parent
source=ROOT/'pd_stream_order_compare.py'
expected='a9266f74c09cc8b29e1a0040236bc38440e0a24f9d75a9c8197cddc3b4378f72'
assert hashlib.sha256((ROOT/'controller_spec.json').read_bytes()).hexdigest()==expected
state=json.loads((ROOT/'state.json').read_text());assert state['status']=='failed'
assert not (ROOT/'retained_mode_warm_request.json').exists(), 'never repeat an issued terminal request'
spec=importlib.util.spec_from_file_location('frozen_run266',source);x=importlib.util.module_from_spec(spec);spec.loader.exec_module(x)
owner=json.loads(Path('/data/tiankuan/wio/glm52-pd/controller-owner.json').read_text())
assert owner['run_id']=='GLM-RUN-0266-TERMINAL' and owner['status']=='running' and x.m.live(owner['owner'])
assert not x.m.live(state['owner']), 'original controller must be retired'
before={h:x.rpc(h,'guard' if h=='166' else 'guardfresh') for h in ('166','167')}
x.rpc('167','switch',0)
result=x.request('retained_mode_warm')
witness=x.rpc('167','witness',0)
after={h:x.rpc(h,'guard' if h=='166' else 'guardfresh') for h in ('166','167')}
assert all(before[h]['root']['pid']==after[h]['root']['pid'] and before[h]['root']['start_ticks']==after[h]['root']['start_ticks'] for h in before)
decision=json.loads((ROOT/'research_decision.json').read_text());assert not decision['research_positive']
stack=dict(H6=True,H5=True,H9=False,H8=False,H4=False,MC2_mode=1,event_mode=1,H9_mode=0,comparison_completed=True,pure_KV_API_repairs=True,heavy_observers=False,same_FULL_configuration=True,Current=None,formal_product_KEEP=False,original_controller_status='failed',terminal_reconciliation=str(JOB),normal_requests=21)
x.m.write('retained_witness.json',witness);x.m.write('guards_after.json',after);x.m.write('retained_stack.json',stack)
out=dict(passed=True,one_preplanned_terminal_request=True,D_reload=False,source_mutation=False,candidate_retry=False,same_workers=True,original_state_preserved=True,request=result,retained_stack=stack)
(JOB/'result.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'passed':True,'normal_requests':21,'H9':False,'H6':True,'H5':True,'same_workers':True,'original_Run266_status':'failed'}))
