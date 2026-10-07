"""One D-only current FULL-path profile, no mode/model/config changes."""
import importlib.util,json,shlex,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PLAN=json.loads((ROOT/'functional_plan.json').read_text())
s=importlib.util.spec_from_file_location('fixed_complete_PD_client',Path(PLAN['matched_root'])/'pd_compare.py')
c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
d=c.d;m=c.m

def inventory():
    code="from pathlib import Path;import json;r=Path(%r);print(json.dumps({'path':str(r),'files':[{'path':str(p.relative_to(r)),'size':p.stat().st_size} for p in r.rglob('*') if p.is_file()] if r.exists() else []}))"%PLAN['profile_root']
    p=subprocess.run(['ssh','-o','BatchMode=yes','root@172.16.10.167',shlex.join(['/usr/bin/python3','-c',code])],capture_output=True,text=True,timeout=45)
    assert p.returncode==0,p.stderr[-1000:]
    return json.loads(p.stdout)

def profile_control(action):
    t=time.monotonic_ns()
    status,raw=d.fetch('http://172.16.10.167:9900/'+action+'_profile',{},timeout=300)
    row=dict(action=action,status=status,body=raw.decode(errors='replace'),client_wall_s=(time.monotonic_ns()-t)/1e9,
        dispatch_scope='native EngineCore Executor.profile collective RPC all16; successful API completion, not independent in-memory introspection')
    m.write(action+'_profile.json',row)
    assert status==200,row
    return row

def workflow():
    assert json.loads((Path(PLAN['matched_root'])/'state.json').read_text())['status']=='completed'
    assert json.loads((Path(PLAN['matched_root'])/'decision.json').read_text())['verdict']=='INCONCLUSIVE'
    owner=json.loads(Path('/data/tiankuan/wio/glm52-pd/controller-owner.json').read_text())
    assert owner['run_id']==PLAN['run_id'] and owner['status']=='running' and m.live(owner['owner'])
    assert d.rpc('167','epoch')['epoch']=='candidate'
    c.ROOT=d.x.ROOT=m.ROOT=ROOT
    d.x.PLAN=dict(d.PLAN,run_id=PLAN['run_id']);d.oldclient.PLAN=d.x.PLAN
    guards={h:d.rpc(h,'guard' if h=='166' else 'guardfresh') for h in ('166','167')};m.write('guards_before.json',guards)
    witness=d.rpc('167','witness',0);m.write('witness_before.json',witness)
    assert {z['transition'] for z in witness['H11_rows']}=={PLAN['transition']}
    assert len(witness['H11_rows'])==16 and all(len(z['rows'])==2 for z in witness['H11_rows'])
    # Warm from Run275 consumed both heavy observer snapshots at this unchanged transition.
    initial=inventory();m.write('profile_inventory_before.json',initial);assert not initial['files'],'unexpected prior traces; do not overwrite or merge'
    attempted=False;stopped=False;results=[]
    try:
        attempted=True
        profile_control('start')
        result=c.measured_request('profile_on_complete',True)
        # Generic accepted client labels profiler_active=false; preserve raw result
        # and give the actual surrounding control state explicitly here.
        m.write('profiled_request_context.json',dict(profiler_active=True,only_D=True,label='profile_on_complete',not_performance_AB=True))
        results.append(result)
    except BaseException as error:
        m.write('failure.json',dict(error=repr(error),phase='profile_on',results=results));raise
    finally:
        if attempted:
            try:
                profile_control('stop');stopped=True
            except BaseException as error:
                m.write('profile_stop_failure.json',dict(error=repr(error),off_verified=False));raise
    assert stopped
    post=c.measured_request('profile_off_complete',True);results.append(post)
    assert results[0]['workload_signature']==post['workload_signature']
    final_witness=d.rpc('167','witness',0);assert final_witness['H11_rows']==witness['H11_rows']
    final_guards={h:d.rpc(h,'guard' if h=='166' else 'guardfresh') for h in ('166','167')}
    assert all(final_guards[h]['root']==guards[h]['root'] and final_guards[h]['worker_start_ticks']==guards[h]['worker_start_ticks'] for h in guards)
    after=inventory();assert after['files'],'profile acknowledged but no output evidence'
    m.write('profile_inventory_after.json',after);m.write('retained_witness.json',final_witness);m.write('guards_after.json',final_guards)
    m.write('retained_stack.json',dict(H6=True,H5=True,H11=False,H11_mode=0,H9=False,H10=False,pure_KV_API_repairs=True,profiler_off_ack=True,normal_reload=0,parameter_scan=False,Current=None,formal_product_KEEP=False))
    m.write('diagnostic_status.json',dict(status='completed',results=results,profile_only_D=True,profiler_off_ack=True,all16_transfer=True,trace_attribution_pending=True,performance_gain=None,Current=None))

if __name__=='__main__':
    assert sys.argv[1]=='workflow';workflow()
