"""Draft only. Same-worker immutable RoPE layout comparison, conditional on actual correctness.

No normal model reload. Owned prior lifecycle is reused only for one declared
failure recovery, never by replaying the prior controller/workflow.
"""
import importlib.util
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT/'functional_plan.json').read_text())
DIAGNOSTIC = Path(PLAN['diagnostic_root'])
spec = importlib.util.spec_from_file_location('accepted_rope_diagnostic',DIAGNOSTIC/'pd_rope_correctness.py')
d = importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
m = d.m
cache_spec = importlib.util.spec_from_file_location('cache_counter_parser',PLAN['cache_collector'])
cache = importlib.util.module_from_spec(cache_spec);cache_spec.loader.exec_module(cache)
PODS = ['172.16.10.166:9081','172.16.10.167:9900']
collector = cache.HitRateCollector(PODS)


def measured_request(label, complete=False):
    before = collector.snapshot()
    m.write(label+'_cache_before.json',before)
    result = d.request(label,complete)
    after = collector.snapshot()
    m.write(label+'_cache_after.json',after)
    rates = collector.compute_hit_rate(before,after)
    m.write(label+'_cache_delta.json',rates)
    endpoints = rates['per_endpoint']
    assert set(endpoints) == set(PODS)
    rows = {pod:endpoints[pod]['dp0'] for pod in PODS}
    assert all(set(endpoints[pod]) == {'dp0'} for pod in PODS)
    if complete:
        validate_cache(rows,result['prompt_tokens'])
    request_id = json.loads((ROOT/(label+'_P.raw')).read_text())['kv_transfer_params']['remote_request_id']
    m.write(label+'_transfer.json',d.rpc('167','transfergate',label,request_id))
    result = dict(result,cache_by_endpoint=rows,all16_native_transfer=True)
    m.write(label+'_matched_result.json',result)
    return result


def validate_cache(rows,prompt_tokens):
    assert all(v['hbm_hits']==0 and v['hbm_queries']==prompt_tokens for v in rows.values()), 'fresh salt local queries/hits'
    assert rows[PODS[0]]['ext_queries']==prompt_tokens and rows[PODS[0]]['ext_hits']==0, 'P external query is admission volume; hits zero'
    assert rows[PODS[1]]['ext_queries']==rows[PODS[1]]['ext_hits']==prompt_tokens, 'D external KV transfer volume'


def decide(rows):
    assert len(rows)==4
    assert all(math.isfinite(row[key]) and row[key]>0 for row in rows for key in ('D_wall_s','PD_wall_s','P_wall_s','TPOT_ms'))
    a1,b1,a2,b2 = rows
    drift_d = abs(a1['D_wall_s']-a2['D_wall_s'])
    drift_pd = abs(a1['PD_wall_s']-a2['PD_wall_s'])
    differences=[]
    for a,b in ((a1,b1),(a2,b2)):
        differences.append(dict(D_saving_s=a['D_wall_s']-b['D_wall_s'],
            PD_saving_s=a['PD_wall_s']-b['PD_wall_s'],
            P_saving_s=a['P_wall_s']-b['P_wall_s'],
            TPOT_saving_ms=a['TPOT_ms']-b['TPOT_ms']))
    exact_work = all(row['workload_signature']==a1['workload_signature'] for row in rows)
    keys=('hbm_queries','hbm_hits','ext_queries','ext_hits')
    cache_signature=lambda row:{pod:{key:row['cache_by_endpoint'][pod][key] for key in keys} for pod in PODS}
    exact_cache=all(cache_signature(row)==cache_signature(a1) for row in rows)
    positive = exact_work and exact_cache and all(z['D_saving_s']>drift_d and z['PD_saving_s']>drift_pd and z['TPOT_saving_ms']>0 for z in differences)
    return dict(verdict='POSITIVE_LIMITED_RESEARCH' if positive else 'INCONCLUSIVE',
        all_complete_work_signatures_match=exact_work,all_complete_cache_query_work_matches=exact_cache,
        cache_signature=cache_signature(a1),D_A1_A2_drift_s=drift_d,PD_A1_A2_drift_s=drift_pd,
        comparisons=differences,H12_research_stack=positive,performance_scope='two repeated natural23 EOS complete standard-PD requests per path; not formal80K/600/93%',
        Current=None,formal_product_KEEP=False)


def effective_decision(evidence, terminal_ok, recovered, error):
    result=dict(evidence or {},measurement_verdict=evidence['verdict'] if evidence else None,
        H12_research_stack=False,terminal_verified=terminal_ok,recovery_used=recovered,
        terminal_error=error,Current=None,formal_product_KEEP=False)
    result['verdict']='INVALIDATED' if not evidence or recovered or error or not terminal_ok else evidence['verdict']
    result['H12_research_stack']=bool(evidence and evidence['H12_research_stack'] and terminal_ok and not recovered and not error)
    return result


def transition(witness):
    values={z['transition'] for z in witness['H12_rows']}
    assert len(witness['H12_rows'])==16 and len(values)==1
    return values.pop()


def unused_recovery():
    code="from pathlib import Path;import json;r=Path(%r);print(json.dumps({'unused':not any((r/'epochs/baseline_recovery'/n).exists() for n in ('launch.json','namespace_owner_167.json','native_167.log'))}))"%str(DIAGNOSTIC)
    p=subprocess.run(['ssh','-o','BatchMode=yes','root@172.16.10.167',shlex.join(['/usr/bin/python3','-c',code])],capture_output=True,text=True,timeout=30)
    assert p.returncode==0,p.stderr[-1000:]
    assert json.loads(p.stdout)['unused']


def recover():
    # The successful diagnostic must not already have consumed this epoch.
    row=d.rpc('167','epoch');assert row['epoch']=='candidate'
    m.write('failure_recovery_cleanup.json',d.rpc('167','recoverycleanup'))
    m.write('failure_recovery_restore.json',d.rpc('167','restore'))
    d.rpc('167','recoverymode');d.rpc('167','launch','baseline_recovery');d.ready()
    measured_request('H6_H5_recovery_warm')
    return dict(one_owned_recovery=True,H6=True,H5=True,H11=False,H12=False)


def workflow():
    state=json.loads((DIAGNOSTIC/'state.json').read_text());assert state['status']=='completed'
    proof=json.loads((DIAGNOSTIC/'correctness_reduced.json').read_text());assert proof['passed']
    owner=json.loads(Path('/data/tiankuan/wio/glm52-pd/controller-owner.json').read_text())
    assert owner['run_id']==PLAN['run_id'] and owner['status']=='running' and m.live(owner['owner'])
    assert d.rpc('167','epoch')['epoch']=='candidate'
    unused_recovery()
    d.x.ROOT = m.ROOT = ROOT
    d.x.PLAN = dict(d.PLAN,run_id=PLAN['run_id'])
    d.oldclient.PLAN = d.x.PLAN
    results=[];begun=False;retained=0;evidence=None;finished=False;comparison_error=None
    m.write('decision.json',dict(verdict='PENDING',H12_research_stack=False,Current=None,formal_product_KEEP=False))
    try:
        m.write('guards_before.json',{h:d.rpc(h,'guard' if h=='166' else 'guardfresh') for h in ('166','167')})
        begun=True
        last=PLAN['initial_transition']
        assert transition(d.rpc('167','witness',0))==last
        # Start with one fixed dense-layout warm so A1 has a new, owned transition.
        d.rpc('167','switch',1);measured_request('priming_layout_warm')
        witness=d.rpc('167','witness',1);assert transition(witness)==last+1;last=transition(witness)
        m.write('priming_layout_witness.json',witness);d.rpc('167','observeoff')
        for label,mode in (('A1',0),('B1',1),('A2',0),('B2',1)):
            d.rpc('166','guard');d.rpc('167','switch',mode)
            measured_request(label+'_warm')
            witness=d.rpc('167','witness',mode);assert transition(witness)==last+1;last=transition(witness)
            m.write(label+'_warm_witness.json',witness);d.rpc('167','observeoff')
            results.append(measured_request(label+'_complete',True))
            witness=d.rpc('167','witness',mode);assert transition(witness)==last
            m.write(label+'_complete_witness.json',witness)
        evidence=decide(results);m.write('comparison_evidence.json',evidence)
        retained=int(evidence['H12_research_stack']);finished=True
    except BaseException as error:
        comparison_error=repr(error)
        m.write('failure.json',dict(error=repr(error),results=results));raise
    finally:
        if begun:
            recovered=False;terminal_error=None;terminal_ok=False;witness=None;guards=None
            try:
                d.rpc('167','guardfresh');d.rpc('167','switch',retained)
                measured_request('retained_mode_warm')
                witness=d.rpc('167','witness',retained);d.rpc('167','observeoff')
                guards={h:d.rpc(h,'guard' if h=='166' else 'guardfresh') for h in ('166','167')}
                terminal_ok=True
            except Exception as error:
                terminal_error=repr(error);m.write('terminal_failure.json',dict(error=terminal_error))
                # Exactly one declared recovery, including one recovery warm.
                recovered=True;retained=0
                try:
                    m.write('failure_recovery.json',recover())
                    witness=d.rpc('167','witness',0)
                    guards={h:d.rpc(h,'guard' if h=='166' else 'guardfresh') for h in ('166','167')}
                    terminal_ok=True
                except Exception as recovery_error:
                    m.write('decision.json',effective_decision(evidence,False,True,repr(recovery_error)))
                    m.write('recovery_failure.json',dict(error=repr(recovery_error),first_error=terminal_error))
                    raise
            m.write('retained_witness.json',witness);m.write('guards_after.json',guards)
            decision=effective_decision(evidence,terminal_ok,recovered,comparison_error or terminal_error)
            stack=dict(H6=True,H5=True,H11=False,H12=bool(retained),H10=False,H9=False,H8=False,H4=False,
                MC2_mode=1,event_mode=1,H11_mode=0,H12_mode=retained,H9_mode=0,pure_KV_API_repairs=True,
                comparison_completed=finished,decision=decision,Current=None,formal_product_KEEP=False)
            m.write('retained_stack.json',stack)
            m.write('decision.json',decision)
            if finished and terminal_error:
                raise RuntimeError('terminal health failed; retain H6/H5 only pending attribution')
        if finished:
            m.write('comparison_status.json',dict(status='completed',results=results,decision=decision))


if __name__=='__main__':
    assert sys.argv[1]=='workflow'
    workflow()
