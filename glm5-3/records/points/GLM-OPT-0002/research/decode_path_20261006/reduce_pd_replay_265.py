"""Saved-input Run265 audit: canonical error, clean transfer, dynamic replay."""
from pathlib import Path
import ast,hashlib,json,sys

def main(root):
    identities={}
    def read(name):
        raw=(root/name).read_bytes();identities[name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)};return json.loads(raw)
    state=read('state.json');assert state['status']=='completed' and state['completed_stages']==['emptyterminalreplay']
    plan=read('functional_plan.json');retained=read('retained_stack.json');guards=read('guards_after.json');before=read('guards_before.json')
    assert guards['166']['root']==before['166']['root'] and guards['167']['root']!=before['167']['root']
    assert all(z['health']==200 and z['idle'] and len(z['device_owners'])==16 for z in guards.values())
    assert all(retained[z] for z in ('H6','H5','graph_diagnostic_supported','temporary_graph_configuration','observer_resident','kv_identity_patch','api_empty_terminal_patch'))
    assert not any(retained[z] for z in ('H4','H8','performance_claim')) and retained['Current'] is None
    native=read('retained_witness.json');assert native['H6']['all_rank_witness'] and len(native['H6']['rows'])==16
    assert all(z['dispatch_cached_true'] and z['combine_cached_true'] and z['mode']==1 and z['library_sha256']==plan['native_candidate']['sha256'] for z in native['H6']['rows'])
    assert len(native['H5_rows'])==16 and all(z['mode']==1 and not z['configured_overlap'] and z['returned_none'] for z in native['H5_rows'])
    fault=read('kv_fault_events.json');assert fault[-1] is None
    errors=[e['error'] for e in fault if e and 'error' in e];assert len(errors)==1 and errors[0]['code']==500 and errors[0]['type']=='InternalServerError'
    assert not any(c.get('token_ids') for e in fault if e for c in e.get('choices',[]))
    decision=read('kv_fault_decision.json');assert decision['fail_closed'] and decision['no_target_or_draft_execution'] and decision['all_rank_observed']
    fault_rows=read('kv_fault_transfer_witness.json');assert {z['rank'] for z in fault_rows}==set(range(16))
    r5=next(z for z in fault_rows if z['rank']==5)
    ids=[e['ids'] for e in r5['events'] if e['scope']=='kv_fault' and e['kind']=='invalid_report' and e['ids']]
    assert ids==[[1,2]],ids
    failed_graphs=read('kv_fault_graph_witness.json');assert len(failed_graphs)==32
    assert all(not any(k.startswith('kv_fault:') for k in row['counts']) for row in failed_graphs)
    details=[]
    for label,prompt_tokens,completion_tokens in [('graph_short',2334,8),('graph_complete',58,23)]:
        result=read(label+'_result.json');events=read(label+'_events.json');assert events[-1]['value'] is None
        token_ids=[t for e in events if e['value'] for c in e['value'].get('choices',[]) for t in c.get('token_ids',[])]
        assert result['prompt_tokens']==prompt_tokens and result['completion_tokens']==completion_tokens
        assert token_ids==result['token_ids'] and sum(v for k,v in result['external_KV_delta'].items() if 'hits_total' in k)==prompt_tokens
        if label=='graph_complete':
            assert result['semantic_accepted'] and result['finish_reason']=='stop'
            assert token_ids==plan['complete_golden']['token_ids'] and result['final_content']==plan['complete_golden']['final_content']
        else:
            golden=json.loads((root.parent/'GLM-RUN-0262/correctness_mode0_short_result.json').read_text())
            assert token_ids==golden['token_ids'] and result['finish_reason']=='length'
        work=read(label+'_workload.json');assert all(work['checks'].values())
        transfers=read(label+'_transfer_witness.json');assert {z['rank'] for z in transfers}==set(range(16))
        assert {z['pid'] for z in transfers}==set(guards['167']['worker_namespace_pids'].values())
        for row in transfers:
            assert row['policy']=='fail'
            events=[e for e in row['events'] if e['scope']==label]
            starts=[e for e in events if e['kind']=='transfer_before'];ends=[e for e in events if e['kind']=='transfer_after']
            assert starts and len(starts)==len(ends)
            assert {e['request_id'] for e in starts}=={e['request_id'] for e in ends}
            assert all(e['request_id'] not in e['failed_pending'] for e in ends)
            assert not any(e['kind']=='invalid_report' and e['ids'] for e in events)
        graphs=read(label+'_witness.json');assert len(graphs)==32
        for draft in (False,True):
            selected=[z for z in graphs if z['draft']==draft];assert {z['rank'] for z in selected}==set(range(16))
            assert {z['pid'] for z in selected}==set(guards['167']['worker_namespace_pids'].values())
            for row in selected:
                assert not row['errors']
                if draft:assert not row['entries'] and all(':NONE:' in k for k in row['counts'])
                else:assert row['counts'].get(label+':target:FULL:replay_after',0)>=2
        details.append({'label':label,'prompt_tokens':prompt_tokens,'completion_tokens':completion_tokens,'exact_IDs_and_terminal':True,'all16_clean_transfers':True,'workload_signature':work['signature']})
    gate=next(n for n in ast.parse((root/'pd_graph_bucket_diagnostic.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='adjudicate')
    ns={};exec(compile(ast.Module(body=[gate],type_ignores=[]),'frozen_Run265_adjudicate','exec'),ns)
    replay=ns['adjudicate'](read('graph_complete_witness.json'));assert replay['real_dynamic_replay_supported']
    expected=read('diagnostic_decision.json');assert replay==expected
    out={'run_id':'GLM-RUN-0265','verdict':'CANONICAL_KV_FAILURE_AND_CLEAN_DYNAMIC_REPLAY_SUPPORTED','negative_API_error':errors[0],'logical_invalid_ids':ids,'no_failed_request_target_or_draft_all16':True,'positive_PD_fixtures':details,'real_dynamic_replay':replay,'H6_H5_retained':True,'performance_claim':False,'Current':None,'full_API_accepted':False,'input_identity':identities,'limitations':['One uniform bucket2/K1 request shape and two golden fixtures; no concurrency or80K/600/93% SLA acceptance.','Observers add CPU/device synchronization/file IO; latency is not a matched gain result.','KV logical identity and empty-terminal API repair are correctness candidates, not performance patches.']}
    (root/'diagnostic_reduced.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'verdict':out['verdict'],'all_rank_real_replays':[z['actual_replay_count'] for z in replay['all_rank_details']],'canonical_error':errors[0],'H6_H5_retained':True,'performance_claim':False}))

if __name__=='__main__':main(Path(sys.argv[1]))
