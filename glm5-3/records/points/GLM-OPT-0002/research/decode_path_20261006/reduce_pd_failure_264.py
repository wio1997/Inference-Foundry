"""Audit Run264's observed KV failure and API error drop from saved raw."""
from pathlib import Path
import hashlib,json,sys

def main(root):
    identities={}
    def read(name):
        p=root/name;raw=p.read_bytes();identities[name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)};return json.loads(raw)
    state=read('state.json');assert state['status']=='failed'
    plan=read('functional_plan.json');retained=read('retained_stack.json');guards=read('guards_after.json')
    assert retained['H6'] and retained['H5'] and retained['event_mode']==retained['MC2_mode']==1
    assert not any(retained[k] for k in ('H4','H8','performance_claim','temporary_graph_configuration','observer_resident','kv_identity_patch'))
    assert guards['166']['root']==read('guards_before.json')['166']['root']
    assert all(z['health']==200 and z['idle'] and len(z['device_owners'])==16 for z in guards.values())
    witness=read('retained_witness.json');assert witness['H6']['all_rank_witness']
    assert len(witness['H6']['rows'])==len(witness['H5_rows'])==16
    assert all(z['dispatch_cached_true'] and z['combine_cached_true'] and z['mode']==1 and z['library_sha256']==plan['native_candidate']['sha256'] for z in witness['H6']['rows'])
    assert all(z['mode']==1 and not z['configured_overlap'] and z['returned_none'] for z in witness['H5_rows'])
    events=read('kv_fault_events.json')
    assert events[-1] is None and not any(e and 'error' in e for e in events)
    choices=[c for e in events if e for c in e.get('choices',[])]
    assert not any(c.get('finish_reason') or c.get('token_ids') for c in choices)
    assert [e['usage']['completion_tokens'] for e in events if e and e.get('usage')]==[0]
    ranks=[read('transfer_witnesses/rank%d.json'%rank) for rank in range(16)]
    assert {z['rank'] for z in ranks}==set(range(16))
    r5=ranks[5];assert r5['rank']==5
    task=next(e for e in r5['events'] if e['kind']=='transfer_before' and e['scope']=='kv_fault')
    assert task['local_logical_block_ids']==[[1,2]]
    assert task['local_block_ids'][0]==list(range(16,32))
    reports=[e['ids'] for e in r5['events'] if e['kind']=='invalid_report' and e['scope']=='kv_fault' and e['ids']]
    assert reports and all(ids==[1,2] for ids in reports)
    candidate_log=root/'epochs/candidate/native_167.log';log=candidate_log.read_text()
    assert 'Failing 1 request(s) due to KV load failure' in log and task['request_id'] in log
    for rank in range(16):
        for model in ('target','draft'):
            row=read('witnesses/%s_rank%d.json'%(model,rank))
            assert row['rank']==rank and not any(k.startswith('kv_fault:') for k in row['counts'])
    recovered=read('H6_H5_recovery_warm_result.json')
    golden=json.loads((root.parent/'GLM-RUN-0262/correctness_mode0_short_result.json').read_text())
    assert recovered['prompt_tokens']==2334 and recovered['completion_tokens']==8
    assert recovered['token_ids']==golden['token_ids'] and recovered['finish_reason']=='length'
    assert not (root/'kv_fault_decision.json').exists() and not (root/'graph_short_result.json').exists()
    out=dict(run_id='GLM-RUN-0264',verdict='INTERNAL_KV_FAIL_CLOSED_API_EMPTY_ERROR_DROPPED',logical_invalid_ids=reports,physical_group0=task['local_block_ids'][0],no_target_or_draft_execution_all16=True,API_error_contract_passed=False,clean_graph_test_issued=False,recovered_H6_H5_eager_golden=True,performance_claim=False,Current=None,input_identity=identities)
    (root/'failure_reduced.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='input_identity'}))

if __name__=='__main__':main(Path(sys.argv[1]))
