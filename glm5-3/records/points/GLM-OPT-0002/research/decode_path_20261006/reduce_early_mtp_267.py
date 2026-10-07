"""Independent saved-input H10 reduction; no import of controller decision code."""
from pathlib import Path
import hashlib,json,math,re,statistics

ROOT=Path(__file__).resolve().parents[2]/'runs/GLM-RUN-0267'
def read(name):return json.loads((ROOT/name).read_text())
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8),(a,b)

def main():
    plan=read('functional_plan.json');gold=plan['complete_golden'];state=read('state.json')
    assert state['status']=='completed' and read('earlymtpcompare.phase.json')['exit_code']==0
    identity=read('download_identity.json')
    for name,row in identity.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==row['sha256'],name
    results={z['label']:z for z in read('comparison_results.json')};allrows={}
    for path in sorted(ROOT.glob('*_request.json')):
        label=path.name.removesuffix('_request.json');body=read(path.name);r=read(label+'_result.json');events=read(label+'_events.json')
        assert events[-1]['value'] is None
        values=[e['value'] for e in events];sse=[]
        for line in (ROOT/(label+'_D.sse')).read_text().splitlines():
            if line.startswith('data: '):sse.append(None if line[6:]=='[DONE]' else json.loads(line[6:]))
        assert sse==values
        choices=[c for e in values if e for c in e.get('choices',[])]
        assert not any(e and 'error' in e for e in values)
        ids=[t for c in choices for t in c.get('token_ids',c.get('delta',{}).get('token_ids',[])) or []]
        complete=body['max_tokens']==96;expected=gold['token_ids'] if complete else [785,1196,374,10156,264,3405,304,8452]
        assert ids==r['token_ids']==expected and r['completion_tokens']==len(expected)
        assert r['finish_reason']==('stop' if complete else 'length')
        assert r['prompt_tokens']==(gold['prompt_tokens'] if complete else 2334)
        if complete:assert r['final_content']==gold['final_content']
        usage=[e['usage'] for e in values if e and e.get('usage')][-1]
        assert usage['completion_tokens']==len(ids) and usage['prompt_tokens']==r['prompt_tokens']
        arrivals=[e['arrived_ns'] for e in events if e['value'] and any(c.get('token_ids') or c.get('delta',{}).get('token_ids') for c in e['value'].get('choices',[]))]
        close(r['TPOT_ms'],(arrivals[-1]-arrivals[0])/1e6/(len(ids)-1));close(r['PD_wall_s'],r['P_wall_s']+r['D_wall_s'])
        assert any('hits_total' in k and v==r['prompt_tokens'] for k,v in r['external_KV_delta'].items())
        work=read(label+'_workload.json');delta={k:work['after']['counters'][k]-v for k,v in work['before']['counters'].items()}
        assert delta==work['delta'] and all(v>=0 and float(v).is_integer() for v in delta.values())
        counters={k.split('{',1)[0]:int(v) for k,v in delta.items()}
        signature={'num_drafts':counters['vllm:spec_decode_num_drafts_total'],'num_draft_tokens':counters['vllm:spec_decode_num_draft_tokens_total'],'num_accepted_tokens':counters['vllm:spec_decode_num_accepted_tokens_total'],'accepted_position0':counters['vllm:spec_decode_num_accepted_tokens_per_pos_total'],'invalid_draft_tokens':counters['vllm:spec_decode_num_drafts_total']-counters['vllm:spec_decode_num_draft_tokens_total']}
        assert signature==work['signature'] and all(work['checks'].values())
        if label in results:assert results[label]['workload_signature']==signature
        allrows[label]=dict(r,workload_signature=signature)
    assert len(allrows)==21 and len(results)==12
    native=(ROOT/'epochs/candidate/native_167.log').read_text(errors='replace')
    workers=[]
    for prefix,mode in (('A1',0),('B1',1),('A2',0),('B2',1)):
        witness=read(prefix+'_witness.json');roles=witness['same_worker_identity'];workers.append(roles)
        assert witness['H9_mode']==0 and witness['H10_mode']==mode and witness['H6_mode']==witness['H5_event_mode']==1
        assert witness['H6']['all_rank_witness'] and witness['H6']['mode']==1 and witness['H6']['library_sha256']==plan['native_candidate']['sha256']
        assert len(witness['H6']['rows'])==len(witness['H5_rows'])==len(witness['H9_rows'])==len(witness['H10_rows'])==16
        assert all(z['dispatch_cached_true'] and z['combine_cached_true'] and z['mode']==1 for z in witness['H6']['rows'])
        for rows in (witness['H5_rows'],witness['H9_rows'],witness['H10_rows']):
            assert {z['rank'] for z in rows}==set(range(16))
            assert {z['pid'] for z in rows}==set(roles['worker_namespace_pids'].values())
        assert all(z['mode']==1 and z['returned_none'] and not z['configured_overlap'] for z in witness['H5_rows'])
        assert all(not z['eligible'] and z['target'] and z['mode']==0 and not z['skipped_host_sync'] and z['runtime_mode']=='FULL' and z['capture_sizes']==[2] and z['eager_breaks']==0 and z['capture_num_graphs']==1 and z['candidate_graph_sha256']==plan['candidate_sha256'] and z['candidate_runner_sha256']==plan['runner_production_sha256'] for z in witness['H9_rows'])
        assert all(z['mode']==mode and z['eligible'] and z['early']==bool(mode) and not z['pp'] and z['greedy'] and not z['async_scheduling'] and z['num_spec']==z['scheduled_K']==1 and z['generators']==0 and not z['prompt_logprobs'] and not z['routed_experts'] and z['model_type']=='glm_moe_dsa' and z['candidate_runner_sha256']==plan['runner_production_sha256'] for z in witness['H10_rows'])
        label=prefix+'_complete';transfer=read(prefix+'_transfer.json');request_id=read(label+'_P.raw')['kv_transfer_params']['remote_request_id']
        lines=[line for line in native.splitlines() if 'KV cache transfer for request '+request_id+' took ' in line]
        assert transfer['rows']==lines and {int(re.search(r'local_device_id (\d+)',line).group(1)) for line in lines}==set(range(16)) and len(lines)==16
    assert all(z==workers[0] for z in workers)
    complete={p:allrows[p+'_complete'] for p in ('A1','B1','A2','B2')}
    assert all(z['workload_signature']==complete['A1']['workload_signature'] for z in complete.values())
    assert all(allrows[p+'_measure%d'%i]['workload_signature']==allrows['A1_measure0']['workload_signature'] for p in complete for i in range(2))
    drift=max(abs(complete['A2']['D_wall_s']-complete['A1']['D_wall_s']),abs(complete['B2']['D_wall_s']-complete['B1']['D_wall_s']))
    pairs=[]
    for a,b in (('A1','B1'),('A2','B2')):
        x,y=complete[a],complete[b];short={p:statistics.median(allrows[p+'_measure%d'%i]['TPOT_ms'] for i in range(2)) for p in (a,b)}
        pairs.append(dict(pair=a+'->'+b,D_TPOT_ms=[x['TPOT_ms'],y['TPOT_ms']],D_saving_s=x['D_wall_s']-y['D_wall_s'],PD_saving_s=x['PD_wall_s']-y['PD_wall_s'],P_saving_s=x['P_wall_s']-y['P_wall_s'],TPOT_reduction=1-y['TPOT_ms']/x['TPOT_ms'],short_TPOT_medians=short))
    positive=all(p['D_saving_s']>drift and p['PD_saving_s']>0 and p['TPOT_reduction']>0 and list(p['short_TPOT_medians'].values())[1]<list(p['short_TPOT_medians'].values())[0] for p in pairs)
    decision=read('research_decision.json');assert positive==decision['research_positive'];close(drift,decision['D_drift_s'])
    retained=read('retained_witness.json');stack=read('retained_stack.json');guards=read('guards_after.json')
    assert stack['H6'] and stack['H5'] and not any(stack[x] for x in ('H4','H8','H9')) and stack['H9_mode']==0 and stack['H10']==positive and stack['H10_mode']==int(positive)
    assert retained['same_worker_identity']==workers[0] and all(z['mode']==0 and not z['skipped_host_sync'] for z in retained['H9_rows'])
    assert retained['H10_mode']==int(positive) and all(z['mode']==int(positive) and z['early']==positive for z in retained['H10_rows'])
    assert all(z['health']==200 and z['idle'] and len(z['device_owners'])==16 for z in guards.values())
    out=dict(verdict='POSITIVE_LIMITED_RESEARCH_H10_RETAINED' if positive else 'INCONCLUSIVE_NOT_REPEATED_H10_OFF',research_positive=positive,controller_status='completed',terminal_reconciliation_needed=False,normal_requests=21,complete_PD_matched=True,short_work_matched=True,all16_replay_guard_and_transfer=True,same_workers=True,D_drift_s=drift,pairs=pairs,complete_MTP_signature=complete['A1']['workload_signature'],short_MTP_signature=allrows['A1_measure0']['workload_signature'],retained_research_stack=['H6','H5']+(['H10'] if positive else []),Current=None,formal_SLA=False,full_API=False,limitations=['Fixed FULL bucket2 and eager K1 MTP, one request at a time; not formal80K/600/93% workload or all dynamic API acceptance.','No measured native barrier service or global largest remaining gap; no speed claim relative to old eager configuration.'])
    (ROOT/'performance_reduced.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('limitations','pairs')}))

if __name__=='__main__':main()
