#!/usr/bin/env python3
"""Use disjoint function-self diagnostic exposure for a paid common-frame screen."""
from hashlib import sha256
from pathlib import Path
import itertools,json
from statistics import median

ROOT=Path('audit/execution-search'); TIMING=Path('audit/framework-timing-171')
NAMES={
 'cp':'vllm_ascend.attention.context_parallel.dsa_v41_cp:AscendDSAV41CPMetadataBuilder.build',
 'base':'vllm_ascend.attention.dsa_v41:AscendDSAV41MetadataBuilder.build',
 'local':'vllm_ascend.attention.context_parallel.dsa_cp:AscendDSACPMetadataBuilder._build_local_token_metadata',
}
def main():
    paths=[TIMING/n for n in ['function-analysis.json','adapter.json','receipt.json','source-details.json','analyze_functions.py']]
    fine=json.loads(paths[0].read_text());adapter=json.loads(paths[1].read_text());receipt=json.loads(paths[2].read_text());assert receipt['status']=='CAPTURED_RESTORED'
    dep=json.loads((ROOT/'common-frame-measured-entry.json').read_text())
    for row in dep['sources']:
        match=[v for v in adapter['modules'].values() if v['sha256']==row['runtime_sha256']];assert len(match)==1
        assert sha256(Path(row['local_path']).read_bytes()).hexdigest()==row['runtime_sha256']
    ranks=[]
    for p in fine['ranks']:
        functions={r['name']:r for r in p['functions']};fs={key:functions[name] for key,name in NAMES.items()}
        self_ms={key:r['prepare_self_free_ms'] for key,r in fs.items()};counts={key:r['calls'] for key,r in fs.items()}
        assert p['closed_cycles']==11 and all(v>0 for v in counts.values())
        assert abs(sum(p['prepare_free_function_partition_ms'].values())-p['prepare_free_ms'])<1e-5
        envelope=sum(self_ms.values());cp_local=self_ms['cp']+self_ms['local']
        assert 0<=cp_local<=envelope<=p['prepare_free_ms']<=p['activity_ms']['Free']
        ranks.append(dict(ep=p['ep'],dp=p['dp'],window_ms=p['window_ms'],closed_cycles=p['closed_cycles'],prepare_free_ms=p['prepare_free_ms'],
                          marker_prepare_free_ms=p['added_marker_prepare_free_ms'],body_marker_unmatched=p['body_marker_unmatched'],
                          function_self_prepare_free_ms=self_ms,calls=counts,
                          calls_per_cycle={k:v/p['closed_cycles'] for k,v in counts.items()},base_minus_two_CP_calls=counts['base']-2*counts['cp'],
                          cp_local_allfree_ms=cp_local,three_function_allfree_ms=envelope))
    assert len(ranks)==16 and sorted(r['ep'] for r in ranks)==list(range(16))
    scenarios=[]
    for cp_share,local_share,retention,exposure,h in itertools.product([0,.25,.5,1],[0,.5,1],['average_first_call_per_cycle',.25],[.5,1],[.1,.5,2]):
        vals=[]
        for r in ranks:
            t=r['function_self_prepare_free_ms'];n=r['closed_cycles'];c=r['calls']
            retain={k:(n/c[k] if isinstance(retention,str) else retention) for k in ['cp','local']}
            gross=cp_share*t['cp']*(1-retain['cp'])+local_share*t['local']*(1-retain['local'])
            net=exposure*gross-h*n
            vals.append(dict(ep=r['ep'],conditional_gross_ms=gross,retained_work_fraction=retain,conditional_net_ms=net,conditional_own_window_pct=net/r['window_ms']*100))
        mean_net=sum(v['conditional_net_ms'] for v in vals)/16;mean_window=sum(r['window_ms'] for r in ranks)/16
        scenarios.append(dict(cp_self_share_eligible=cp_share,local_helper_self_share_eligible=local_share,retained_first_producer=retention,
                              free_overlap_to_shifted_critical_fraction=exposure,added_nonoverlap_ms_per_cycle=h,base_builder_whole_retained=True,
                              mean_conditional_net_ms=mean_net,conditional_mean_window_pct=mean_net/mean_window*100,
                              rank_conditional_net_minmax_ms=[min(v['conditional_net_ms'] for v in vals),max(v['conditional_net_ms'] for v in vals)],rank_values=vals))
    group=fine['groups']['all'];means={k:sum(r['function_self_prepare_free_ms'][k] for r in ranks)/16 for k in NAMES}
    assert all(abs(next(r for r in group['functions'] if r['name']==NAMES[k])['prepare_self_free_ms']-v)<1e-7 for k,v in means.items())
    result=dict(status='MEASURED_FUNCTION_SELF_COMMON_FRAME_PAID_SENSITIVITY',ranks=ranks,scenarios=scenarios,
                mean_window_ms=group['window_ms'],mean_prepare_free_ms=group['prepare_free_ms'],mean_marker_prepare_free_ms=group['added_marker_prepare_free_ms'],
                function_self_prepare_free_means_ms=means,cp_local_self_envelope_ms=means['cp']+means['local'],
                three_function_envelope_ms=sum(means.values()),
                source_binding='Five currentdisk moduleSHAs equal instrumentedadapter hashes. Actual CPbuild calls _global_builder.build then super.build(localcommon), and runner resets batchshared once/call while resource shared dict is perKVgroup. Structuraltwoowners is sourceproof, not exact27 distinctgroups fromcounts; two unmatchedFTbody entries and clippedwindow counts retained.',
                limits=['New detailed mirroredC8 diagnostic1025.503ms averagewindow,11closedcycles/rank; different from old839ms or940/930ms windows and original8distinctC8. No fullrequestgain extrapolation.',
                        'prepare_self_free is disjoint innermostfunction/prepare/Free overlap with90.3469ms markerboundary separately removed byoriginalanalyzer. Never subtract marker twice or add CP inclusive/global/base/helper nesting.',
                        'Function self includes librarycalls/waits and residualprobe effects; not purePython or a causalremovablephase. Average-firstcall retention is explicitproxy, not measured firstproducer cost or27 guaranteed identicalcalls.',
                        'Base builder whole retained in everypaidscenario. Shared batch/slot/C2/rope/native task credit alreadyimplemented cannot be countedagain. Perresource CPdevicepointers, slot/table, originaldevice authoritative seqLens and consumerattrs remain required.',
                        'Costs include replacement compute/hostsnapshots/pins/H2D/perresourceassembly/events/contention in h. Exposure factor is a sensitivity for shiftedcriticalwork, not observedrequestcriticalfraction; allnegative results retained.',
                        'No newprimitivebit, graph/alias/lifecycle or nativeimplementation admission. Crossgeneration/keyreuse and wholeobject sharing remain stopped.'],
                inputs=[dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest()) for p in paths+[ROOT/'common-frame-measured-entry.json']],
                analyzer_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),whole_chain_complete=False,new_primitive_admissions=0,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[])
    (ROOT/'common-frame-measured-price.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    chosen=[s for s in scenarios if s['cp_self_share_eligible'] in [0,.25] and s['local_helper_self_share_eligible']==1 and s['retained_first_producer']=='average_first_call_per_cycle' and s['free_overlap_to_shifted_critical_fraction']==.5 and s['added_nonoverlap_ms_per_cycle']==.5]
    print(json.dumps(dict(function_means=means,window=group['window_ms'],marker=group['added_marker_prepare_free_ms'],scenarios=len(scenarios),selected=[{k:v for k,v in s.items() if k!='rank_values'} for s in chosen]),ensure_ascii=False))

if __name__=='__main__':main()
