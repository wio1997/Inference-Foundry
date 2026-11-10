#!/usr/bin/env python3
"""Reprice the actual26-call runner frame, retaining measured first bodies."""
from pathlib import Path
from hashlib import sha256
from collections import defaultdict
import json,itertools
ROOT=Path('audit/execution-search');TIMING=Path('audit/framework-timing-171')
def union_ns(spans):
    total=0;end=None
    for a,b in sorted(spans):
        if end is None or a>=end:total+=b-a;end=b
        elif b>end:total+=b-end;end=b
    return total
def main():
    logpath=ROOT/'common-frame-caller-logs.json';finepath=TIMING/'function-analysis.json';logs=json.loads(logpath.read_text());fine=json.loads(finepath.read_text());ranks=[]
    for r in logs['ranks']:
        assert r['counts']['local']=={'draft_build_for_drafting':11,'runner_build_attention':286}
        assert r['counts']['cp']==r['counts']['local'] and r['counts']['base']=={'draft_build_for_drafting':22,'runner_build_attention':572}
        assert len(r['target_batches'])==11 and all(b['helper_calls']==26 for b in r['target_batches'])
        p=next(p for p in fine['ranks'] if p['ep']==r['ep']);by={s['name']:s for s in p['functions']}
        cpname='vllm_ascend.attention.context_parallel.dsa_v41_cp:AscendDSAV41CPMetadataBuilder.build';hname='vllm_ascend.attention.context_parallel.dsa_cp:AscendDSACPMetadataBuilder._build_local_token_metadata'
        target=[s for s in r['selected'] if s['kind']=='cp' and s['caller']=='runner_build_attention'];groups=defaultdict(list)
        for s in target:groups[s['parent']].append(s)
        assert set(groups)=={b['parent_id'] for b in r['target_batches']}
        cp_cost_ns=0
        for owner,values in groups.items():
            first=min(values,key=lambda v:v['wall_ns']);a=first['wall_ns'];b=a+first['duration_ns'];assert len(values)==26 and first['fully_enclosed']
            children=[s for s in r['selected'] if s['kind']!='cp' and s['caller']=='runner_build_attention' and a<=s['wall_ns'] and s['wall_ns']+s['duration_ns']<=b]
            assert sorted(s['kind'] for s in children)==['base','base','local']
            cp_cost_ns+=first['duration_ns']-union_ns([(s['wall_ns'],s['wall_ns']+s['duration_ns']) for s in children])
        clock=r['clock'];scale=1+clock['slope_ppm']/1e6;guard_us=clock['required_enclosure_residual_us']+1
        # Overpay original first bodies rather than assume their Free fraction:
        # helper whole body; CP remainder includes unselected child/marker work.
        helper_pay=sum(b['first_helper_body_ns'] for b in r['target_batches'])/1e6/scale+11*2*guard_us/1000
        cp_pay=cp_cost_ns/1e6/scale+11*8*guard_us/1000
        ranks.append(dict(ep=r['ep'],window_ms=p['window_ms'],closed_cycles=11,runner_CP_calls=286,draft_CP_calls=11,runner_CP_calls_per_frame=26,
                          CP_self_prepare_free_ms=by[cpname]['prepare_self_free_ms'],helper_self_prepare_free_ms=by[hname]['prepare_self_free_ms'],
                          retained_first_helper_body_plus_guard_ms=helper_pay,retained_first_CP_remainder_plus_guard_ms=cp_pay,
                          original_first_helper_body_ns=sum(b['first_helper_body_ns'] for b in r['target_batches']),original_first_CP_remainder_ns=cp_cost_ns,
                          original_FT_body_misses=r['FT_body_unmatched_in_original_analysis']))
    scenarios=[]
    for cf,hf,retention,exposure,newcost in itertools.product([0,.25,.5,1],[0,.5,1],['count26_mean','recorded_first_body_overpay'],[.5,1],[.1,.5,2]):
        vals=[]
        for r in ranks:
            cp=cf*r['CP_self_prepare_free_ms'];helper=hf*r['helper_self_prepare_free_ms']
            if retention=='count26_mean':cp*=25/26;helper*=25/26
            else:
                if cf:cp-=r['retained_first_CP_remainder_plus_guard_ms']
                if hf:helper-=r['retained_first_helper_body_plus_guard_ms']
            net=exposure*(cp+helper)-newcost*11
            vals.append(dict(ep=r['ep'],gross_after_retained_first_ms=cp+helper,conditional_net_ms=net,own_window_pct=net/r['window_ms']*100))
        mean=sum(v['conditional_net_ms'] for v in vals)/16;window=sum(r['window_ms'] for r in ranks)/16
        scenarios.append(dict(CP_self_fraction_eligible=cf,helper_self_fraction_eligible=hf,first_retention=retention,exposure_fraction=exposure,new_nonoverlap_ms_per_cycle=newcost,base_builder_whole_retained=True,
                              conditional_mean_net_ms=mean,conditional_mean_window_pct=mean/window*100,rank_values=vals))
    result=dict(status='ACTUAL_RUNNER26_AND_DRAFT1_CALLER_BOUND_PAID_PRICE',ranks=ranks,scenarios=scenarios,
                qualification='Processlocal parentIDs classifyactualcallers; not tensorargument identity or performanceimprovement. prepareinput sourceblock2287–2530 calls runner buildmetadata, not draftpropose. Thus helperprepareFree budget is the runner frame, while 27allwindowcalls includes a separate draft1. Firstbodyoverpay conservatively retains recordedhelper wholebody andCPbody minus knownmarkedchildren plus clockguards; not actual firstFree measurement, arbitrarynewprotocol bound or unprofiled price.',
                source_findings=['V41 build_for_drafting839–842 calls self.build withoutkwargs. LLMBase shared_dsa_draft_cache doesnot reach currentV41build; oldLegacyDSA cache source cannot qualify V41draft sharing.',
                                 '12current sourcepins and originalhelperAST separate rawCPU/postCPU/device/resource tiers. No newnativebit/admission. Initialmean27retention was sensitivity; use26callerbound/recordedbody cost now, keepoldartifact intact.'],
                limits=['144scenarios retain negatives and entirebase work. Newh includescacheguard/snapshot/copy/pin/events/resource costs; eligibility/exposure remainassumptions.','Original2FTbody misses remainunknown; hostparentlog IDs recovercaller counts but do not synthesize missingdeviceFree timing.','No newNPU/profile/requests; originalfrozenC8/ordinary fullrequest and deviceprimitive gates remain.'],
                inputs=[dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest()) for p in [logpath,finepath,ROOT/'common-frame-key-dependencies.json']],analyzer_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),whole_chain_complete=False,new_primitive_admissions=0,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[])
    (ROOT/'common-frame-caller-price.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    select=[s for s in scenarios if s['CP_self_fraction_eligible'] in [0,.25] and s['helper_self_fraction_eligible']==1 and s['first_retention']=='recorded_first_body_overpay' and s['exposure_fraction']==.5 and s['new_nonoverlap_ms_per_cycle']==.5]
    print(json.dumps(dict(ranks=len(ranks),cases=len(scenarios),selected=[{k:v for k,v in s.items() if k!='rank_values'} for s in select]),ensure_ascii=False))
if __name__=='__main__':main()
