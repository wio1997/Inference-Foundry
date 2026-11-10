#!/usr/bin/env python3
"""Source/resource review of QLI; existing data only, no runtime imports.

Capacity formulas describe declared user-workspace layout, not resident VRAM.
Merge cost scenarios never convert PMU ratios or record counts into latency.
"""
from hashlib import sha256
import json
from pathlib import Path

from analyze_attention_output_delivery import extract

OUT=Path('audit/execution-search')
BASE=Path('/Users/wio/work/branch-fusion-audit-20261104')


def pin(p):
    p=Path(p)
    return dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest())


def main():
    rows=[]
    for name in ['indexer-native-execution-identity.json','indexer-native-extra-dependencies.json']:
        for r in json.loads((OUT/name).read_text())['sources']:
            p=Path(r['local_path']);h=sha256(p.read_bytes()).hexdigest()
            assert h==r['runtime_sha256']
            rows.append(dict(runtime_path=r['runtime_path'],path=str(p),sha256=h))
    byname={Path(r['path']).name:Path(r['path']).read_text() for r in rows}
    assert 'bool supportFd_ = false;' in byname['quant_lightning_indexer_v2_metadata_aicpu.h']
    meta=byname['quant_lightning_indexer_v2_metadata_aicpu.cpp']
    # All writes that turn FD on occur inside the ASCEND950 branch.
    assert meta.count('supportFd_ = true;')==1
    assert meta.index('supportFd_ = true;')>meta.index('if (validSocVersion == ValidSocVersion::ASCEND950)',meta.index('void QuantLightningIndexerV2MetadataCpuKernel::CalcSplitInfo'))
    kernel=byname['quant_lightning_indexer_v2_kernel_arch22.h']
    assert 'vectorService.InitLDBuffers(pipe);' in kernel and 'SyncAll();' in kernel
    vec=byname['quant_lightning_indexer_v2_service_vector_arch22.h']
    assert 'pipe->Reset();' in vec
    package_path=OUT/'indexer-package-identity.json'
    pkg=json.loads(package_path.read_text());assert len(pkg['worker_processes'])==16
    lib=pkg['custom_opapi']['path']
    assert all(lib in r['mapped_custom_libraries'] for r in pkg['worker_processes'])
    binary,= [r for r in pkg['kernel_package'] if r['path'].endswith('.o')]
    metadata,= [r['metadata'] for r in pkg['kernel_package'] if 'metadata' in r]
    assert metadata['sha256']==binary['sha256']=='f1dd0d93d703c252a51c517fac851c055d0a09ba373c22dbbd92d1b042daaa23'
    keys=[r['tilingKey'] for r in metadata['kernelList']];assert len(keys)==4
    cap=[]
    for g in [32,64]:
        # Conditional C24 example; use C symbolically, do not equate a DFC grant
        # with physical cores or the runtime mixed-kernel GetBlockNum contract.
        c=24
        host_per_c=256*2048*4*2 + 2*4*2*2048*4 + 2*4*16*8
        mm1_per_c=2*4*2048*4
        weight_per_c=16*(4*g)*2*2
        ld_per_c=4*2*2*2048*4
        complete_per_c=mm1_per_c+weight_per_c+ld_per_c
        cap.append(dict(g=g,
            host_declared_user_increment_bytes_per_host_AIC=host_per_c,
            kernel_explicit_regions_bytes_per_kernel_block=dict(mm1=mm1_per_c,weight=weight_per_c,LD=ld_per_c,total=complete_per_c),
            example_host_AIC_equals_kernel_blocks=c,
            example_host_declared_increment_MiB=c*host_per_c/2**20,
            example_complete_explicit_regions_MiB=c*complete_per_c/2**20,
            example_declared_minus_regions_MiB=c*(host_per_c-complete_per_c)/2**20,
            example_no_LD_accessed_regions_MiB=c*(mm1_per_c+weight_per_c)/2**20))
    handoff_path=OUT/'indexer-delivery-entry.json';h=json.loads(handoff_path.read_text())
    ws=h['windows'];data=extract(BASE)
    lookup={(w['dp'],w['tp'],w['carrier'],w['layer']):w for w in ws}
    sources=h['source_layers']
    def price(fraction,exposure,new_cost_us):
        carriers={m:0. for m in data['carrier_order']}
        for sec in data['arrival_sections']:
            for l in sources:
                ds=[]
                for dp in [0,1]:
                    credits=[]
                    for tp in range(8):
                        w=lookup.get((dp,tp,sec['M'],l))
                        credits.append(max(0.,w['qli_us']*.75*fraction*exposure-new_cost_us) if w else 0.)
                    ds.append(max(credits))
                lo,hi=sec['layers'][l]['DP0_minus_DP1_us']
                def gain(d):return max(d,0)-max(d-ds[0],-ds[1])
                carriers[sec['M']]+=max(gain(lo),gain(hi))/1000
        req=[]
        for r in data['requests']:
            credit=sum(carriers[m] for m in data['carrier_order'][:data['carrier_order'].index(r['first_eager_M'])+1])
            req.append(dict(slot=r['slot'],old_ms=r['TTFT_ms'],conditional_credit_ms=credit))
        old=max(r['old_ms'] for r in req);saved=old-max(r['old_ms']-r['conditional_credit_ms'] for r in req)
        return dict(original_kernel_time_fraction_in_accumulator=fraction,exposed_fraction=exposure,
                    assumed_accumulator_time_reduction=.75,new_cost_us_per_source=new_cost_us,
                    P90_saved_ms=saved,P90_percent=100*saved/old,requests=req)
    prices=[price(f,e,h) for f,e,h in [(1.,1.,0.),(.25,1.,0.),(.25,1.,5.),(.25,1.,20.),(.5,.5,5.),(.5,1.,5.)]]
    thresholds=[]
    for exposure in [1.,.5]:
        for cost in [0.,5.,20.]:
            lo,hi=0.,1.
            for _ in range(35):
                mid=(lo+hi)/2
                if price(mid,exposure,cost)['P90_percent']>=1:hi=mid
                else:lo=mid
            thresholds.append(dict(exposure=exposure,new_cost_us_per_source=cost,
                                   required_accumulator_time_fraction=hi if price(hi,exposure,cost)['P90_percent']>=1 else None))
    history={}
    for n in ['QLI-CANDIDATE-PMU.json','QLI-TOKEN-MERGE-VALUE.json','QLI-ON-BATCHED-SLOT-C8.json','QLI-MEMBERSHIP-OVERLAY.json']:
        p=BASE/'full-review'/n;d=json.loads(p.read_text());history[n]=dict(pin=pin(p))
        if n=='QLI-CANDIDATE-PMU.json':
            history[n].update(scope=d['scope'],resource_bands=d['by_mode_and_duration_band'],limits=d['limits'])
        elif n=='QLI-TOKEN-MERGE-VALUE.json':
            assert d['source_contract']['service_sha256']==next(r['sha256'] for r in rows if r['path'].endswith('service_vector_arch22.h'))
            assert d['source_contract']['helper_sha256']==next(r['sha256'] for r in rows if Path(r['path']).name=='quant_lightning_indexer_v2_vector.h')
            history[n].update(source_contract=d['source_contract'],limits='Historical phase/ordinary scenarios are not newly matched request gains; current eager costs repriced separately here.')
        elif n=='QLI-ON-BATCHED-SLOT-C8.json':
            history[n].update(decision=d['decision_zh'],gates=d['gates'],limits=d['limits'])
        else:
            assert d['compile_and_runtime']['original_kernel_sha256']==binary['sha256']
            history[n].update(native_results=d['native_results'],timing=d['timing'],limits=d['limits'])
    out=dict(status='BOUNDED_NATIVE_QLI_BRANCH_RESOURCE_AND_PAID_VALUE_REVIEW',whole_chain_complete=False,
        every_replacement_bitwise=True,source_pins=rows,package_identity=pin(package_path),
        current_package=dict(opapi_sha256=pkg['custom_opapi']['sha256'],workers_mapped_selected_library=16,
                             binary_sha256=binary['sha256'],compiled_keys=keys,
                             qualification=pkg['scope']),
        source_branch=dict(fixed='910C arch22, INT8 Q/K, FP16 weights/scales, TND/PA_BBND, G32/D128/top512, ratio1/2, mode1/2/3',
            metadata='source supportFd=False on non-Ascend950; assignments consume complete S1 groups; no S2 split or LD merge payload on this producer path. Not a live per-call metadata buffer assertion.',
            empty_epilogue='ProcessDecode still InitLDBuffers(pipe Reset), ICachePreLoad and SyncAll before testing LD enable. Thus no LD DATA does not mean no epilogue work.',
            candidate='mode1: per8-key max, integer bit-pattern -inf pad/+inf tail pin, original sort/merge and2048 ordered block IDs. mode2: membership after all QK/scoring; preserves finite NEG_HUGE filler locations. mode3: no candidate buffer.',
            physical='PA KeyNd2NzForPA uses block_table*keyStride0+pageoffset*128; scale uses separate signed64-bit block_table*scaleStride0. Inner axes contiguous, page0 padding allowed, zero stride uses compact fallback.',
            sort='Token full tile Sort32+multiway merge retains2048 accumulator though output512; candidate has its own2048 accumulator and integer Adds(0) V copy to preserve index bit patterns. Stable-prefix reduction is a candidate, not device bitwise admission.'),
        workspace= dict(formula_rows=cap,
            scope='Host increment excludes platform LibApi workspace L, which must be retained. Explicit kernel regions use kernel GetBlockNum B; host reserve uses platform core count C. MiB examples assume B=C=24, not current pool/residency measurement. No per-layer/per-graph sum; graph/ACL allocator alias and high-water policy unknown.',
            candidate='Reconcile declared256-row score allocation against explicit4-row score region before adding a new bank. Native array bounds/hidden sync scratch/user-workspace prefix and graph lifetimes must be checked before shrinking. Source-only capacity opportunity, no speedup or bitwise admission.'),
        token_accumulator_paid_scenarios=prices,token_accumulator_one_percent_thresholds=thresholds,
        price_scope='Only actual368 eager QLI sites, old same-cohort DP arrival envelope; retain quant and chronological sort.75% record reduction is assumed75% accumulator TIME reduction only as a scenario. PMU ratios are never time fractions. No decode/ordinary inference or additive gain.',
        matched_history=history,
        decisions=[
            dict(category='已停止',scope='Delete existing LD merge payload under fixed910C metadata producer',result='No such payload; direct credit0. Empty InitLDBuffers/SyncAll epilogue is separate and remains conditional.'),
            dict(category='待确认',scope='Smaller token prefix accumulator together with whole QLI membership/producer',result='Preserve full tile sort and candidate2048 separately; need original intrinsic tie/NaN/underfull512 bytes and phase/exposure/new cost. f=.25 free premise is marginal;20us/source cost changes1% decision in this scenario.'),
            dict(category='待确认',scope='Right-sized native workspace capacity enabling a larger useful bank',result='Declared versus explicit regions differs materially; actual alias/high-water/hidden prefix and legal buffer bound/bitwise gates remain. No assumed752MiB/8 copies.'),
            dict(category='待确认',scope='Shared membership representation / sparse scoring / SMLA megakernel',result='Original half chain, current SubMulAdd/filler, distinct query spaces and group publication retained. Old native byte-mask result supports local mechanism; batched-slot C8 failed frozen robust request gates and is not a new accepted implementation.'),
        ],
        next_action='Continue source-owned latent to index K/norm/RoPE/dynamic quant/two scatters and long-KV epilog consumers, from actual update_keys caller; A5 folded cache is guard-excluded on910C. Whole delivery price before child fusions.',
        inputs=[pin(handoff_path),*data['source_pins']],
        new_NPU_work=0,new_profile_collection=0,model_loads=0,service_requests=0,remote_writes=[],service_changes=[])
    (OUT/'indexer-native-contract-frontier.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(sources=len(rows),workspace=cap,paid_P90=[{k:r[k] for k in ['original_kernel_time_fraction_in_accumulator','exposed_fraction','new_cost_us_per_source','P90_percent']} for r in prices],thresholds=thresholds,whole_chain_complete=False),ensure_ascii=False))


if __name__=='__main__':main()
