#!/usr/bin/env python3
"""Persist exact route/index/last-reader ranges, including reused prior ranges."""
from pathlib import Path
from hashlib import sha256
import json

ROOT=Path('audit/execution-search')
RANGES={
 'dispatch_ffn_combine_kernel.hpp':[[188,210],[360,433],[655,731],[940,1037],[1038,1190]],
 'moe_init_routing_quant_v2.cpp':[[1,133]],
 'moe_v2_fullload_dynamic_quant.h':[[85,218]],
 'moe_v2_sort_one_core.h':[[42,162]],
 'moe_v2_sort_multi_core.h':[[221,289],[366,373]],
 'moe_v2_sort_base.h':[[20,70]],
 'moe_v2_mrgsort.h':[],
 'moe_v2_mrgsort_out.h':[[142,213]],
 'moe_v2_src_to_dst_op.h':[[58,164]],
 'moe_v2_expert_token_out.h':[[108,148],[201,233],[290,310]],
 'moe_v2_common.h':[[29,56]],
}

def main():
    data=json.loads((ROOT/'dfc-row-map-dependencies.json').read_text());sources=[];excerpts=[]
    for r in data['sources']:
        p=Path(r['local_path']);b=p.read_bytes();h=sha256(b).hexdigest();assert h==r['runtime_sha256'];lines=b.decode().splitlines();rr=RANGES[p.name]
        sources.append(dict(runtime_path=r['runtime_path'],path=str(p),sha256=h,reviewed_ranges=rr,
                            qualification='Exact ranges only. kernel655–731/initRoutingcpp reuse prior bounded review with identical currentSHA. mrgsort.h identity-only; original Sort/MrgSort retained, stability not assumed.'))
        for a,z in rr:
            assert 1<=a<=z<=len(lines);excerpts.append(dict(runtime_path=r['runtime_path'],sha256=h,first_line=a,last_line=z,lines=lines[a-1:z]))
    result=json.loads((ROOT/'dfc-row-map-delivery.json').read_text())
    resolved=[
      'Masknull returns uniformly; otherwise original expertIdx mutated in-place to dummy384 for inactive rows, then originalSyncAll. FullLoad and regularsort produce original expertorder and originalinverseR; flag2 means counts, notcumsum. Keep whole385entry count/dummy domain.',
      'Given originalR, unique mapU[R[t]]=floor(t/topK) preserves original expert/source order without assuming sortties. FullLoad firstsort and one/multicore workspace have forwardrouteIDs; derivingmap fromoriginalinverse also works but costs scatter/read/publication. Keep originalR for finalUnpermute.',
      'Countready has0x800000 sentinel even forzero counts, allranks wait then subtract. GMM consumes allgroup flags including no-workgroups; emptyexpert data omission doesnot permit removing originalpublication/wait protocol.',
      'Original receivergmA is contiguous expertmajor and aliases activation ptrPermutedToken. Deleting inputgmA materialization gives no entirearena capacitycredit. Sourcepeerinput/unique map lifetime changes with directGMMremote reader; conservativewhole originalendjoin must cover everynewreader.',
      'Original endCrossRankSync precedes finalUnpermute, so offsetD and inverseR are stillread afterthatbarrier. They cannotbe retired with inputcodes. Both CombineV1 andV2 consumer boundaries retained; no replacement epilogue or return roundings.',
      '454saved fullEP16 routegroups,7264ranklayercalls and2376dummyrecords;21792finite receiverbytecomparisons/1,984,179,456opaque bytes equal under threewithinexpert tieorders. 80savednativeinverse witnesses give exactoriginalmap inputs for fiveoldroutes. No floating/nativequant/gather test.',
      'Currentowner headerSHA equals pinned originalrouteowner7dcef4c; oldfive diagnosticM3routes sourceobserved21000/smooth0/18cores/oneexpandedrow. Recomputed15roundmax16rankpaireddifferences and output/count hashes. Freepreposition medians1.5425–6.9225us; oldinput-only budget14.2497us, sooriginalfixedconsumer childremainsstopped. Counterfactualnotphysicalbound/fullmodelgain.'
    ]
    pending=result['pending']+[
      'New source-row/scale representation and original native quant perprimitive bit/generation incldummy/padding, not only finaloutputs. Lower mrgsort implementation intentionallyretained; new sorter would reopen originalties/indices gates.',
      'Changed consumer GMM gather-to-L1/Ktile/resource/fixpipe, earlierpergroup publication, largeM/sharedschedule and actualpaidphase remain separatehighvalueparent; firstexpert and fullslab readiness not interchangeable.',
    ]
    review=dict(status='BOUNDED_DFC_ROUTE_MAP_AND_LAST_READERS_WITH_OLD_INPUT_CHILD_PRUNING',sources=sources,resolved=resolved,remaining=pending,
                evidence=['dfc-row-map-dependencies.json','dfc-row-map-delivery.json','dfc-input-delivery-frontier.json','dfc-gemm-tile-frontier.json'],
                precision_contract='EVERY_REPLACEMENT_PRIMITIVE_BITWISE_IDENTICAL',whole_chain_complete=False,new_primitive_admissions=0,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[])
    (ROOT/'dfc-row-map-source-review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'dfc-row-map-source-excerpts.json').write_text(json.dumps(dict(scope='Listedranges only; identity-only headers and retainednativehelper interiors not claimedfullyreviewed.',excerpts=excerpts),ensure_ascii=False,indent=2)+'\n')
    frontier=dict(**review,candidates=[
        dict(scope='OriginalfixedGMM/flags M3 inputpreposition or sourcequant-only standalone',classification='已停止',result='Fiveactualarchivedroutes,syntheticinputs/weights, pairedcounterfactualmediansbelowoldlocalreference. Do notrerun samevariants; not universalwholeparent or nativebit rejection.'),
        dict(scope='Use sorted destinationposition/topK as originalrow, dropdummyinverse entries or dropzeroexpert flags while keepingoriginalreaders',classification='已停止',result='OriginalR/U distinction, fullrouteindex domain and unconditionalgroupflag drains contradictthese specificrecipes.'),
        dict(scope='Originalorder unique map plus changedGMM/earlierpublication/largeM/sharedresources',classification='待确认',result='Concrete perprimitivebits, actualphase/cost, nativeoperandlayout/capture/resources and sourcebanklastreader gaps retained. Not samefixedconsumerbudget.'),
        dict(scope='New native implementation admitted',classification='已确认可实现',result='0; opaqueCPU bytes andoldfivefinaloutputhashes do not qualify anewprimitive.')])
    (ROOT/'dfc-row-map-frontier.json').write_text(json.dumps(frontier,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(sources=len(sources),nonempty_reviewed=sum(bool(s['reviewed_ranges']) for s in sources),excerpts=len(excerpts),native_admissions=0)))

if __name__=='__main__':main()
