#!/usr/bin/env python3
"""Persist bounded DFC quant-to-first-GMM source review and concrete frontier."""
from pathlib import Path
from hashlib import sha256
import json
ROOT=Path('audit/execution-search')
RANGES={
 'dispatch_ffn_combine.h':[[250,291]],
 'dispatch_ffn_combine_kernel.hpp':[[15,69],[210,257],[313,356],[435,535],[655,930]],
 'moe_init_routing_quant_v2.cpp':[[1,133]],
 'moe_v2_fullload_dynamic_quant.h':[[221,397]],
 'moe_v2_gather_dynamic_quant.h':[[116,208],[450,547]],
 'dispatch_ffn_combine_tiling.cpp':[[1,49],[250,286]],
 'moe_init_routing_quant_v2_tiling.h':[[1,35],[94,159],[342,434]],
 'moe_init_routing_v2_tiling.h':[[1,42],[155,244]],
 'tiling_base.h':[[1,69]],
}
def main():
    sources=[];excerpts=[]
    for name in ['dfc-input-delivery-entry.json','dfc-input-host-dependencies.json']:
        for r in json.loads((ROOT/name).read_text())['sources']:
            p=Path(r['local_path']);b=p.read_bytes();h=sha256(b).hexdigest();assert h==r['runtime_sha256'];lines=b.decode().splitlines();rr=RANGES[p.name]
            sources.append(dict(runtime_path=r['runtime_path'],path=str(p),sha256=h,reviewed_ranges=rr,qualification='Only listed bodies; CPU host execution is not loaded NPU key or numerical/phase qualification.'))
            for a,z in rr:
                assert 1<=a<=z<=len(lines);excerpts.append(dict(runtime_path=r['runtime_path'],sha256=h,first_line=a,last_line=z,lines=lines[a-1:z]))
    resolved=[
      'Original host call fixes quantMode1,smoothNone,dropPad0,expertCount385,AIV40,UB196352. Current originalDoTiling CPU replay12shapes gives localM3 FullLoad21000 and18Compute calls/3unique rows; local416436 calls;816/817/818 Gather11000 with836/837/838 calls. Larger diagnostic3328/6540 source11010. Loaded binary/tilingreceipt remains unqualified.',
      'FullLoad and Gather1H load one original row/Compute per core/loop rowsegment, then repeatedly write expandedRowIdx destinations. Sharing alreadyexists inside segment but no crosscore reuse: TopK partition can split same originalrow. Neither universal uniqueonce nor sixfold perinput is valid.',
      'Original arithmetic is casttoFP32, optionalsmooth(nonselected), Abs/ReduceMax, scalar /127, vectorDiv, FP32->half CAST_TRUNC then INT8 CAST_ROUND. Scale copied in32byte trailingrecord, not interchangeable by formula or separateDynamicQuant name.',
      'Original initRouting writes expanded code+scale in localpeer offsetA, allAIVjoin, count publication/waits and cumsum, then remote CopyGMToGMPerToken reads expertordered packedrecords via UB and writes contiguous workspacegmA plus separate FP32scale1. SyncAll and pergroup CrossCoreSetFlag publish to GMM1. GMM1 consumes those exact contiguous rowdomains and originalB/scales, with allgroup flags drained including zero-workgroups.',
      'AIC starts GMM1, waits beforeGMM2; AIV DispatchAndCombine runsconcurrently. GMM1 already overlaps groupready transport and publishesfirstepiloguebatch at originalgranularity. A new plan cannot also claim allthis overlap or treat wholeDFC as inputphase.',
      '1032 sourceconsistent balanced DFC parents (480target24+36draft20 eachrepresentativerank), allactual localM3. EntiretargetDFC15.979/17.537percentwindow; actualnormready-toFFNPost21.652/23.357percent. Conditional10percentDFC removed andh5 yields1.343/1.496percentwindow. Not actualnewkernel/requestgain.',
      'OriginalC8 2560DFC tasks and interrankclock-qualified common suffix allfree20.090–22.690percent modeledTTFT; half-removing a9.287–10.489percent commonstage withh5 reachesold1percentreference. Thisiswholeparent, not inputquant phase or absolute arbitraryschedulebound.'
    ]
    pending=[
      'Unique sender-row representation must preserve expandedRowIdx/stable expert route order and xActiveMask/dummyexpert mutation, actualsource/receiver ownership, fullcapacity tail and zero-work flags. Read remaining sort/count/indexproducer bodies and guardedconsumer map.',
      'Keep quant codes/scale bitwise, including zero/nonfinite/ties and original FP16TRUNC->INT8ROUND behavior. Unique-row fewerowners reducesparallelism; sharing withsharedexpertDynamicQuant needs separate primitivebit qualification, cannotreuse byshape.',
      'New remote gather or indirect GMM inputcopy needs paid bytes/transactions, originalK/nativecopy/Fixpipe/resources and pergroup publication. Removing receivergmA changes original operand loader; old contiguous copy cannot consume a source-row map.',
      'Crosslayer/crossiteration sourcebank retained until every remote GMMreader completes; originaldispatchcopydone not sufficient for bypass. Native graph/arena/peer/allocator generations and failurecleanup must be proven. Original globalend barrier may provide conservative retirement only after actualnewreaders are bounded.',
      'Original same-load inputquant/route/dispatch phase share, pergroup exposed waits, unique destination multiplicities and newgather/resource slowdown remain unknown. Use existingdata first; only collectnative phase if it changes go/nogo. LargerDFC remains pending while next gaps are scanned.'
    ]
    review=dict(status='BOUNDED_DFC_INPUT_QUANT_ROW_OWNERS_AND_FIRST_GMM_DELIVERY',sources=sources,resolved=resolved,remaining=pending,whole_chain_complete=False,precision_contract='EVERY_REPLACEMENT_PRIMITIVE_BITWISE_IDENTICAL',new_primitive_admissions=0,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],container_observation='Nine currentdisk sources match through readonlydocker-cp stdouttar. DFCentry/hostpins observedsameownedcontainer running; nothealth/restorationproof and no serviceaction bythisscan.')
    (ROOT/'dfc-input-delivery-source-review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'dfc-input-delivery-source-excerpts.json').write_text(json.dumps(dict(scope='Exact listed ranges only, not all linked native bodies or legalbranches.',excerpts=excerpts),ensure_ascii=False,indent=2)+'\n')
    frontier=dict(**review,candidates=[dict(scope='Credit eachTopK route as independentfullrowquant in everyshape, or claim everyinput isquantizedexactlyonce',classification='已停止',result='Source/cpu refute universalclaims. localM3 real6xduplication exists; largeprefill mostlyalreadyshared. Stop incorrectcredit, not rowowner implementation.'),dict(scope='Unique senderrow quant+packedbank+expertordered dispatch+firstGMM parent',classification='待确认',result='Concrete rowmap/mask/flags/lifetime/nativebit/phase/cost gaps listed. Largerwholeparent remains worthdeepening; not admitted.'),dict(scope='Direct shareDFCquant withsharedDynamicQuant fromsameBF16shape',classification='待确认',result='Needs separate originalprimitive bit/scale/cast comparison and allreaders; no algebraicpass. No computecredit untilqualified.'),dict(scope='New native replacements admitted',classification='已确认可实现',result='0. Original hostDoTiling CPU execution is not NPUcorrectness or performance.')],evidence=['dfc-input-delivery-entry.json','dfc-input-host-dependencies.json','dfc-input-host-cpu.json','dfc-input-delivery-price.json','dfc-input-delivery-source-review.json','dfc-input-delivery-source-excerpts.json'])
    (ROOT/'dfc-input-delivery-frontier.json').write_text(json.dumps(frontier,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=len(sources),excerpts=len(excerpts),new_primitive_admissions=0)))
if __name__=='__main__':main()
