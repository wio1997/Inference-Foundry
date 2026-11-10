#!/usr/bin/env python3
"""Persist bounded coefficient/cross-layer consumer and graph policy findings."""
from pathlib import Path
from hashlib import sha256
import json
ROOT=Path('audit/execution-search')
RANGES={
 'models/deepseek_v41/model.py':[[844,920],[972,988],[1060,1097],[1242,1331],[1354,1373],[1420,1490],[1505,1612]],
 'models/deepseek_v41/dspark.py':[[248,277],[311,334]],
 'engram/layer.py':[[1,57]],
 'compilation/acl_graph.py':[[16,132],[156,320]],
 'compilation/updatable_graph.py':[[60,176]],
 'ascend_config.py':[[116,171],[568,585],[836,849]],
 'utils.py':[[627,686],[1156,1164],[1896,1901]],
 'worker/model_runner_v1.py':[[536,550],[815,846],[3186,3212],[4515,4568]],
 'fused_moe/shared_experts.py':[[18,45]],
 'hc_pre_m_k_split_core.h':[[442,522]],
 'hc_pre_base.h':[[428,442],[549,590]],
}
def main():
 deps=ROOT/'hc-coefficient-consumer-dependencies.json';d=json.loads(deps.read_text());sources=[];excerpts=[]
 for r in d['sources']:
  assert not r.get('missing');p=Path(r['local_path']);b=p.read_bytes();h=sha256(b).hexdigest();assert h==r['runtime_sha256'];l=b.decode().splitlines();rr=next((v for k,v in RANGES.items() if r['runtime_path'].endswith('/'+k)),[])
  sources.append(dict(runtime_path=r['runtime_path'],path=str(p),sha256=h,reviewed_ranges=rr,scope='Only listed ranges reviewed, [] identity only. Not activebinary/tilingdata or nativebit/capture proof.'))
  for a,z in rr:
   assert 1<=a<=z<=len(l);excerpts.append(dict(runtime_path=r['runtime_path'],sha256=h,first_line=a,last_line=z,lines=l[a-1:z]))
 facts=[
  'Both actual target and DSpark entry allocate FP32pre_mix zeros and set firstcoefficient1.0 before firstdecoder. Decoderattention consumes passedpre_mix; FFN consumes attn_pre; returnedffn_pre feeds nextdecoder. None fallback is not the actualfirstattention in these two forwards.',
  'Original stage2 writes newpre and post, computes FP32softmax+epsilon and fixed iterTimes row/column Sinkhorn, then comb. Post/comb neededby samephaseHCPost; newpre remainsnextphase/decoder live. Newarithmeticoutputs cannot be discarded when moving oldpremix collapse earlier.',
  'Own wholecoefficient completion joined before ownHCPost conservatively orders laterDSpark auxmean snapshots (beforeEngram), Engram residual updates, nextpremix reads, guardedMTP fullstate/gathers and finaloriginalcollapse. Final torch.sum is not proven bit-equivalent to HCPre ProcessY leftfold.',
  'Current framework captures runnable in a graphpool, retains graph outputs/workspaces through weakrefs, dispatches NONE/mismatched modes eagerly, and may synchronize main before FULL replay dependingENPU/draft flags. Newfork must preserve generation/storage rather than treating Pythonreturn or tupleallocation as completion.',
  'UpdatableGraph ExternalEvent wait/reset gates host taskmetadata updates, not a reusable coefficientdone event. Existing capturedEngram producer has SP/parallel guards, and frozen source/savedconfig has Engram overlapfalse and fusedMC2 disables sharedexpert overlap. None of these existing optional paths grants newHCstream legality.',
  'A proposed single dedicated coefficient stream must acquire after actualmain input/Engram writes, retain oldpremix/X plus alloriginal outputs/workspace, run originalHCPre and join before each originalPost; no coefficient HCCL moved or new crosscard collective order assumed.',
  'Saved aicore_time/aiv_time exist for all960 targetHC calls EP0 and903/960 EP8. LiteralN/A is unknown, not zero or absentwork. Engine-duration fields are pressure sensitivities only, not exclusive phases, freecapacity or actual concurrent slowdown.'
 ]
 remaining=[
  'Exact native original ProcessY primitive bits for separate/fusedearly branch; original inputRMS retained, FFN BF16->FP32 widening and all intermediate rounding mandatory.',
  'Actual FULLcapture/multistream Event and allocator/workspace ownership, perbucket generations and partial-submit/error cleanup; source/finite DAG does not prove runtime.',
  'Actual earlybranchcost and resourcecontention. Currenttwo-parent conditional e5/h5/delta0 net2.6717/2.7395%window; PMU-pressure proxy canreduce below.5% or negative, not a measuredstop bound.',
  'Original C8 fullrequest/sixordinary, draft, zero-query/padding and largeprefill shape/crosscard regressions. Public prices remain two short mirroredload windows, not requestgain.',
  'Next unclosed larger delivery: global Engram lookup rows through replicatedWKV, rotated gate and nextHC input; derive current consumer/weight/layout and wholepaidprice before any small gate recipe.'
 ]
 review=dict(status='BOUNDED_PREMIX_SEEDS_COEFFICIENT_TAIL_CROSS_LAYER_AND_GRAPH_POLICY_REVIEW',sources=sources,resolved=facts,remaining=remaining,precision_contract='EVERY_REPLACEMENT_PRIMITIVE_BITWISE_IDENTICAL',whole_chain_complete=False,new_primitive_admissions=0,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[])
 (ROOT/'hc-coefficient-consumer-source-review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n');(ROOT/'hc-coefficient-consumer-source-excerpts.json').write_text(json.dumps(dict(qualification='Exact reviewed ranges only. Identity-only files not bodycoverage. Rawexistingbody paths/SHA retained.',excerpts=excerpts),ensure_ascii=False,indent=2)+'\n')
 frontier=dict(**review,candidates=[
  dict(scope='Both attention and FFN premix earlydelivery with wholeoriginalcoeff retained and ownPost joins',classification='待确认',result='Actualseed/consumer source supports bothparents inclfirstlayers; sixfinite normalDAGs andthreebadrecipewitnesses. Need nativebits/streamcapture/ownership/paidcontention, not allfound or runtimeadmission.'),
  dict(scope='Treat actualfirsttarget/draft attention as no-premix; borrow unrelated externalmetadataevent',classification='已停止',result='Actualseeds non-None. Graphupdate/Engram event generations are not newcoefficientdone proof.'),
  dict(scope='Missingproducerfence, skipconsumerwithoutcleanup, or drop originalcoeffoutputs',classification='已停止',result='Finitewitnesses plus realpost/comb/newpre readers reject exactrecipes; fullprivateowner/fence design retained.'),
  dict(scope='Native newimplementations admitted',classification='已确认可实现',result='0; exactsource/finiteDAG/paidmodels not nativebit or requestproof.')
 ],evidence=['hc-coefficient-consumer-source-review.json','hc-coefficient-consumer-source-excerpts.json','hc-both-delivery-price.json','hc-fork-join-cpu.json','hc-pmu-a1-invalid.json'])
 (ROOT/'hc-coefficient-consumer-frontier.json').write_text(json.dumps(frontier,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=len(sources),nonempty_reviewed=sum(bool(r['reviewed_ranges']) for r in sources),excerpts=len(excerpts),all_sha_match=True)))
if __name__=='__main__':main()
