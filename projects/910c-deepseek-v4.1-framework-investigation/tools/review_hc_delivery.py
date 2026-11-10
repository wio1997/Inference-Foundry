#!/usr/bin/env python3
"""Record bounded actual HC arithmetic and early-delivery consumer frontier."""
from pathlib import Path
from hashlib import sha256
import json
ROOT=Path('audit/execution-search')
RANGES={
 'models/deepseek_v41/model.py':[[260,310],[844,920]],
 'device/device_op.py':[[65,73],[2032,2041]],
 'device/hardware_profile.py':[[256,292]],
 'csrc/torch_binding.cpp':[[1390,1453],[1455,1575],[3261,3265],[3426,3434],[3445,3453]],
 'rms_norm_cast_torch_adpt.h':[[1,25]],
 'hc_post/op_host/hc_post_tiling.cpp':[[155,231]],
 'hc_post/op_kernel/hc_post.cpp':[[1,58]],
 'hc_post_d_split.h':[[104,173],[241,361]],
 'hc_pre/op_host/hc_pre_tiling.cpp':[[20,66],[90,159],[260,398]],
 'hc_pre/op_host/hc_pre_tiling.h':[[18,50]],
 'hc_pre/op_kernel/hc_pre.cpp':[[16,48],[90,109]],
 'hc_pre_base.h':[[443,560]],
 'hc_pre_cube_compute.h':[[20,52],[267,343],[418,477]],
 'hc_pre_m_k_split_core.h':[[26,175],[318,452]],
 'rms_norm_cast/op_kernel/rms_norm_cast.cpp':[[1,29]],
 'rms_norm_cast/op_kernel/rms_norm_cast.h':[[1,156]],
 'rms_norm_cast/op_kernel/rms_norm_cast_base.h':[[1,114]],
}
def main():
 d=json.loads((ROOT/'hc-delivery-dependencies.json').read_text());sources=[];excerpts=[]
 for r in d['sources']:
  assert not r.get('missing');p=Path(r['local_path']);b=p.read_bytes();h=sha256(b).hexdigest();assert h==r['runtime_sha256'];l=b.decode().splitlines()
  rr=next((v for k,v in RANGES.items() if r['runtime_path'].endswith('/'+k)),[])
  sources.append(dict(runtime_path=r['runtime_path'],path=str(p),sha256=h,bytes=len(b),reviewed_ranges=rr,scope='Listed ranges only. Empty = identity only; not activebinary/key/bit or fullbody coverage.'))
  for a,z in rr:
   assert 1<=a<=z<=len(l);excerpts.append(dict(runtime_path=r['runtime_path'],sha256=h,first_line=a,last_line=z,lines=l[a-1:z]))
 facts=[
  'Actual _C_ascend HCPreV3 -> npu_hc_pre_v3_npu -> run_hc_pre_fusion -> aclnnHcPre with yBF16, post/comb/preFP32. A3 source selects key0 mixed1:2 Ksplit part1 SyncAll then part2. Key0 source route is not actual captured tilingdata/binary proof.',
  'FFNHCPre receives attn_pre, not its own newpre. With hasPreMix=1, ProcessY uses preMixLocal; newpre/post/comb are later consumers. Therefore arithmetic data dependency permits separate earlycollapse branch while coefficient work remains live.',
  'HCPost uses FP32 Mul and ordered Add for each residualhcIndex then BF16 RINT. ProcessY widens roundedBF16, FP32 multiplies and fourterm leftfold, then BF16 RINT. Fusion must retain both intermediate rounding boundaries.',
  'RMSNormCast BF16: widen -> square -> per-element1/D multiply -> 64lane accumulator and WholeReduceSum -> epsilon/sqrt/div -> x*rstd*gamma inFP32 -> BF16 RINT -> widenBF16 toFP32. FP32router input is intentionally widened roundedoutput, not pre-roundFP32.',
  'HCPre coefficient GEMM toggles HF32 on AB, Ksplit/workspace partials then stage2 reductions. Collapse does not require coefficient branch only when suppliedpreMix present; first/no-premix remains dependent.',
  'FFNPost requires retained roundedresidual, currentpost/comb and MLPresult. ffn_pre returned to nextdecoder. Internalrouter and external F.linear branches both read routinginput; full lower-level factory branch not newly audited here.',
  'Current HCPre allocates at least208MiB workspace by source max(required,default), with castFP32 doublebuffer and per-K partials. Keeping coefficient branch does not free that workspace or provephysicalpeakreduction.'
 ]
 remaining=[
  'Remaining stage2 post/comb/Sinkhorn/pre out and exact nextdecoder/Engram/final pre_mix readers/generations; optional no-premix/zero-row/ordinary/prefill/FULL fallbacks.',
  'Exact active original HCPost/HCPre/RMS compiled binary/tilingdata and actual primitive bit tests; source macros/profile dtype alone cannot admit replacement.',
  'Legal splitoutput graph/capture: independent branch events, coefficient join only at FFNPost/nextpremix, no early alias/overwrite or crossrank collective-order inversion; actual current stream framework to follow.',
  'Native resource sharing: original mixed24/48 and gate/DFC compete. Need paid actualearly cost, MLPslowdown and phase flag/UB buffers; analytic overlap is not free concurrency.',
  'Full original C8 request and sixordinary/prefill/draft/no-premix exposure, noise and regressions. Short mirrored windows not request prediction.'
 ]
 review=dict(status='BOUNDED_HC_NATIVE_ARITHMETIC_AND_PREMIX_EARLY_DELIVERY_REVIEW',sources=sources,resolved=facts,remaining=remaining,whole_chain_complete=False,precision_contract='EVERY_REPLACEMENT_PRIMITIVE_BITWISE_IDENTICAL',new_primitive_admissions=0,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[])
 (ROOT/'hc-delivery-source-review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
 (ROOT/'hc-delivery-source-excerpts.json').write_text(json.dumps(dict(qualification='Exact reviewed excerpts; rawlocal bodies retained bypath/SHA. Empty identities not bodyreview.',excerpts=excerpts),ensure_ascii=False,indent=2)+'\n')
 frontier=dict(**review,candidates=[
  dict(scope='Premix-present HcPost->early originalcollapse/norm->MLP, coefficients parallel with later join',classification='待确认',result='Retain originalHCPre work. Additionalcollapse5us+h5, zeroMLPcontention gives1.33975/1.36849%window;40usMLPslowdown flipsnegative. Source independence is real; cost/events/bit require proof.'),
  dict(scope='Feed pre-round normalizedFP32 directly to router',classification='已停止',result='Violates originalroundedBF16-widened FP32input contract; not an admitted equivalence. Does not stop fusedwiden inside consumer.'),
  dict(scope='Delete persistedHCPost residual because normalizedY is produced',classification='已停止',result='LaterFFNPost reads roundedfourwayresidual; larger recomputation/owner redesign would be distinct paid scope.'),
  dict(scope='Flag-only hasPreMix=0 earlycollapse or skipcoefficient computation',classification='已停止',result='No-premix Y needs newpre; post/comb/pre consumers remain alive. Preserve originalfallback and retainedcoefficient branch.'),
  dict(scope='Native replacement implementations admitted',classification='已确认可实现',result='本轮0; source/analytical price only.')
 ],evidence=['hc-delivery-catalog.json','hc-delivery-dependencies.json','hc-delivery-source-review.json','hc-delivery-source-excerpts.json','hc-early-delivery-price.json','O投影原生资源与SP残差FFN交付判断.md'])
 (ROOT/'hc-delivery-frontier.json').write_text(json.dumps(frontier,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(source_identities=len(sources),reviewed_nonempty=sum(bool(r['reviewed_ranges']) for r in sources),excerpts=len(excerpts),whole_chain_complete=False)))
if __name__=='__main__':main()
