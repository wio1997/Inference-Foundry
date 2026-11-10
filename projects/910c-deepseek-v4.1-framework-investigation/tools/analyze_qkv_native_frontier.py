#!/usr/bin/env python3
"""Persist real Q/KV route, bounded native arithmetic and paid larger-parent judgment."""
from hashlib import sha256
from pathlib import Path
import json

ROOT=Path('audit/execution-search')
SUMMARY='介绍：沿真实Q/KV输入、Wq_a/WKV、QR量化、Wq_b到SWA/query读者，扩大到head布局贯穿SMLA并直接交付TP O投影。到了哪里：当前源码/量化描述/142个checkpoint header（payload0），两代表rank各516完整Q/KV段、480目标样本和96真实QLI producer；CANN pertoken/basic/opt舍入边界。结果：保留原三GEMM、其它全免仅1.020%–1.025%窗口；只拆Qb仍被SWA挡住。贯穿SMLA删除原transpose/A2A、Qb取1/4耗时、用旧AG参考付新通信并另付h5us，SWA不变条件净1.228%–1.299%窗口，SWA半省为2.646%–2.723%；不是请求预测。INT32 K1280无溢出不代表最终BF16等价，CPU反例0x44d2对0x44d1证明scale重结合不能盲换。新准入0，全链路未扫全，每替换原语逐位一致。'
NEXT='loop039：按head分布走Wq_b原INT32+原反量化→全局query/selected-ID同代交付→SMLA头/行分核与原MMAD-softmax-PV顺序→inverseRoPE→原TP O投影布局。先核global query metadata缺口、96次global index复用、target24共同资格与零query/草稿20回退、NZ权重视图/缓存代次和graph collective顺序。继续付成本整段判断；只为会改变去留的原tiling/epilogue/新SMLA时间补证，不拿新窗套旧请求分母。'
RANGES={
 'quant_batch_matmul_v3.cpp':[[15,94],[322,430]],
 'quant_batch_matmul_v3_basic_epilogue.h':[[102,139],[196,255]],
 'quant_batch_matmul_v3_pertoken.h':[[453,540]],
 'quant_batch_matmul_v3_pertoken_opt.h':[[507,595]],
 'quant_batch_matmul_v3_pertoken_basic.h':[[244,278],[347,363]],
 'cv_linear.py':[[13,189]],'linear.py':[[130,180],[521,603]],
 'w8a8_dynamic.py':[[20,143]],'dsa_v1.py':[[1620,1650]],
 'dsa_cp.py':[[77,91],[1445,1475]],
 'dsa_v41_cp.py':[[29,75],[78,220],[450,490]],
 'model.py':[[377,448]],'method_adapters.py':[[74,142],[193,224]],
 'linear_op.py':[[145,158],[471,494]],'modelslim_config.py':[[265,349],[655,720]],
}
def main():
 rec={};missing=[]
 for n in ['qkv-native-dependencies.json','qkv-extra-dependencies.json','qkv-route-identity.json']:
  for r in json.loads((ROOT/n).read_text())['sources']:
   if r.get('missing'):missing.append(r['runtime_path']);continue
   p=Path(r['local_path']);b=p.read_bytes();h=sha256(b).hexdigest();assert h==r.get('runtime_sha256',r.get('sha256'))
   ranges=RANGES.get(p.name,[]);assert all(1<=a<=z<=len(b.decode().splitlines()) for a,z in ranges)
   rec[r['runtime_path']]=dict(runtime_path=r['runtime_path'],path=str(p),sha256=h,bytes=len(b),reviewed_ranges=ranges,reviewed_json_fields=['kernelList.tilingKey/taskRation/kernelType','supportInfo','binFileName/declaredSHA/blockDim/coreType'] if p.suffix=='.json' else [],scope='Only stated ranges/fields; empty = identity only. Some unchanged Python ranges carry previous explicit review after seven current source SHA matches. Source identity is not live object or loaded-key proof.')
 review=dict(status='BOUNDED_CURRENT_QKV_ROUTE_AND_CANN_INTEGER_DEQUANT_BRANCHES',sources=list(rec.values()),missing_paths=missing,source_count=len(rec),reviewed_nonempty=sum(bool(r['reviewed_ranges']) for r in rec.values()),whole_chain_complete=False,
  actual_route=['CP enabled constructs ReplicatedLinear Wq_b, all64 heads/32768 channels. Bias=False for Wq_a/Wq_b/WKV.',
   'ModelSlim FLOAT selects AscendUnquantizedLinearMethod for Wq_a/WKV; W8A8_DYNAMIC selects adapter/scheme for Wq_b. Model dtypeBF16 -> scale parameterBF16 despite checkpointF32. Scheme transposes weights then NZ and flattens scale; preserve actual rounded BF16 scales.',
   'Per representative rank516 Qb calls all aclnnQuantMatmulWeightNz/QuantBatchMatmulV3, input INT8/INT8/BF16/FLOAT, outputBF16, raw BlockNum22/MixBlockNum22. Those22 fields are not tilingKey22 and do not prove an active key.',
   'Installed910_93 BF16-scale/NZ compiled option list has25 keys including1:1 and84 with1:2, dynamicblockDim. No runtime key bound.',
   'Native pertoken/basic use INT32 MM output -> AscendDequant with channel scale -> FP32 token multiply -> BF16 CAST_RINT. OPT casts INT32 toFP32, precombines pertoken×channel scales then multiplies, CAST_RINT output. Exact loaded path and intrinsic remain unresolved.',
   'Original restore_tp_heads explicitly transposes [localToken,TP,localHead,D] then all_to_all_single to [allToken,localHead,D]. Candidate extends headlayout through SMLA so O-proj may receive that same desired layout directly.',
   'Existing global cache metadata builder explicitly build_query_metadata=False. Local metadata uses device length minus original suffix and owns sliced global RoPE views; global cache metadata is not ready-made global query metadata.'])
 (ROOT/'qkv-native-source-review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
 prices=json.loads((ROOT/'qkv-joint-delivery-price.json').read_text());small=[];wide=[]
 for p in prices['profiles']:
  small.append(dict(ep=p['ep'],groups=p['groups'],paid_scenarios=p['scenarios']))
  wide.append(dict(ep=p['ep'],**p['head_partition_through_attention']))
 frontier=dict(status='QKV_NATIVE_ORDER_BOUNDED_HEAD_LAYOUT_THROUGH_ATTENTION_PENDING',whole_chain_complete=False,every_replacement_bitwise=True,report='QKV原生反量化与贯穿Attention的head布局判断.md',source_review='qkv-native-source-review.json',price='qkv-joint-delivery-price.json',scalar_counterexample='qkv-dequant-order-cpu.json',confirmed_implementable=[],
  stopped=[dict(scope='Qb-only head partition then restore CP query, unchanged SWA/cache and other arrivals in these representative windows',reason='Cache already later thanQb inall516 instances/rank, so freeQb alone cannot advance join. Added collectives cannot turn this fixed-arrival zero credit positive. Not a complete request/allshape bound.'),
   dict(scope='Universal scale reassociation or treating INT32-safe as final BF16 equality',reason='Finite typed CPU formula counterexample endsBF16 0x44d2 vs0x44d1; active CANN branch/intrinsic not certified. Preserve original exact epilogue.')],
  pending=[dict(scope='Wq_b -> RoPE -> SMLA -> inverseRoPE -> TP O-projection layout, target24 first',value='Qb quarter duration, old largerinputAG as reference/new gathers paid, h5us:1.228%-1.299%window without SWA savings; with SWA half2.646%-2.723%; not request forecast',gaps=['loaded original dequant key/intrinsic and whole primitive bitgate','new global query metadata, causal/sink/head offsets and exact SMLA MMAD/reduce order','QR and96producer selectedID byte gather/epoch reuse, noquery/padded/emptyDP/common group fallback','NZ channel slice/descriptor, BF16 rounded scale and full-weight fallback ownership','SMLA duration/resources, packing/metadata/flags/graph collectives and actual critical TP/DP/request exposure'])],
  capacity_scope='35MiB/rank weight-read payload difference for Qb channel partition is not measured HBM saving or released VRAM. Full-weight fallback may retain40MiB or add5MiB shard copy; no capacity credit.',
  case_scope='Only target-like24rows/480 instances for paid candidate. Existing all16 inventory has fewer nativequant/rope calls onEP7/15; source local zero-query legal branch and draft20 must keep original fallback/common collective qualification. Ordinary/prefill not swept by this two-window price.',
  native_order=dict(INT32_partial_bound=20971520,all_values_exact_FP32=False,finite_formula_counterexample='qkv-dequant-order-cpu.json',native_bit_admitted=False),
  same_scope_prices=small,larger_layout_prices=wide,next_action=NEXT,new_NPU_work=0,new_profile_collection=0,model_loads=0,service_requests=0,remote_writes=[],service_changes=0)
 (ROOT/'qkv-native-frontier.json').write_text(json.dumps(frontier,ensure_ascii=False,indent=2)+'\n')
 notes=dict(previous_goal_turn_classification='PROGRESS: loop037 native cache publication and loop038 fixed-Qb join evidence committed dd98703/public2ad6256, changes next action tolargerQ/KV; no progress count reset needed.',
  observations=['This turn171 hostname check passed. Owned container running, sameimage/id; previous exited observation remains historical, no service health probe or service action.',
   'Missing guessed quantization/ascend.py retained; actual selector discovered at quantization/configs/modelslim_config.py and read.',
   'A local source-print requested linear.py beyondEOF and raisedIndexError after printing598-603; review range capped603. A source-review reader expected sources but schedule review uses source_pins; corrected schema read. Neither native failure.',
   'Full-parent analyzer initially used costly repeatedstream scans; exact typed pools added without altering scope. No native/performance measurement generated.'],
  scalar_scope='Finite IEEE formula counterexample is not CANN intrinsic/native bit proof or complete original model input. Test/falsification admits no replacement.',new_primitive_admissions=0,whole_chain_complete=False)
 (ROOT/'qkv-native-exploration-notes.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(source_pins=len(rec),reviewed_ranges_nonempty=review['reviewed_nonempty'],compiler_json_selected_fields=2,missing=len(missing),new_primitive_admissions=0)))
if __name__=='__main__':main()
