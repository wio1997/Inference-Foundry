"""HC fork with typed index query/weight and scalar-product publication cuts."""
from pathlib import Path
from collections import Counter
from itertools import product
import hashlib,json,math,struct
from finalize_origin_coverage_v9 import N,O,W,read,write


def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def half(x):
    try:return struct.unpack('e',struct.pack('e',float(x)))[0]
    except OverflowError:return math.copysign(math.inf,x)


def f32(x):return struct.unpack('f',struct.pack('f',x))[0]


def times(r):
    def calc(fork,early,late,compact):
        u=(r['y']if fork else max(r['y'],r['c']))+r['n'];q=u+r['q'];w=(u if early else q)+r['w'];rope=q+r['r']
        release=rope if late else max(rope,w);cast=release+r['t'];quant=max(cast,w)+r['v'];other=cast+r['d']
        if compact:quant+=r['f']+r['p'];qli=max(other,quant)
        else:qli=max(other,quant,w)+r['f']
        coeff=r['c']+(r['cd']if early else 0)
        return max(coeff,qli+r['o'])
    return dict(baseline=calc(False,False,False,False),HC_fork_only=calc(True,False,False,False),early_weight=calc(True,True,False,False),
        early_weight_late_join=calc(True,True,True,False),compact_product_in_aux=calc(True,True,True,True))


def main():
    assert not(O/'REVIEW-CHECKPOINT-v139.json').exists()
    prof=read(O/'hc-index-query-weight-native-producers-v139.json');legacy=read(O/'index-projection-native-frontiers-v106.json');root=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend';parts=[]
    for rel,a,b in [('models/deepseek_v41/model.py',887,920),('models/deepseek_v41/indexer.py',34,74),('models/deepseek_v41/indexer.py',167,179),('models/deepseek_v41/indexer.py',251,286),
        ('attention/dsa_v41.py',396,430),('attention/dsa_v41.py',467,499),('attention/dsa_v41.py',523,532),('attention/dsa_v41.py',580,587)]:
        p=root/rel;parts.append(dict(**pin(p),first_line=a,last_line=b,source=p.read_text().splitlines()[a-1:b]))
    p=N/'attention-operator-source-v60/vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/quantize_indexer_query.py';parts.append(dict(**pin(p),first_line=11,last_line=70,source=p.read_text().splitlines()[10:70]))
    p=N/'attention-operator-source-v60/vllm-workspace/vllm-ascend/csrc/attention/quant_lightning_indexer_v2/op_kernel/arch22/quant_lightning_indexer_v2_service_vector_arch22.h';parts.append(dict(**pin(p),first_line=723,last_line=766,source=p.read_text().splitlines()[722:766]))
    write(O/'hc-index-typed-input-and-round-contract-v139.json',dict(parts=parts,
        reused_answers=['v64BF16Weight→FLOAT/64→FP16 and query-vs-keyquant','v82IndexQBF16/RoPE/head128quant','v83distinctQR1280-vs-hidden5120inputs/FP16product','v106QAWeightpanel/earlyWeightstart-latejoin/sameauxstream/splitKrounding','v93HCraw4HvsYinformationandPost/pre4deadline'],
        facts=[
            'HC current coefficient consumes raw4H/FLOAT-HF32 X; normalized BF16 Y-hidden5120 feeds QA and IndexWeight. IndexQ instead consumes separately normalized QR1280 after QA, then activation INT8 quantization and quantized4096 projection. These are not one shared-input HC/IndexQ/Weight GEMM.',
            'QA and Weight can share source input panels under current row/epoch/layout/math gates; a joint1280+32 output must keep QA1280 norm separate and Weight32 unnormalized. Distinct matrices do not prove value equality or inequality. PackedN1312 views need a new stride ABI or explicit adapter; separate outputs are an alternative.',
            'Two source calls quantize the same QR expression for QB and IndexQ. Matching DynamicQuant descriptors/source expression is not historical address/value/quantizer identity proof. Conditional reuse requires same QR generation/rounding/quant method and all consumers/lifetimes; no extra HC-specific quant-CSE credit beyond that chosen graph.',
            'Keep BF16Y/full5120normBF16/QA BF16/QR1280normBF16 and IndexQ BF16→64tailRoPEBF16→amax128. Quantizer scale is RN_FP16(div_rn_FP32(amax,127)),clampmin2^-24; INT8 quant uses that rounded scale with div_rn/nearbyint/clamp. PreRoPEamax or early-partialheadamax is not equivalent.',
            'Per-head amax requires all128 post-RoPE values, not all tokens. A finite-value max(nonrot64) and max(rotated64) decomposition is possible under exact ownership/NaN/round semantics. New head/row release still needs actual tile/native publication/scale16-head-group32B ownership; current whole kernel boundary cannot expose arbitrary partial tiles.',
            'Weight branch keeps BF16 MatMul output→FLOAT*2^-6→FP16 before the native FP16 Weight*Qscale product. Fused epilogues may keep required rounds privately; scaled FP16 first or dropping intermediate FP16 Weight has counterexamples already in v64/v106.',
            'Weight starts after IndexQ event in current source, overlaps main query RoPE, then main waits Weight before compression/query-quant publication. Weight and IndexQuant share one aux stream. Removing main early wait does not make those aux tasks concurrent; earlyWeight also competes with QA/IndexQ/HCcoeff Cube work.',
            'Compact FP16 alpha=RN_FP16(RN_FP16(WeightBF16/64)*QscaleFP16) can replace separate QLI weight/scale planes only if all observers/ABI/readers close; Qscale is still used inside INT8 quantization. Existing QLI M16-broadcast product workspace remains unless independently redesigned. Retaining old planes adds compact-plane traffic instead of saving it.',
            'A separate finite-value compatibility variant may feed compact alpha as existing QLI Weight with a persistent FP16+1 queryScale plane, only in the pinned arch22 product-only Qscale-use domain with original native identity/FTZ/nonfinite/readers/capture gates. It keeps current QLI entry ABI and broadcast/product work but adds unit-plane reads/storage; it does not have the same traffic credit as a new alpha-only ABI.',
            'Fusing alpha into aux IndexQuant changes first WeightFP16 reader. Leaving main WeightFP16 conversion after main waits IndexQuant_done creates a producer-consumer wait cycle. Move cast into earlier aux/fused computation with original rounding and correct publication, or keep product in QLI; not a free event deletion.',
            'Currentfull4Hcoeff/postcomb/nextFFNpre4/rawXPost remain. Reprice one HC/QA/IndexQ/Weight/quant/cache/QLI/attention continuation graph and joint Cube/Vector/link resources. Saved TASK interval overlap is not hardware utilization, causal overlap availability or actual gain.'
        ],new_native_math_alias_all_readers_or_speedup_proved=False,new_device_work=False,overall_complete=False))
    oldkeys={(r['database_index'],r['stages']['IndexQ']['task_record_rowid'])for r in legacy['frontiers']};newkeys={(r['database_index'],r['producers']['INDEX_Q']['TASK']['task_record_rowid'])for r in prof['records']};assert len(oldkeys)==640 and len(newkeys)==192 and not oldkeys&newkeys
    stats=Counter();geoms=Counter();legacy_serial=0
    for r in legacy['frontiers']:legacy_serial+=r['stages']['IndexWeight']['streamId']==r['stages']['IndexQuant']['streamId']
    for r in prof['records']:
        ob=r['observed'];stats['query_before_weight']+=ob['IndexQ_TASK_end_before_Weight_TASK_start'];stats['weight_quant_same_aux']+=ob['Weight_and_IndexQuant_same_stream'];stats['weight_rope_interval_overlap']+=ob['Weight_RoPE_TASK_interval_overlap_ns']>0
        if ob['IndexQuant_C2KV_TASK_interval_overlap_ns']is not None:stats['C2_frontiers']+=1;stats['quant_C2KV_interval_overlap']+=ob['IndexQuant_C2KV_TASK_interval_overlap_ns']>0
        p=r['eager_HC_sparse_parent']['eager_HC_SP_parent'];m=p['SP_allGather']['COMM']['count']//5120;q=int(r['producers']['WEIGHT']['definition']['outputShapes_text'].strip('"').split(',')[0]);global_rows=int(p['SWA_KV']['definition']['inputShapes_text'].strip('"').split(',')[0]);geoms[m,q,global_rows]+=1
    write(O/'hc-index-profile-coverage-and-streams-v139.json',dict(new_eager_frontiers=192,legacy_graph_frontiers=640,combined_known_saved_index_frontiers=832,disjoint_IndexQ_TASK_keys=True,
        legacy_root_bound=512,legacy_phase_unbound=128,legacy_same_aux_stream=legacy_serial,eager_observations=dict(stats),
        no_native_promotion_or_counterfactual_overlap_proved=True,new_device_work=False,overall_complete=False))
    cases=[];classes=Counter()
    for y,c,n,q,w,r,t,d,v,f,p,cd,o in product([0,5],[0,5,20],[1,3],[1,5],[1,5],[1,3],[0,2],[1,8],[1,3],[1,2],[0,2],[0,5],[0,8]):
        a=dict(y=y,c=c,n=n,q=q,w=w,r=r,t=t,d=d,v=v,f=f,p=p,cd=cd,o=o);a.update(times(a));cases.append(a)
        tag='COMPACT_FASTER'if a['compact_product_in_aux']<a['early_weight_late_join']else'COMPACT_SLOWER'if a['compact_product_in_aux']>a['early_weight_late_join']else'TIED';classes[tag]+=1
    write(O/'hc-index-aux-stream-and-coeff-DAGs-v139.json',dict(cases=cases,case_count=len(cases),compact_vs_separate_FP16_classes=dict(classes),
        symbols='y:externalYready,c:HCcoeffready,n:fullHnormservice,q:QA/QRnorm/QB/IndexQready path,w:WeightBF16+FLOATscale service,r:IndexRoPEservice,t:eligible main cast/prequant service,d:other compression/cache-ready tail,v:IndexQuantservice,f:WeightFP16cast,p:extra compact-product service,cd:assumedearlyWeight coeffcontention delay,o:common QLI/attention/output continuation service.',
        continuation_note='Common QLI/attention/output continuation o is priced BEFORE the final coeff join, so its possible hiding of the coefficient deadline is retained. Endpoint is HCPost INPUT readiness; common unchanged Post service and later pre4/FFN are not priced. Additional queue/resource costs need repricing.',
        domain='Hypothetical units,otherwise noncontending resources except ordered aux Weight→IndexQuant. Current Weight starts after IndexQ; early variants start at fullHnormready. Late join releases eligible main work after RoPE while aux quant still waits Weight. Compact mode moves original FP16 Weight cast into quant before product. Services not measured/native optimal-time bounds.',new_device_work=False,overall_complete=False))
    # Finite reference counterexamples; no torch/CANN/device execution.
    pre=1.0;rot=0.70703125;nonrot=.5;pre_scale=max(half(f32(pre/127)),2**-24);post_scale=max(half(f32(rot/127)),2**-24)
    write(O/'hc-index-scalar-round-and-cycle-witnesses-v139.json',dict(
        intermediate_FP16=read(O/'index-projection-rounding-and-readiness-v106.json')['intermediate_FP16_counterexample'],
        FP16_too_early=read(O/'index-context-operator-contract-v64.json')['witnesses']['wrong_Float16_first'],
        RoPE_amax=dict(pre_RoPE_tail=[1,0],post_RoPE_tail_BF16=[rot,rot],nonrot_value=nonrot,pre_scale=pre_scale,post_scale=post_scale,
            wrong_pre_RoPE_quant=round(f32(nonrot/pre_scale)),correct_post_RoPE_quant=round(f32(nonrot/post_scale)),scope='Finite two-coordinate45degree rotation after BF16round,other128headfeatures≤.5. Not actual position/CANN native proof.'),
        partial_head=dict(first_value=1,possible_late_value=127,wrong_early_quant=127,correct_after_full_head_quant=1,scope='Complete perhead128amax changes earlier quant values; per-head information dependency,not whole-op barrier.'),
        fused_alpha_wait_cycle=dict(edges=[['MAIN_WEIGHT_FP16_DONE','AUX_QUANT_PRODUCT_DONE'],['AUX_QUANT_PRODUCT_DONE','MAIN_WEIGHT_FP16_START'],['MAIN_WEIGHT_FP16_START','MAIN_WEIGHT_FP16_DONE']],
            original_source='Mainwaitprepared.quantize_done before select_projected line256 WeightFP16. NewauxproductrequiresWeightFP16;unchanged placementcycles.',
            acyclic_repairs=['Perform required WeightFP16 rounding inside aux/fused quant before product,with publication/readiness/lifetime contract.','Move separate WeightFP16 task before quant first-reader and price event/stream/layout changes.','Keep product in QLI and separate original Weight/query-scale inputs.']),
        reference_only_not_native_parity=True,new_device_work=False,overall_complete=False))
    models=[]
    for (m,q,global_rows),cnt in sorted(geoms.items()):
        models.append(dict(sender_rows=m,query_rows=q,global_KV_rows=global_rows,frontiers=cnt,Weight_BF16_or_scaled_FP16_query_value_content=64*q,
            Weight_fixed_sender_value_content=64*m,query_head_scale_FP16_content=64*q,compact_alpha_FP16_content=64*q,
            both_old_scalar_planes_private_store_read_value_IO=256*q,compact_alpha_private_store_read_value_IO=128*q,
            persistent_unit_scale_read_value_content=64*q,compact_alpha_plus_unit_plane_scalar_value_IO=192*q,persistent_unit_plane_at_this_query_extent_content=64*q,
            additional_alpha_IO_if_old_planes_kept=128*q,QLI_declared_M16_product_broadcast_store_read_value_IO=2048*q,
            query_INT8_value_content=4096*q,query_INT8_plus_scale_content=4160*q,query_BF16_4096_content=8192*q,
            QA_Weight_joint_input_one_read_reference=2*5120*q,separate_two_input_reads_reference=4*5120*q,
            optional_packedN1312_to_two_contiguous_outputs_value_IO=4*1312*q,
            QA_BF16_weights=13107200,Weight_BF16_weights=327680,QA_Weight_matrix_value_contents=13434880,HC_FLOAT_coeff_matrix_value_contents=1966080,
            IndexQ_INT8_matrix_value_content=5242880,IndexQ_BF16_dequant_plane_content=8192,all_named_matrix_planes_value_content=20652032,
            no_actual_HBM_wire_allocation_necessary_all_algorithm_or_native_tiling_proof=True))
    write(O/'hc-index-product-delivery-and-work-v139.json',dict(models=models,
        variants=[dict(id='KEEP_RECEIVER_WEIGHT_AND_QUERY_SCALE',contract='Original FP16 weight/FP16 queryScale input ABI to QLI; preserve independent typed producers and all rounds.'),
            dict(id='DELIVER_WEIGHT_BF16_OR_SCALED_FP16',contract='64B/queryrow either cut; prior-to-SP owner cut additionally needs actual query/rank rowmap,pad/trim,param replicas/newM. BF16 delivery leaves F32/64/FP16 at receiver; scaledFP16 retains original projection round before scaling.'),
            dict(id='COMPACT_ALPHA_FROM_AUX_QUANT',contract='INT8query uses original rounded FP16Qscale; compute original FP16Weight first and FP16 product,emit compact alpha if all old plane readers closed,newQLIABI. Preserve scale16head32B write ownership or new proven layout. Break source waitcycle by moving required cast,not deleting wait only.'),
            dict(id='COMPACT_ALPHA_IN_QLI',contract='Original producers stay; QLI consumes raw planes and may fuse preparation internally. Existing broadcast16 workspace/crosscore handoff is a separate algorithm/layout cost,not automatically deleted.')],
        existing_QLI_unit_scale_variant=dict(id='COMPACT_ALPHA_WITH_PERSISTENT_UNIT_SCALE',contract='In pinnedarch22 Qscaleproduct-only finite-alpha/nativeidentity domain, feed compactalpha as Weight and read persistentFP16+1 Qscale. Keep existingQLIentryABI/product/broadcastwork;priceunitbankcapacity/reads/initialization/lifetime/newpositions and all original observers. NativeFTZ/nonfinite/loadedkernel/otherarchitecture-gates remain open.'),
        one_graph='Compact alpha is one chosen replacement cut; if old weight/Qscale outputs remain, alpha adds content. Original QLI broadcastM16 plane not erased by compact alpha. QAWeightpanel,QRquantreuse,earlyWeight,latejoin/FP16epilogue andHCfork share one graph,one warmgrant and joint resource cut.',
        theory='SYMBOLIC_CEILING only: necessary servicedomain work and valid joint mixed HF32/BF16/INT8/Vector/link capacity upper still needed. Matrix/plane contents and conventional dense references are not measured HBM/coldworking-set/all-algorithm work or physical speedup/MFU.',
        new_device_work=False,overall_complete=False))
    write(O/'hc-index-composition-clauses-v139.json',dict(new_pairs=[['F01','F09'],['F01','F10']],new_pair_clauses=2,new_groups=0,
        group_extensions=[['HG93-HC-ATTENTION-FORK-DEADLINE','F09'],['HG93-HC-ATTENTION-FORK-DEADLINE','F10']],
        clauses=[dict(id='REL139-01',a='F01',b='F09',contract='HCexternalY/fullHnorm→QA/QRnorm→quantizedIndexQ→BF16/RoPEBF16/head128quant is a nonlinear typed chain,not HCraw4Hcoef GEMM. Preserve all reductions/rounds and actual QRquant replica/epoch/readers. Perhead/row pipelining requires new publication/ownership; moving amax beforeRoPE or incompleteheadchangesINT8. Currentcoeff/Post/pre4 andsharedresource deadlinesremain.'),
            dict(id='REL139-02',a='F01',b='F10',contract='Weight reads normalizedhidden5120,may share QA input or start early after that input ready; IndexQ reads QR1280. Keep BF16Weight→F32/64→FP16→FP16product. Late join does not parallelize sameauxWeight/IndexQuant; fusedalpha changesfirstreader andmustbreak FP16cast/quantwaitcycle. Price originalscalar/broadcastplane/readers/nativeM/pad/rank andHCcoeff/Post/pre4 inonegraph.')],
        all_runtime_gates_proved=False,critical_path_gain_us=None,whole_pair_or_all_subsets_closed=False,new_device_work=False,overall_complete=False))
    names=['hc-sparse-postcheck-result-v138p.json','candidate-relation-clauses-v138.json','hc-index-query-weight-native-producers-v139.json','hc-sparse-eager-native-frontiers-v138.json',
        'index-projection-source-contract-v106.json','index-projection-native-frontiers-v106.json','index-projection-rounding-and-readiness-v106.json','index-weight-partition-and-precision-v106.json',
        'index-context-operator-contract-v64.json','Q_B-index-epilogue-extension-v82.json','attention-joint-interface-models-v83.json','HC-attention-deadline-and-precision-models-v93.json']
    paths=[O/n for n in names]+[Path(p['path'])for p in parts]
    write(O/'hc-index-inputs-v139.json',dict(inputs=[pin(p)for p in sorted(set(paths))],profile_first=True,
        questions=['Which eager IndexQ/Weight/QR-quant/source streams are bound to current HC frontiers?',
            'Which distinct input/RMS/head-amax/FP16 boundaries survive HC/producer fusion?',
            'Can compact Weight*Qscale publication and an earlier Weight launch improve one HC graph without event cycles or overlapping interface credits?'],new_device_work=False,overall_complete=False))
    print(json.dumps(dict(status='HC_INDEX_QUERY_WEIGHT_COMPOSITION_REVIEWED',new_eager_frontiers=192,combined_known_frontiers=832,DAG_cases=len(cases),compact_classes=dict(classes),eager_observations=dict(stats),new_scoped_pairs=2,new_device_work=False)))


if __name__=='__main__':main()
