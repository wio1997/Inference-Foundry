"""HC captured regimes, exact graph edges and source-conditioned work/IO models."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter,defaultdict
import hashlib,json,math,statistics,bisect
from finalize_origin_coverage_v9 import N,O,read,write
from mirror_saved_profile_core_v53 import CACHE

def stat(xs):
    a=sorted(xs)
    return dict(n=len(a),minimum=a[0],median=statistics.median(a),p90=a[int(.9*(len(a)-1))],maximum=a[-1]) if a else None
def shape(s):return [[int(x) for x in a.split(',') if x] for a in s.strip('"').split(';')]
def ceildiv(x,y):return (x+y-1)//y
def up(x,y):return ceildiv(x,y)*y

def main():
    assert not (O/'HC-operator-profile-and-edges-v58.json').exists()
    stamp=datetime.now(timezone.utc).isoformat()
    inputs=read(O/'HC-operator-inputs-v58.json');src={}
    for r in inputs['files']:
      if r['status']=='READ':
        p=Path(r['local_path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'];src[p.name]=(r,p.read_text())
    assert len(src)==8
    configrow=next(r for r in read(O/'sampling-guard-inputs-v55.json')['files'] if r['path'].endswith('/config.json'))
    cfg=read(Path(configrow['local_path']))['text_config']
    assert [cfg['hc_mult'],cfg['hc_sinkhorn_iters'],cfg['hc_eps'],cfg['rms_norm_eps']]==[4,20,1e-6,1e-20]
    led=read(O/'operator-shape-ceiling-ledger-v55.json')
    regs=[r for r in led['observed_regimes'] if r['operator'] in ['HcPre','HcPost']]
    counts=Counter();Ms=set();models=[];rows=[];h=4;H=5120;K=h*H;C=(h+2)*h;B=1.6e12
    for r in regs:
      ins=shape(r['signature']['inputShapes_text']);op=r['operator'];m=ins[0][0] if op=='HcPre' else ins[0][1]
      counts[op]+=r['executions'];Ms.add(m)
      if op=='HcPre':assert ins==[[m,h,H],[C,K],[3],[C],[m,h]]
      else:assert ins==[[1,m,H],[1,m,h,H],[1,m,h],[1,m,h,h]]
      rows.append(dict(regime_id=r['regime_id'],operator=op,M=m,origin=r['origin'],context=r['context'],executions=r['executions'],
        profile_ns=r['measured_native_task_duration'],signature=r['signature'],all_observed_pre_mix_present=op=='HcPre'))
    assert len(regs)==16 and counts=={'HcPre':8944,'HcPost':8944}
    for m in sorted(Ms):
      preQ=4*K*C+2*m*K+2*m*H+4*m*(C+h)+4*C+12
      postQ=2*m*H*(1+2*h)+4*m*(h+h*h)
      mdim=min(24,ceildiv(m,256));single=up(ceildiv(m,mdim),16)
      blockM=[min(single,max(0,m-j*single)) for j in range(mdim)]
      assert sum(blockM)==m
      padded=sum(up(x,16) for x in blockM)
      splitK=up(ceildiv(K,24//mdim),256);rk=ceildiv(K,splitK)
      preExtra=10*m*K
      partialValues=8*m*C*rk+128*m*rk
      # These are logical value payloads, not measured external-memory transactions.
      models.append(dict(M=m,H=H,h=h,K=K,C=C,
        pre_dense_projection_HF32_ops_2perMAC=2*m*K*C,
        pre_semantic_sumsq_conventional_ops=2*m*K,
        pre_collapse_FP32_mul_add_ops=(2*h-1)*m*H,
        pre_weight_FLOAT_bytes=4*K*C,
        pre_minimum_descriptor_value_bytes=preQ,pre_cold_reference_io_floor_us=preQ/B*1e6,
        pre_source_additional_cast_and_reread_value_bytes=preExtra,
        pre_source_partial_projection_and_Gram_value_bytes=partialValues,
        source_conditioned_tiling=dict(AIC_count_assumption=24,m_partitions=mdim,splitM=single,K_partitions=rk,splitK=splitK,
          real_M_per_partition=blockM,requested_rounded_M_sum=padded,
          HF32_AB_requested_ops=2*padded*K*C,
          Gram16x16_requested_ops=2*padded*16*K,
          source_cast_FP32_workspace_write_read_bytes=8*m*K,
          source_second_BF16_input_read_bytes=2*m*K,
          qualification='Reconstructed from exact source and 24/48 profile launch descriptor, not native tiling/instruction/binary attestation. Gram computes 16x16 blocks and uses diagonal; AB explicitly enables HF32. Requested N=24 is not a physical issued-N claim.'),
        post_FP32_multiplications=m*H*(h+h*h),post_FP32_additions=m*H*h*h,
        post_total_FP32_mul_add_ops=m*H*(h+2*h*h),post_input_BF16_to_FP32_values=m*H*(h+1),post_output_FP32_to_BF16_values=m*H*h,
        post_minimum_descriptor_value_bytes=postQ,post_cold_reference_io_floor_us=postQ/B*1e6,
        pre_to_norm_removable_BF16_Y_write_read_bytes=4*m*H,
        post_to_next_collapse_removable_one_residual_read_bytes=2*m*h*H,
        pre_Sinkhorn_normalization_rounds_after_softmax=39,
        pre_Sinkhorn_divisions_after_softmax=39*m*h*h,
        theoretical_compute_time_unbound='Physical HF32/A2/Vector instruction peaks and special-function/normalization dependency rates unbound. Do not substitute BF16/INT8 peak.',
        qualification='Descriptor-value cold-reference model; does not assert compulsory HBM reads, actual cache traffic, speedup or MFU. Whole source model adds GM payloads but cache tier/padding/duplicate weight requests are unpriced.'))
    # The native record IDs define role edges; proximity alone is never used to infer a producer.
    graph=read(CACHE/'analysis/native-execution-identity-v14/graph-blocks-PMU-and-MC2-v14.json')
    bydb=defaultdict(list)
    for g in graph['graph_executions']:bydb[g['database_index']].append(g)
    edges=[];times=defaultdict(list);exposure=[]
    for i in range(32):
      d=read(CACHE/f'analysis/native-execution-identity-v14/db-{i:02d}.json')
      tasks={r['task_record_rowid']:r for r in d['TASK_records']};defs={r['globalTaskId']:r for r in d['COMPUTE_definitions']}
      compute=sorted([t for t in tasks.values() if t['globalTaskId'] in defs],key=lambda t:(t['startNs'],t['task_record_rowid']))
      starts=[t['startNs'] for t in compute]
      def op(t):return defs[t['globalTaskId']]['opType_text']
      def edge(kind,g,b,a,z):
        assert a['endNs']<=z['startNs']
        lo=bisect.bisect_left(starts,a['endNs']);hi=bisect.bisect_left(starts,z['startNs'])
        interval=[x for x in compute[lo:hi] if x['task_record_rowid'] not in [a['task_record_rowid'],z['task_record_rowid']] and x['endNs']>a['endNs']]
        edges.append(dict(kind=kind,database_index=i,graph_ordinal=g['graph_ordinal'],block_ordinal=b['block_ordinal'],root_bound=bool(g['prepare_root_keys']),
          producer_task_rowid=a['task_record_rowid'],consumer_task_rowid=z['task_record_rowid'],producer_op=op(a),consumer_op=op(z),
          producer_duration_ns=a['endNs']-a['startNs'],consumer_duration_ns=z['endNs']-z['startNs'],gap_ns=z['startNs']-a['endNs'],
          same_stream=a['streamId']==z['streamId'],intervening_compute_by_start=dict(Counter(op(x) for x in interval))))
      for g in bydb[i]:
        totalHC=0;previousPost=None
        for b in g['blocks']:
          ts=sorted([tasks[rid] for rid in b['task_record_rowids']],key=lambda t:(t['startNs'],t['task_record_rowid']))
          pre=[t for t in ts if op(t)=='HcPre'];post=[t for t in ts if op(t)=='HcPost']
          assert len(pre)==len(post)==2
          assert pre[0]['startNs']<post[0]['startNs']<pre[1]['startNs']<post[1]['startNs']
          for t in pre+post:totalHC+=t['endNs']-t['startNs']
          for which,pt in zip(['ATTENTION','FFN'],pre):
            norm=ts[ts.index(pt)+1]
            assert op(norm)==('RmsNorm' if which=='ATTENTION' else 'RmsNormCast')
            assert shape(defs[norm['globalTaskId']]['inputShapes_text'])==[[3,H],[H]]
            edge(which+'_PRE_TO_NORM',g,b,pt,norm)
            if g['prepare_root_keys']:
              times[which+'_HcPre'].append(pt['endNs']-pt['startNs']);times[which+'_Norm'].append(norm['endNs']-norm['startNs'])
          if g['prepare_root_keys']:
            for which,t in zip(['ATTENTION','FFN'],post):times[which+'_HcPost'].append(t['endNs']-t['startNs'])
          edge('INTRALAYER_POST_TO_PRE',g,b,post[0],pre[1])
          merge=[t for t in ts if op(t)=='ForeachAddList'];assert len(merge)==1
          edge('FFN_MERGE_TO_POST',g,b,merge[0],post[1])
          if previousPost is not None:edge('CROSSLAYER_POST_TO_PRE',g,b,previousPost,pre[0])
          previousPost=post[1]
        if g['prepare_root_keys']:
          exposure.append(dict(database_index=i,graph_ordinal=g['graph_ordinal'],HC_task_sum_ns=totalHC,graph_device_control_span_ns=g['graph_control_pair_ns'],
            ratio=totalHC/g['graph_control_pair_ns'],interpretation='Device task exposure ratio only; not request gain/MFU or strict end-to-end bound.'))
    ec=Counter(e['kind'] for e in edges);assert ec=={'ATTENTION_PRE_TO_NORM':3200,'FFN_PRE_TO_NORM':3200,'INTRALAYER_POST_TO_PRE':3200,'FFN_MERGE_TO_POST':3200,'CROSSLAYER_POST_TO_PRE':3120}
    sums={}
    for kind in ec:
      rr=[r for r in edges if r['kind']==kind];bound=[r for r in rr if r['root_bound']]
      sums[kind]=dict(all_edges=len(rr),root_bound=len(bound),phase_unbound=len(rr)-len(bound),same_stream=sum(r['same_stream'] for r in rr),
        gap_ns=stat([r['gap_ns'] for r in bound]),zero_intervening_compute_by_start=sum(not r['intervening_compute_by_start'] for r in rr),
        intervening_operator_executions=dict(sum((Counter(r['intervening_compute_by_start']) for r in rr),Counter())))
    write(O/'HC-operator-profile-and-edges-v58.json',dict(generated_utc=stamp,regimes=rows,regime_count=16,executions=dict(counts),
      graph_executions=80,graph_layers=3200,root_bound_graph_layers=2560,phase_unbound_graph_layers=640,
      exact_graph_role_edges=edges,edge_summary=sums,root_bound_stage_ns={k:stat(v) for k,v in times.items()},
      root_graph_HC_exposure=exposure,root_graph_HC_exposure_ratio=stat([r['ratio'] for r in exposure]),
      identity_scope='Native graph block and exact TASK records + pinned model dataflow. Shape/time adjacency does not establish tensor generation/alias/readiness or legality. Intervening-by-start count excludes earlier-started overlapping tasks and noncompute control; zero is not idle.',
      overall_complete=False))
    write(O/'HC-work-traffic-and-reference-bounds-v58.json',dict(generated_utc=stamp,models=models,
      formulas=['HcPre coefficient reference: rstd=1/sqrt(sum(X^2)/(hH)+norm_eps); logits=(Xflat @ fn.T)*rstd; separate scale/base per pre/post/comb group.',
        'pre=sigmoid(scale0*logits_pre+base_pre)+hc_eps; post=2*sigmoid(scale1*logits_post+base_post).',
        'When pre_mix exists, Y=BF16(sum_i FP32(X_i)*external_pre_mix_i). New pre is a separate live output for the next HC, not the coefficients used for current Y.',
        'comb starts from row-softmax(scale2*logits_comb+base_comb)+hc_eps; first column normalization then iterTimes-1 (row,column) pairs. With20 iterations:39 dependent normalization rounds after softmax.',
        'HcPost: output_j=BF16(FP32(x)*post_j + sequential_i[FP32(residual_i)*comb_i,j]); native source uses separate Mul then Add in i order.',
        'Qpre_min=4KC+2MK+2MH+4M(C+h)+4C+12. Qpost_min=2MH(1+2h)+4M(h+h^2).',
        'Current source additionally writes/reads4MK FP32 cast values per direction, re-reads2MK BF16 for collapse, materializes split-K projection and16-value Gram rows. Logical GM payloads are not HBM.',
        'Tref>=max(F_HF32/P_HF32,F_A2/P_A2,F_Vector/P_Vector,Q_HBM/B_HBM,Q_L2/B_L2,normalization/reduction dependencies), with actual typed capacity and necessary traffic still explicitly unbound.'],
      published_reference_HBM_bytes_per_s=B,reference_scope='Previously verified published per-die reference, not current physical resource attestation. No FP32/HF32 capacity inferred from376BF16/752INT8 peak.',
      overall_complete=False))
    # This source shape issue constrains row regrouping, without diagnosing the loaded run.
    split=src['hc_pre_m_k_split_core.h'][1];base=src['hc_pre_base.h'][1];cube=src['hc_pre_cube_compute.h'][1];host=src['hc_pre_tiling.cpp'][1]
    assert 'curRowFactor * tilingData->hcMult);' in split and 'AscendC::SetHF32Mode(1)' in cube
    assert 'rowFactor_ = rowFactor_ + 1' in host and 'ReduceSumARAPerf(yCastLocal' in base
    ranges={'torch_binding.cpp':[(1439,1452),(1541,1577)],'hc_pre.cpp':[(100,123)],
      'hc_pre_m_k_split_core.h':[(115,161),(362,376),(396,440),(476,514)],
      'hc_pre_base.h':[(428,441),(504,533),(550,568)],'hc_pre_cube_compute.h':[(418,437),(457,476)],
      'hc_post_d_split.h':[(299,356)],'hc_pre_tiling.cpp':[(150,164),(221,257),(282,308)]}
    parts=[]
    for name,rs in ranges.items():
      r,t=src[name];lines=t.splitlines()
      for lo,hi in rs:parts.append(dict(path=r['local_path'],sha256=r['sha256'],first_line=lo,last_line=hi,source=lines[lo-1:hi]))
    alignment=[]
    for row in range(3):
      valid=list(range(row*8,row*8+4));alignment.append(dict(row=row,valid_indices=valid,epsilon_updated_indices=[x for x in valid if x<3*4]))
    assert [len(r['epsilon_updated_indices']) for r in alignment]==[4,4,0]
    write(O/'HC-operator-numerical-and-dependency-contract-v58.json',dict(generated_utc=stamp,parts=parts,
      model_config=dict(path=configrow['local_path'],sha256=configrow['sha256'],hc_mult=4,hc_sinkhorn_iters=20,hc_eps=1e-6,norm_eps=1e-20),
      facts=['All16profile regimes include external pre_mix. Current collapsedY does not depend on currentprojection/coefficients, but newpre/post/comb remain live.',
        'Projection AB explicitly enables HF32 onFLOATinputs; A2 computes16x16 Gram blocks and stage2 selects diagonal. Native exact reduction/precision differs from genericFP32dot assumptions.',
        'Preserve BF16 outputY before RMSNorm; RmsNormCast outputsBF16 andFLOAT widened fromroundedBF16 per priorF04 contract. Preserve downstreamrouted/shared consumers.',
        'HcPost creates fresh output and widensBF16x/residual toFP32; separate Mul/Add in increasingresidual-i order followedRINTtoBF16. Do not replace with unrounded producer or reassociated/FMAexpression without fidelity contract.',
        'Post output is nextPre input and nextPost residual, and may feed aux/Engram/tail. Post→Pre fusion does not eliminate all residual materialization.',
        'Original source usesglobal SyncAll between stage1 andstage2 and whole-kernelcompletion before downstreamnorm. EarlyY visibility requires explicit operator split/handoff, not a launch-order-only change.'],
      epsilon_alignment_condition=dict(source_issue='Column reduction buffer stride8 forhc4, but twoepsilonAdds lengths usecurRowFactor*4. Withmultiple rows, valid column sums can be missed; row reduction hasdifferent packed layout and mustnot blindlyreceive same change.',
        independent_index_witness=alignment,source_has_matching_condition=True,source_host_may_increase_rowFactor=True,
        captured_active_defect_proved=False,loaded_binary_identity_proved=False,
        implication='Repartitioning rows/fusing must explicitly retain or define numericalcontract; inspectactualvariant/rowFactor if it changes decision. No untestedbugfix/activefault claim.',
        primary_context='https://github.com/vllm-project/vllm-ascend/issues/19357'),
      conditional_analysis_deliverable=True,implementation=False,native_test=False,whole_module_closed=False,overall_complete=False))
    variants=[
      dict(id='F02-YCOEFF-SPLIT',decision='CONDITIONAL_PRIORITY',proposal='Split delayedexternal-pre_mix collapse+RMSNorm from currentHCcoefficient production; scheduleY branch without waiting forcurrentprojection/Sinkhorn.',
        conditions=['Externalpre_mix present (allcaptured16regimes); retainnewpre/post/comb andtheirdeadlines atHCpost/nextpre.',
          'PreserveYBF16round andnorm numerics. Price2launches/sync pluscompetitionforVector/HBM/Cube-caststage;coefficientdeadline andnewvisibility/alias contract required.'],
        traffic_delta='Splittingalone neednotremovebytes; collapsingwithnorm conditionally removes4MH BF16interface. Mainopportunity isdependencypathreordering,not deletingprojectionwork.'),
      dict(id='F02-PRENORM',decision='CONDITIONAL',proposal='Fuse currentHcPre Y collapse epilogue withRMSNorm/RmsNormCast.',
        conditions=['KeepBF16Yround beforeRMS andwholeH reduction.', 'Keepnewpre/post/comb;forFFNpreserveBF16normoutput plusFLOATwidenedroundedplane. Anyrstd/otherreaders mustremain.'],
        traffic_delta='4MH Y write/read ifnootherYconsumer; coeff/HF32/Sinkhornworknotremoved. Fullkernel maystilldelayY behindcoefficientpath.'),
      dict(id='F03-POST-COLLAPSE-NORM',decision='CONDITIONAL',proposal='HcPost producesroundedresidual tiles fornext delayedcollapse+norm onchip.',
        conditions=['KeepPostseparateMul/Addorder andBF16round beforecollapse.', 'RetainresidualwritefornextHcPost/aux/Engram/tail andcurrentcoefficientproducer;cannot erase interveningconsumers.',
          'NeedwholeHnorm reduction,matchingrowmap/pre_mix andrealreadiness;graphadjacencyaloneisinsufficient.'],
        traffic_delta='Can saveone2MhH residualread bycollapse plus4MH Yinterface ifallconditionshold. Cannotcountfull4MhH producerwrite/readelimination whilelaterresidualconsumer remains.'),
      dict(id='F02-CAST-WORKSPACE',decision='CONDITIONAL',proposal='AvoidexternalFP32castwrite/read usingcompatibleonchipVector→Cube handoff, orretilewithinHC.',
        conditions=['Nativepath explicitlyHF32AB andFLOATweights;dtypechangesmustpreservesemantics. NounqualifiedBF16weightconversion.', 'Priceonchipcapacity,barriers,layout,split-K andCube/Vectorparallelism.'],traffic_delta='Currentcastinterface8MK logicalbytes;M3=491520B,notprovenHBMtraffic.'),
      dict(id='F02-GRAM-DIAGONAL',decision='CONDITIONAL_NUMERICAL_VARIANT',proposal='Replace full16x16 A2Gramblock work withnecessaryrow-sumsquares.',
        conditions=['Preserve orexplicitlydefineFP32/Cube reductionerror contract;Vectorreductionnotunconditionallybit-equivalent.', 'PriceVectoroccupancy andchangedprojection/Sinkhornreadiness;do notuseBF16peakforHF32/A2.'],traffic_delta='Semantic2MK sumsquareworkversussource2*Mpadded*16*K requestedGramwork;fulltilepadding andoffdiagonalworkaredistinctfromprojection.'),
      dict(id='F23-MERGE-POST',decision='CONDITIONAL_RETAINED',proposal='Inline singletonshared+routed BF16merge intoHcPost.',
        conditions=['PreserveDownBF16round thenmergeBF16round beforeFP32HCPost arithmetic;keeproutedready andallreaders.','Alternative/sharedinterfacewithv57ONEADD/DownAdd;do notcountsameAddremovaltwice.'],traffic_delta='4MH mergewrite+HCreadconditional;doesnotremoveHCresidualread/output.'),
      dict(id='F03-ALGEBRAIC-PREFOLD',decision='REJECT_UNCONDITIONAL',proposal='Cancel/foldpost→pre intooneunroundedmatrix or eliminateallresidualstorage.',
        conditions=['FailsunconditionalBF16/reduction-ordercontract;currentHCnewcoefficients dependonroundedresidual andnonlinearRMS/sigmoid/Sinkhorn.', 'Newtensor/alias/generationconstraintsandaux/Engram/tailconsumersremain.'],traffic_delta=None)]
    write(O/'HC-fusion-split-reorganization-variants-v58.json',dict(generated_utc=stamp,variants=variants,
      interactions=['YCOEFF-SPLIT andPRENORM cancompose;their4MH interface saving iscountedonce.',
        'POST-COLLAPSE-NORM andwholeHCfusion competeonresourceownership;residualwritecannotbeclaimedgoneifnextHCPostreadsit.',
        'CAST-WORKSPACE andGRAM-DIAGONAL changeinternalresourcebalance;bytedeletionplusCubeopsavingsnotindependentruntimegains.',
        'MERGE-POST,ONEADD andDOWNADD refer tosameAddstage/removableinterfaces;preservebothBF16rounds andavoiddoublecounting.'],
      no_new_main_candidate=True,conditional_operator_analysis_deliverable=True,measured_speedup=None,implementation=False,overall_complete=False))
    print(json.dumps(dict(regimes=16,executions=dict(counts),M_domains=sorted(Ms),edges=dict(ec),edge_summary=sums,root_stage_ns={k:stat(v) for k,v in times.items()},
      root_HC_exposure_ratio=stat([r['ratio'] for r in exposure]),models=models[:2],variants=len(variants),overall_complete=False)))

if __name__=='__main__':main()
