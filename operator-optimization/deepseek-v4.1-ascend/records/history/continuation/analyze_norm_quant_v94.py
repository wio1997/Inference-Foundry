"""Offline norm/quant legality, shared interfaces and typed resource subwork cuts."""
from collections import Counter,defaultdict
from fractions import Fraction as F
from pathlib import Path
import hashlib,json,math,statistics,struct
from finalize_origin_coverage_v9 import N,O,R,read,write
from mirror_saved_profile_core_v53 import CACHE
from analyze_state_engram_v62 import dimensions

PRIMARY=N/'norm-quant-primary-v94'

def f32(x):return struct.unpack('>f',struct.pack('>f',x))[0]

def bf16(x):
    u=struct.unpack('>I',struct.pack('>f',x))[0]
    u=(u+0x7fff+((u>>16)&1))&0xffff0000
    return struct.unpack('>f',struct.pack('>I',u))[0]

def quant(row):
    a=max(map(abs,row));scale=f32(a/127)
    assert a>0 and math.isfinite(a)
    return [round(f32(x/scale))for x in row],scale

def budget(M,K,consumers,variant,keep_qr):
    # This is a fixed GM-interface value model, not actual HBM or an all-algorithm bound.
    if variant=='SEPARATE':
        return dict(qr_write=2*M*K,qr_reads=2*consumers*M*K,q_writes=consumers*M*K,
          scale_writes=4*consumers*M,q_reads=consumers*M*K,scale_reads=4*consumers*M)
    norm_fused=variant.startswith('NORM_FUSED')
    joint=variant.endswith('JOINT_GEMM')
    return dict(qr_write=2*M*K if not norm_fused or keep_qr else 0,
      qr_reads=0 if norm_fused else 2*M*K,q_writes=M*K,scale_writes=4*M,
      q_reads=M*K if joint else consumers*M*K,scale_reads=4*M if joint else 4*consumers*M)

def stat(xs):
    return dict(n=len(xs),minimum=min(xs),median=statistics.median(xs),maximum=max(xs))if xs else None

def main():
    assert not(O/'REVIEW-CHECKPOINT-v94.json').exists()
    # Preserve exact primary-source provenance. These are public operator implementations, not installed objects.
    idx=read(PRIMARY/'operator-source-index.json');doc=read(PRIMARY/'operator-doc-receipt.json')
    for row in idx['files']+[doc]:
        p=Path(row['local_path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
    parts=[]
    ranges={
      'add_rms_norm_dynamic_quant.cpp':[(18,43)],
      'add_rms_norm_dynamic_quant_normal_kernel.h':[(164,215),(218,255),(318,341)],
      'add_rms_norm_dynamic_quant_single_row_kernel.h':[(125,158),(187,222)],
      'add_rms_norm_dynamic_quant_helper.h':[(169,179)]}
    for name,spans in ranges.items():
        p=PRIMARY/name;lines=p.read_text().splitlines()
        parts.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
          parts=[dict(first_line=a,last_line=b,source=lines[a-1:b])for a,b in spans]))
    caller=read(O/'projection-operator-numerical-contract-v61.json')
    cp=next(p for p in caller['parts']if p['path'].endswith('dsa_v41.py'));path=Path(cp['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==cp['sha256']
    lines=path.read_text().splitlines();parts.append(dict(path=str(path),sha256=cp['sha256'],
      parts=[dict(first_line=a,last_line=b,source=lines[a-1:b])for a,b in [(350,382),(394,410),(470,495)]]))
    write(O/'norm-quant-operator-source-contract-v94.json',dict(primary_commit=idx['commit'],public_source_index=str(PRIMARY/'operator-source-index.json'),
      primary_document_receipt=str(PRIMARY/'operator-doc-receipt.json'),parts=parts,
      facts=[
        'CurrentV4.1source usesqr=attn.q_norm(QA),thenquantizesqrforQ_B,andpassesqrtoIndexerproject_querywhereapplicable. Returnedqr/allalternatecallersremainreader/lifetimegates.',
        'SavednativeRmsNormoutputsDT_BF16[3,1280]andFLOAT[3,1];DynamicQuantconsumesBF16[3,1280]andoutputsINT8plusFLOATrowscale.',
        'PinnedpublicAddRmsNormDynamicQuant keys1/2 calculateRMS/gamma inFLOAT buffers andquantizeFLOATvalues,withoutaBF16normalized-outputroundtrip. Add xOut isnotnormalizedqr.',
        'PublicAPIexistsforA3INT8withBF16gamma andFLOATrowscales,butAPIavailability/version/nullx2adapter/tilingandnativeprecisioncompatibilityareseparategates.',
        'Otherkeys/arch35 andinstalledCANN9.2nativeobjectsareNOTidentifiedbythispublicsource. NoAPIreplacementorwholemodulelegalityclaimed.'
      ],precision_decision='BITWISE_SUBSTITUTION_NOT_ESTABLISHED;DECLARED_MODEL_COUNTEREXAMPLE;TOLERANCE_NATIVE_BINDING_PENDING',
      no_framework_wide_research=True,new_target_source_files=0,loaded_native_binary_attested=False,overall_complete=False))

    old=read(O/'projection-graph-chains-v61.json');pairs={(p['database_index'],p['graph_ordinal'],p['block_ordinal']):p for p in old['QR_shared_quant_candidates']}
    dbs=defaultdict(list)
    for c in old['graph_projection_chains']:dbs[c['database_index']].append(c)
    fronts=[];tasks_used=set();typed=Counter();dur=defaultdict(list)
    for db,cs in sorted(dbs.items()):
        native=read(CACHE/f'analysis/native-execution-identity-v14/db-{db:02d}.json')
        tasks={t['task_record_rowid']:t for t in native['TASK_records']};defs={t['globalTaskId']:t for t in native['COMPUTE_definitions']}
        for c in cs:
            k=(db,c['graph_ordinal'],c['block_ordinal']);pair=pairs.get(k);s=c['stages'];bound=c['root_bound']
            out=dict(database_index=db,graph_ordinal=k[1],block_ordinal=k[2],root_bound=bound,index_query_branch=pair is not None,stages={})
            stages={name:s[name]for name in ['Q_A','Q_NORM','Q_QUANT','Q_B']}
            if pair:stages.update(INDEX_QUANT=pair['index_quant'],INDEX_Q=pair['index_Q_B'])
            for name,old_t in stages.items():
                t=tasks[old_t['task_record_rowid']];d=defs[t['globalTaskId']]
                tasks_used.add((db,t['task_record_rowid']))
                sig={key:d[key]for key in ['blockNum','mixBlockNum','opType_text','inputShapes_text','inputDataTypes_text','outputShapes_text','outputDataTypes_text']}
                typed[(name,json.dumps(sig,sort_keys=True))]+=1
                out['stages'][name]=dict(task_record_rowid=t['task_record_rowid'],signature=sig,startNs=t['startNs'],endNs=t['endNs'])
                if bound:dur[name].append(t['endNs']-t['startNs'])
            nd=out['stages']['Q_NORM']['signature'];qd=out['stages']['Q_QUANT']['signature']
            assert dimensions(nd['inputShapes_text'])==[[3,1280],[1280]]and nd['inputDataTypes_text']=='DT_BF16;DT_BF16'
            assert dimensions(nd['outputShapes_text'])==[[3,1280],[3,1]]and nd['outputDataTypes_text']=='DT_BF16;FLOAT'
            assert dimensions(qd['inputShapes_text'])==[[3,1280]]and qd['outputDataTypes_text']=='INT8;FLOAT'
            if pair:
                assert out['stages']['INDEX_QUANT']['signature']==qd
                assert pair['Q_quant']['task_record_rowid']==s['Q_QUANT']['task_record_rowid']
            out.update(native_pointer_value_identity_proved=False,new_fused_precision_or_resource_path_attested=False)
            fronts.append(out)
    regs=read(O/'operator-shape-ceiling-ledger-v55.json')['observed_regimes'];selected=[]
    for r in regs:
        if r['operator']not in ['RmsNorm','DynamicQuant']or not set(r['module_candidates'])&{'M07','M09'}:continue
        d=r['signature'];sh=dimensions(d['inputShapes_text'])
        if sh and len(sh[0])==2 and sh[0][1]==1280 and d['inputDataTypes_text'].startswith('DT_BF16'):
            selected.append(dict(regime_id=r['regime_id'],operator=r['operator'],M=sh[0][0],N=1280,executions=r['executions'],
              origin=r['origin'],context=r['context'],module_candidates=r['module_candidates'],signature=d))
    write(O/'norm-quant-native-frontiers-v94.json',dict(frontiers=fronts,frontier_count=len(fronts),root_bound=sum(r['root_bound']for r in fronts),
      index_query_frontiers=sum(r['index_query_branch']for r in fronts),root_index_query_frontiers=sum(r['root_bound']and r['index_query_branch']for r in fronts),
      reused_unique_native_TASKs=len(tasks_used),typed_cohorts=[dict(stage=k[0],signature=json.loads(k[1]),executions=n)for k,n in typed.items()],
      root_bound_stage_duration_ns={k:stat(v)for k,v in dur.items()},observed_regimes=selected,
      observed_regime_count=len(selected),observed_regime_executions=sum(r['executions']for r in selected),
      counts_are_saved_evidence_not_new_captures=True,times_are_not_fusion_gain_or_critical_path=True,new_regime_or_assignment_promotions=0,overall_complete=False))

    K=1280;x=[1.0,43/128];eps=1e-6
    sumsq=f32(sum(f32(v*v)for v in x));s=f32(1/math.sqrt(f32(sumsq/K)+eps))
    unrounded=[f32(v*s)for v in x];rounded=[bf16(v)for v in unrounded];rawq,raws=quant(x);precq,precs=quant(unrounded);outq,outs=quant(rounded)
    witness=dict(shape=[1,K],nonzero_prefix=x,trailing_zeros=K-len(x),gamma_all_one=True,epsilon=eps,rstd_model=s,
      FP32_normalized_prefix=unrounded,BF16_normalized_prefix=rounded,raw_codes=rawq,FP32_direct_codes=precq,BF16_then_quant_codes=outq,
      raw_scale=raws,FP32_direct_scale=precs,BF16_then_quant_scale=outs,
      generic_legal_input_not_observed_tensor_values=True,native_device_or_library_test=False,
      arithmetic_domain='FinitepositiveBF16inputs,gamma1,F32scale/normmultiply,modeledpositiveRNEquant;RMSreduction/rsqrtnativeordernotattested.')
    assert rawq==precq==[127,43]and outq==[127,42]
    invariant_cases=0
    for a in range(1,24):
        for b in range(-a,a+1):
            for c in [F(1,7),F(1,2),F(2),F(17,3)]:
                z=[F(a),F(b)];norm=[c*v for v in z]
                assert [round(127*v/max(map(abs,z)))for v in z]==[round(127*v/max(map(abs,norm)))for v in norm]
                invariant_cases+=1
    # An irrevocable final INT8 code cannot be fixed before unseen same-row coordinates under this quantizer.
    prefix=[1.,.25];rowa=prefix+[.5];rowb=prefix+[2.]
    qa,_=quant(rowa);qb,_=quant(rowb)
    assert qa[0]!=qb[0]
    model=dict(exact_real_identity='c>0,a=max|z|>0:Q(cz)=Q(z), scale(cz)=c*a/127;this doesnotremove c whenreturnedscale/dequantobservesc.',
      exact_real_rational_cases=invariant_cases,BF16_rounding_witness=witness,
      row_frontier_witness=dict(same_seen_prefix=prefix,unseen_suffixes=[.5,2.],final_codes_at_first_coordinate=[qa[0],qb[0]],
        conclusion='Withinirrevocableone-passquantizerfullrowamaxmustknownbeforefinalcode;rowscanpipelineindependently. Provisionalcodes/corrections/recomputearedifferentdomains,notruledout.'),
      fusion_rule='KeepQAoutputBF16,RMSarithmetic/gamma/epsandBF16qrroundinginUBbeforeamax/INT8;fusiondoesnotrequireGMmaterializationbutdoesnotlicensecancelingrounds.',
      unknown=['ActualnativeRmsNorm/DynamicQuantV2arithmetic/zero/NaN/Inf/rounding/scaleclampandallowablemodeltolerance.',
        'PublicfusedAPIx2nulladapter,gamma/layout/shape/tilingkeys/installedversionandnorm-outputreaders.',
        'Allqrreaders,lifetimes,rowparametergenerationsanddifferentquantizerconfigmustmatchbeforeproducerreuse.'],overall_complete=False)
    write(O/'norm-quant-precision-and-row-models-v94.json',model)

    models=[]
    for M in sorted({3}|{r['M']for r in selected}):
        for count in [1,2]:
            base=budget(M,K,count,'SEPARATE',True);base_sum=sum(base.values())
            for keep in [False,True]:
                for variant in ['SHARED_QUANT','NORM_FUSED_SHARED_QUANT','NORM_FUSED_JOINT_GEMM']:
                    if variant=='SHARED_QUANT'and not keep:continue
                    if variant=='NORM_FUSED_JOINT_GEMM'and count==1:continue
                    b=budget(M,K,count,variant,keep)
                    models.append(dict(M=M,K=K,quantized_consumers=count,variant=variant,normalized_qr_store_retained=keep if variant.startswith('NORM_FUSED')else True,
                      baseline_components=base,transformed_components=b,baseline_value_bytes=base_sum,transformed_value_bytes=sum(b.values()),
                      conditional_removed_GM_interface_value_bytes=base_sum-sum(b.values()),
                      INT8_VC_GM_bridge_value_bytes=(b['q_writes']+b['q_reads']),
                      HBM_saving_or_speedup=None))
    W=K*(32768+4096);R0=24*65536
    cuts=[]
    for qa_included in [False,True]:
        weights=W+(13107200 if qa_included else 0)
        cuts.append(dict(planes=['Q_B','INDEX_Q']+(['Q_A']if qa_included else[]),weight_bytes=weights,
          deadline='AllrequiredcurrentM3Q_BandIndexQoutputsready,notearlierQ_B-onlydeadline',
          one_initial_L0B_grant_bytes=R0,L1_fixedB_min_cycles=math.ceil(F(weights-R0,24*128)),
          flexible_orientation_min_cycles=math.ceil(F(weights-2*R0,24*384)),
          INT8_useful_compute_min_cycles=2*3*K*(32768+4096)/(24*16384),
          QA_BF16_useful_min_cycles=200 if qa_included else None,mixed_mode_rates_not_added=True,
          domain='Conventionaldensefulloutputs/distinctuncompressedweights/L1required/24AIC/oneinitialwarmgrant/noprecomputedpartials. Windowcoversallchargedservice,notfixednativeTASKstartorwhole-tensorbarrier.',
          actual_native_path_or_physical_capacity_attested=False))
    write(O/'norm-quant-interface-and-resource-models-v94.json',dict(models=models,interface_case_count=len(models),shared_weight_cuts=cuts,
      bridge_domain='Vector-owneduncompressedINT8qproductionandAICconsumptionthroughGM;GMneednotreachHBM. Jointinput/rowscaleone-readisconditionalonlayout/lifetime/tiling/dequantschedule.',
      excluded=['RMSinput/gamma/rstdoutputandworkspace','GEMMweights/channelscales/output/dequantbuffers','Otherqrreadersnotprovenabsent','Actualcache/HBM/launch/collectiveandnativeprecisioncost'],
      joint_budgets_not_added_to_standalone_savings=True,actual_HBM_or_gain_proved=False,overall_complete=False))
    write(O/'norm-quant-variants-v94.json',dict(variants=[
      dict(id='NQ94-PRESERVE-ROUND',decision='CONDITIONAL',candidate=['F05'],contract='FuseRMS→DQwithBF16qrroundinUB;fullrowstatisticsandallreaders/optionalrstdoutputsretained. Noarithmeticcancellationassumed.'),
      dict(id='NQ94-CANCEL-ROW-SCALE',decision='EXACT_REAL_IDENTITY_ONLY;NOT_UNCONDITIONAL_CURRENT_PRECISION',candidate=['F05','F09'],contract='Onlyunderdeclaredunroundedhomogeneousquantdomain;keepcintheobservable rowscale/dequant. BF16roundcounterexampleandscaleobservationblockblindremoval.'),
      dict(id='NQ94-PUBLIC-API',decision='PRECISION_AND_VERSION_GATED',candidate=['F05'],contract='DocumentedA3fusedAPIkeys1/2useFP32normalizedvalues;notbitwiseRmsNormBF16→DQinthestatedwitness. Allowabletolerance/loadedAPI/nullx2/shape/readercontractnotclosed.'),
      dict(id='NQ94-SHARED-QR-FORK',decision='CONDITIONAL_REFINEMENT_OF_V61',candidate=['F05','F06','F09'],contract='Onequantizedqr+scale forQB/IndexQ;chooseonejointinterfacegraph. RetainA3INT8GMbridge,allrowparameters/generations/scalesandseparateoutputs;publictwo-outputsmoothbranchisnottworequiredidenticalcopies.')],
      new_maincandidate=False,implementation=False,native_test=False,overall_complete=False))
    print(json.dumps(dict(native_frontiers=len(fronts),index_query_frontiers=len(pairs),native_tasks_reused=len(tasks_used),regimes=len(selected),
      executions=sum(r['executions']for r in selected),exact_real_cases=invariant_cases,interface_cases=len(models),resource_cuts=len(cuts),overall_complete=False)))

if __name__=='__main__':main()
