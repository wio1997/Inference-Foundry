from copy import deepcopy
from datetime import datetime,timezone,timedelta
from fractions import Fraction
import hashlib,json
from finalize_origin_coverage_v9 import N,O,R,W,T,L,read,write,ctl


def load_receipt(label,embedded=False):
    r=read(N/(label+'.json'));assert r['returncode']==0,label
    x=json.loads(r['stdout'])
    if embedded:
        b=(json.dumps(x['result'],ensure_ascii=False,separators=(',',':'))+'\n').encode()
        assert hashlib.sha256(b).hexdigest()==x['result_sha256'],label
    return x


def graph_check(g):
    ns=[x['id']for x in g['nodes']];assert len(ns)==len(set(ns))
    assert all(e['a']in ns and e['b']in ns for e in g['edges'])
    remaining=set(ns)
    while remaining:
        ready={n for n in remaining if not any(e['b']==n and e['a']in remaining for e in g['edges'])}
        assert ready,'cycle in analysis graph'
        remaining-=ready


def main():
    assert not (O/'REVIEW-CHECKPOINT-v27.json').exists(),'refuse duplicate finalization'
    now=datetime.now(timezone.utc);stamp=now.isoformat()
    p=load_receipt('F33-profile166-v27')
    a=load_receipt('F33-audit166-v27')
    s=load_receipt('F33-source166-v27',True)
    i=load_receipt('F33-index-consumer166-v27',True)
    ps=p['summary'];au=a['summary']
    assert p['result_sha256']==au['profile_source']['sha256']
    assert ps['graph_pairs']==au['graphs_reviewed']==80
    assert ps['projection_overlap_ns']['max']==0
    assert ps['independent_512_norm_candidate_counts']=={'2':80}
    assert all(v['positive']==80 for v in au['observed_overlap_ns'].values())
    assert au['native_ambiguous_roles']==[]
    assert sum(v['TASKs']for v in au['native_role_counts'])==2382
    refs=[dict(label=label,receipt=str(N/(label+'.json')),full_result=obj['full_result'],sha256=obj['result_sha256'])
          for label,obj in [('F33-profile166-v27',p),('F33-source166-v27',s),('F33-audit166-v27',a),('F33-index-consumer166-v27',i)]]
    write(O/'F33-profile-order-contract-v27.json',dict(generated_utc=stamp,evidence_grade='MEASURED_WITH_DERIVED_SOURCE_STREAM_ROLES',
        profile_summary=ps,order_audit=au,evidence_refs=refs,whole_consumer_closure=False,overall_complete=False))
    files=s['result']['files'];index=i['result'];functions=[fn for f in files for fn in f['functions']]
    idx=index['functions'][0];assert idx['name']=='update_keys'and idx['start']==106 and idx['end']==125
    assert any(fn['name']=='_write_compressed_source'and fn['end']==565 for fn in functions)
    assert any(fn['name']=='unquantized_gemm'and fn['start']==49 and fn['end']==54 for fn in functions)
    norm=next(fn for fn in functions if fn['name']=='forward_oot'and fn['start']==83)
    assert any('npu_rms_norm(x, self.weight, self.variance_epsilon)'in line for line in norm['source'])
    consumer=dict(generated_utc=stamp,evidence_grades=['MEASURED','DOCUMENTED','DERIVED','UNKNOWN'],
        source_files=[dict(path=f['path'],sha256=f['sha256'])for f in files]+[dict(path=index['path'],sha256=index['sha256'])],
        reviewed_source_receipts=[refs[1],refs[3]],reuse='profile-raised-source-contract-v16.json;C1-and-padding-adjustment-contracts-v16.json',
        reviewed_facts=[
            'Native SWA projection frames reach cv_linear.matmul:172 -> unquantized_gemm:54 -> torch.nn.functional.linear. Quantized/communicating branches in matmul remain distinct.',
            'Native C1 calls are owned by compressor.forward:71 -> self.norm(self.wkv(x));v16 constructor has a distinct bias-free BF165120->512 nn.Linear and a distinct512-wide RMSNorm.',
            'Actual norm frame layernorm.py:99 calls torch_npu.npu_rms_norm(x,self.weight,self.variance_epsilon),then optional bias if loaded. Profile inputs are BF16;preserve both independent parameter/epsilon/bias contracts and GEMM->BF16 boundary.',
            'DSA_write_compressed_source:544 calls indexer.update_keys(latent,...),then:550 views the same latent and:551 performs in-place long-KV RoPE,then:553-565 publishes to packed backend or scatter.',
            'Indexer.update_keys:108 returns when !owns_k or empty;:110 computes self.wk(latent) then independent128-wide k_norm;:111 rotates the newly projected key,not the512-wide latent.',
            'Indexer.update_keys packed branch writes key/scale and optionally folded index cache;fallback quantizes INT8 key then separately publishes key and FP16 scale.',
            'SWA cache writer has packed backend kind=win or scatter;long-KV writer has kind=cmp or scatter. Same typed cache shape is not proof same physical address.',
            'For each overlapping latent value,the index-key projection old-value read must complete before in-place long-RoPE overwrites it,unless a candidate preserves an independent original-value copy. Rows/tiles may pipeline;this does not require a universal whole-operator barrier. Index publication->long-RoPE is current control order,not proved a universal semantic dependency.',
            'Requested _write_index_cache/_write_long_cache names are absent from DSA;actual publication is inline and in indexer.update_keys. Their absence was not treated as absence of cache writes.'],
        local_questions_answered=['Actual primitive入口','Two independent512norm boundaries','Unrotated latent index reader versus in-place long-RoPE writer','SWA/index/scale/long packed/fallback publication branches'],
        still_unknown=['Historical tensor pointers/strides/storage aliases and actual capture-address/event generations.',
            'Exact backend numerical equivalence for a new tiling or elimination of the materialized BF16 boundary.',
            'Actual post-load cross-rank parameter replication if projections move across row allgather;not needed as equality between the two distinct matrices for local joint projection.',
            'Every future cache/rollback/folded-cache observer and ownership generation beyond the scoped direct source subtree.',
            'End-to-end causal performance of any unimplemented candidate.'],
        stop_rule='Do not reread entire DSA/indexer/SDK. These direct consumer and primitive questions are now answered. Further source reads require a particular conclusion-changing boundary.',
        target_implementation_changed=False,whole_module_closed=False,overall_complete=False)
    write(O/'F33-consumer-and-numerical-contract-v27.json',consumer)
    # Exact arithmetic illustrates semantic constraints; it is not native correctness/performance testing.
    payloads=[]
    for m in [24,6400,6528,6668]:
        x=m*5120*2;weights=2*512*5120*2
        baseline=2*x+weights;joint=x+weights
        payloads.append(dict(rows=m,one_input_bytes=x,two_distinct_weight_bytes=weights,
            independent_single_read_logical_operand_bytes=baseline,joint_single_read_logical_operand_bytes=joint,
            avoided_duplicate_logical_input_bytes=x,
            avoided_fraction_of_that_operand_count=str(Fraction(x,baseline))))
    eps=Fraction(1,1000000)
    independent_squared=[Fraction(1)/(1+eps),Fraction(4)/(4+eps)]
    combined_squared=[Fraction(1)/(Fraction(5,2)+eps),Fraction(4)/(Fraction(5,2)+eps)]
    assert independent_squared[0]!=combined_squared[0]and independent_squared[1]!=combined_squared[1]
    math=dict(generated_utc=stamp,evidence_grade='DERIVED_IN_DECLARED_ARITHMETIC_DOMAIN',
        exact_projection_identity='For fixed matrices A,B: concat_features(X*A^T,X*B^T)=X*concat_rows(A,B)^T in exact arithmetic. A andB need not be equal;both parameter sets must be preserved.',
        required_norm_identity='For U=X*A^T,V=X*B^T,use concat(RMS512(U;gammaA,epsA),RMS512(V;gammaB,epsB));do not substitute RMS1024(concat(U,V)).',
        joint_norm_counterexample=dict(each_group_width=512,groupA_all_values=1,groupB_all_values=2,all_gamma=1,
            illustrative_eps=str(eps),independent_output_squared=[str(x)for x in independent_squared],joint_output_squared=[str(x)for x in combined_squared],
            scope='Counterexample to universal jointnormequivalence,not a claim this is the deployed model epsilon/input or a native BF16 parity check.'),
        output_layout=dict(packed_shape=[24,1024],packed_contiguous_strides=[1024,1],each_512_slice_strides=[1024,1],
            contiguous_24x512_strides=[512,1],conclusion='Each feature half is row-strided. Use proven strided consumers,separate output buffers orexplicit conversion;no claim a copy is universally mandatory.'),
        row_allgather_identity='For same per-rank A,B and identical row order, row_allgather(rowwise_projection(Xr))=rowwise_projection(row_allgather(Xr)) in exact arithmetic;rowwise feature norms similarly commute. Loaded rankweights,precision,padding/position/cacheownership remain conditions.',
        materialization_boundary='Native recorded GEMMs output BF16 and norms consume BF16. A fused path may retain an explicit BF16 rounding step without HBM materialization;direct FP32-accumulator norm is not automatically the same function.',
        logical_operand_examples=payloads,
        operand_count_scope='Two dense BF16 matrices,each operand counted once per independent/joint projection;illustrative logical counting only,not measured HBM,universal I/O lower bound or speedup.',
        native_validation=False,hardware_peak_or_physical_gain=None,overall_complete=False)
    write(O/'F33-projection-algebra-and-layout-v27.json',math)
    # Baseline source graph and conditional joint graph use distinct control edges.
    node_names=['hidden','swa_project','swa_norm','swa_rope','swa_publish','c1_project','c1_norm',
                'index_read','index_key_scale_publish','long_rope','long_publish','future_index','future_attention']
    edge_specs=[('hidden','swa_project','RAW'),('hidden','c1_project','RAW'),('swa_project','swa_norm','BF16_RAW'),
        ('swa_norm','swa_rope','RAW'),('swa_rope','swa_publish','RAW'),('swa_publish','c1_project','CURRENT_JOIN_CONTROL'),
        ('c1_project','c1_norm','BF16_RAW'),('c1_norm','index_read','RAW_PRE_LONG_ROPE_LATENT'),('c1_norm','long_rope','RAW'),
        ('index_read','index_key_scale_publish','RAW'),('index_read','long_rope','WAR_COMPLETE_OLD_LATENT_READ'),
        ('index_key_scale_publish','long_rope','CURRENT_CONTROL_RETAINED_FOR_F33_ALONE'),('long_rope','long_publish','RAW'),
        ('index_key_scale_publish','future_index','PERSISTENT_GENERATION'),('swa_publish','future_attention','PERSISTENT_GENERATION'),('long_publish','future_attention','PERSISTENT_GENERATION')]
    baseline=dict(nodes=[dict(id=n)for n in node_names],edges=[dict(a=a,b=b,kind=k)for a,b,k in edge_specs])
    graph_check(baseline)
    joint=dict(nodes=[dict(id=n)for n in node_names if n not in ['swa_project','c1_project']]+[dict(id='joint_project')],edges=[])
    for e in baseline['edges']:
        if e['kind']=='CURRENT_JOIN_CONTROL':continue
        a0='joint_project'if e['a']in ['swa_project','c1_project']else e['a']
        b0='joint_project'if e['b']in ['swa_project','c1_project']else e['b']
        v=dict(e,a=a0,b=b0)
        if v not in joint['edges']:joint['edges'].append(v)
    graph_check(joint)
    variants=[
        dict(id='F33-A',status='CONDITIONAL',change='Shared hidden input tiles with two distinct projections,independent output buffers and512norms.',
            preserves='Both matrices/biases,normparameters and BF16 boundaries,index old-latent read,SWA/index/scale/long publications.',
            open='Tiling/numerics,livebuffer/capture generations and comparisonagainst the fullcurrent multistream schedule.'),
        dict(id='F33-B',status='CONDITIONAL',change='Packed1024-output projection with two512norm groups.',
            open='Rowstride1024 rather than512;consumer strided support/separate outputs/conversion and exact precision. No1024-wide combinednorm.'),
        dict(id='F33-C',status='CONDITIONAL_COMPOSITION_WITH_F32',change='Project local rows then redistribute two512-wide role outputs rather than original5120 hidden features.',
            open='Loaded cross-rank weight/precision equivalence,rowrouting,padding,all globalhidden consumers,slot/position andpersistent/captureownership.',
            logical_feature_ratio=5,physical_network_or_latency_gain=None),
        dict(id='F33-D',status='EXCLUDED_AS_UNCONDITIONAL_SUBSTITUTION',change='Replace both512norms by one1024norm.',reason='Different reduction domain;exactcounterexample recorded.'),
        dict(id='F33-E',status='EXCLUDED_WITHOUT_VALUE_PRESERVATION',change='Move in-place long-RoPE beforeindexlatentread.',reason='Changes indexprojectioninput unless oldvaluecopied/proveninvariant. KeepWAR,notuniversalwhole-index-publication barrier.'),
        dict(id='F33-F',status='UNSUPPORTED_PARAMETER_DEDUPLICATION',change='Compute one512projection anduseitforbothroles.',reason='Sharedinput/weightshape doesnotprove equalmatrices. Localjointprojection doesnotrequire them equal.')]
    graph_contract=dict(generated_utc=stamp,candidate_id='F33',source_graph=baseline,conditional_joint_graph=joint,
        scope='Partial source-refined projection/consumer graph,not a pointer-complete or event-complete wholeblock DAG. WAR applies to overlapping oldvalue locations and permits row/tile pipelining.',
        changed_control_edge='SWA_publish->C1_project is current schedule control,removed only in a proposed joint projection afterfull readiness/alias/event review;keeping it withjoint node would invent a cycle.',
        role_scope='Source-layer20 C1/SWA. Graphrole labels derived from source and uniqueauxstream,not tensorpointer identities.',
        variants=variants,native_implementation=False,overall_complete=False)
    write(O/'F33-joint-projection-variants-v27.json',graph_contract)
    conditions=[
        ('graph_roles','DERIVED_REVIEWED','All80block20pairs retained,uniqueSWAauxstream and C1mainstream consistentwith source;no pointer claim.'),
        ('current_pair_overlap','MEASURED_LOCAL_QUESTION_CLOSED','NoC1/SWAprojectiontimeintersection in80pairs;gap89.742–98.362us includesotherwork,not idle gain.'),
        ('other_overlap','MEASURED_RETAINED','All80showSWAprojection/Qnorm,SWA512norm/Qb,C1projection/indexqueryquant overlap.'),
        ('norms','MEASURED_DOCUMENTED','Two512norms inall80;independentparameters/epsilon/optionalbias mustremain.'),
        ('primitive','MEASURED_DOCUMENTED','NativeSWAunquantizedF.linear and npu_rms_norm frame99 bound;otherquant/communication branches distinct.'),
        ('BF16_boundary','MEASURED_DOCUMENTED_CONDITIONAL','Both recordedoutputs/norminputs BF16;fusionmustpreserve allowedrounding contract.'),
        ('index_old_latent','DOCUMENTED_DERIVED_WAR','Per overlapping value,indexer.wk oldread precedesinplace long-RoPE overwrite orkeepcopy;row/tilepipeliningallowed,no universalwholekernelbarrier.'),
        ('index_publication_order','CURRENT_CONTROL_NOT_UNIVERSAL','Wholeindexpublish precedeslongRoPE currently;F33alone retainsit,relaxation belongsseparateF19-style scheduling proof.'),
        ('cache_backends','DOCUMENTED_RETAINED','SWAwin/longcmp packedwrites and fallbackscatters,indexpacked/folded orINT8key/FP16scale separatewrites.'),
        ('empty_or_not_owns_index','DOCUMENTED_RETAINED','update_keys earlyreturnfor!owns_k or empty;do notdropglobalcacheupdatessolelybecausequeryempty.'),
        ('output_stride','DERIVED_CONDITIONAL','Packedfeaturehalvesstride1024;currentshape24x512doesnotprove consumerstrides.'),
        ('distinct_parameters','DERIVED_SCOPE_CORRECTION','LocaljointGEMM keepsAandB;doesnotrequireA=B.Parameterdeduphasno equalityevidence.'),
        ('cross_rank_parameters','UNKNOWN_FOR_F32_COMPOSITION','Projectionbeforegatherneeds loadedrankparameter/routing/roundingequivalence,not justequalshapes.'),
        ('native_contexts','MEASURED_REVIEWED','Native2382=2x(960targetSWA+24targetC1+207draftSWA).Noalllayer20 ornearesttimepairing.'),
        ('prefill_rows','MEASURED_PARTIAL','C1native24 includes8eachM6400/6528/6668. Noexact nativeSWA layer20 pairing invented.'),
        ('C2','EXCLUDED_FROM_NAIVE_BF16_VARIANT','C2source2/8/14usesFP32compressor;separateF17/F32domain.'),
        ('physical_alias_ownership','UNKNOWN','Actualtensoraddresses,cacheblockownership,paddedrows,capturefixedstorage andevents not recoveredfrom typedshapes.'),
        ('cross_iteration_consumers','PARTIAL_DIRECT_CONSUMER_REVIEW','Futureindex/attention/foldedcache/rollbackgenerations kept;not allmoduleclosure.'),
        ('logical_bytes','DERIVED_DECLARED_DOMAIN_ONLY','Inputsharingoperandexamples24/6400/6528/6668rows;notHBM/networklowerbound,gain orMFU.'),
        ('legal_or_gain','CONDITIONAL_UNPRICED','No targetimplementation,NPU/numericalexperiment orcriticalpathgain inferred.')]
    write(O/'F33-condition-ledger-v27.json',dict(generated_utc=stamp,candidate_id='F33',conditions=[dict(id=f'F33-v27-{k+1:02d}',branch=x,state=y,evidence=z)for k,(x,y,z)in enumerate(conditions)],
        all_conditions_closed=False,overall_complete=False))
    model=deepcopy(read(O/'candidate-profile-ledger-v26.json'));framework=deepcopy(read(O/'framework-candidate-ledger-v26.json'))
    modules=deepcopy(read(O/'module-profile-ledger-v26.json'));matrix=deepcopy(read(O/'module-phase-branch-matrix-v26.json'))
    assert len(model['candidates'])==34 and len(framework['candidates'])==7 and len(modules['modules'])==26
    f=next(c for c in model['candidates']if c['id']=='F33')
    f['profile_review_state']='EXACT80_ORDER_AND_OVERLAP_REVIEWED;DIRECT_CONSUMER_WAR_AND_NATIVE_PRIMITIVES_BOUND;CONDITIONAL'
    f['conclusion_changing_gap']='Actualstorage/strides/capture/eventgenerations,backendnumerics andcompletecacheownership/consumers;cross-rankreplication onlyforF32composition. Localjointprojectiondoesnotrequireequalitybetweenthetwomatrices.'
    f['next_profile_or_contract_action']='Do not repeat settled direct consumers. Advance finite wholemodule/phase/crossboundary coverage;point-query only a particular missing contract that changes conclusions.'
    f['review_v27']=dict(profile='F33-profile-order-contract-v27.json',consumer='F33-consumer-and-numerical-contract-v27.json',math='F33-projection-algebra-and-layout-v27.json',graphs='F33-joint-projection-variants-v27.json',conditions='F33-condition-ledger-v27.json',two_GEMMs_overlap=False,other_work_overlap=True,old_latent_WAR_required=True,whole_module_closed=False)
    f['evidence_files'] += ['F33-profile-order-contract-v27.json','F33-consumer-and-numerical-contract-v27.json','F33-projection-algebra-and-layout-v27.json','F33-joint-projection-variants-v27.json','F33-condition-ledger-v27.json']
    master=deepcopy(f);master.update(source_graph=baseline,candidate_graph=joint,variants=variants,implementation=False,new_performance_result=False,overall_complete=False)
    assert not (R/'candidates/F33.json').exists(),'existingmasterrequiresreviewbeforeoverwrite'
    write(R/'candidates/F33.json',master)
    notes={'M07':'F33two512norm/actualF.linear,nativeBF16andmultistreamoverlapreviewed;packedstrideandnumericalcontract remainconditional.',
           'M08':'C1latentindexread beforelonginplaceRoPE WARbound;C2FP32domain remainsdistinct.',
           'M09':'update_keys ownsK/empty,128norm/keyRoPE,packed/folded andINT8key/FP16scalepublications bound;notallIndexerbranches.',
           'M10':'SWA/longcachepublications havepacked/scatterbranches;typedshapesarenotphysicalaliasproof.',
           'M22':'Conditionaljointgraph explicitly removescurrentSWAjoincontrol;retainingitasjointselfdependencywouldinventcycle. Actualevent/addressgenerationsopen.',
           'M23':'F32projectionbeforegatherrequirescrossrankparameters/routing/precision;logicalfeaturewidthratio5 isnotnetworkgain.',
           'M24':'Indexoldlatent WARretained;futurecacheowner/generation/rollback observers remainopen.'}
    for obj,parent,dest in [(model,'candidate-profile-ledger-v26.json','candidate-profile-ledger-v27.json'),(framework,'framework-candidate-ledger-v26.json','framework-candidate-ledger-v27.json'),(modules,'module-profile-ledger-v26.json','module-profile-ledger-v27.json')]:
        obj.update(generated_utc=stamp,parent_ledger=parent,whole_modules_closed=0,overall_complete=False)
        if obj is model:obj['framework_candidate_ledger']='framework-candidate-ledger-v27.json'
        if obj is modules:
            for m in obj['modules']:
                m['phase_branch_record']='module-phase-branch-matrix-v27.json#'+m['id']
                if m['id']in notes:m['review_v27']=dict(finding=notes[m['id']],evidence=f['evidence_files'][-5:],whole_module_closed=False)
                if m['id']in ['M09','M10','M22','M24']and'F33'not in m['candidate_links']:m['candidate_links'].append('F33')
        write(O/dest,obj)
    matrix.update(generated_utc=stamp,parent_matrix='module-phase-branch-matrix-v26.json')
    for m in matrix['modules']:
        if m['id']in notes:m['branches'].append(dict(branch='F33source-C1/SWAdualprojection/consumer/numericalandpublicationboundaries',state='SCOPED_PROFILE_AND_SOURCE_REVIEWED;WHOLE_MODULE_OPEN',evidence=notes[m['id']],gap='Actualalias/precision/ownership/events andotherfullmodulebranches.'))
    write(O/'module-phase-branch-matrix-v27.json',matrix)
    remaining=deepcopy(read(O/'module-coverage-gaps-v26.json')['remaining'])
    remaining[-1]=remaining[-1].replace('F33 andallotherboundaryconsumers remainopen;','F33directindex-old-latent/WAR/publication/primitivequestions arenowreviewed;actualalias/events/numerics andallotherboundaryconsumers remainopen;')
    remaining.insert(1,'F33all80C1/SWAprojectionpairs have0overlap,butall80showotherCube/vectoroverlap. Do notvalue89.742–98.362usprojectiongap asidle/removable. LocaljointGEMM needstwo distinctweights,notA=B. Indexoldlatent readmustprecedeinplacelongRoPE;wholeindexpublicationorder iscurrentcontrol,notuniversalbarrier. Directquestionsareanswered;advancefinitewholemodulecoverage insteadofrepeatedlyauditingalreadyconditionalcandidates.')
    write(O/'module-coverage-gaps-v27.json',dict(generated_utc=stamp,remaining=remaining,whole_modules_closed=0,overall_complete=False))
    summary=dict(generated_utc=stamp,status='PROGRESS',previous_turn_classification='PROGRESS:v26H07exact64roots/Pythonmaxnorm/directbufferhandoffs',
        graph_pairs=80,projection_pair_overlap_ns=ps['projection_overlap_ns'],projection_pair_gap_ns=ps['C1_start_minus_SWA_end_ns'],
        other_overlap_ns=au['observed_overlap_ns'],native_selected_TASKs=2382,native_C1_projections=24,
        direct_old_latent_WAR_bound=True,two_independent_512_norms_required=True,condition_branches=len(conditions),
        source_files=len(consumer['source_files']),model_candidates=34,framework_candidates=7,
        all_jobs_terminal=True,no_new_model_device_service_or_health_work=True,evidence_refs=refs,
        measured_gain=None,whole_modules_closed=0,overall_complete=False,remaining=remaining)
    write(O/'F33-consumer-summary-v27.json',summary)
    gate=read(O/'PROFILE-CODE-QUESTION-GATE-v27.json');gate.update(status='PROFILE_AND_NECESSARY_POINT_SOURCE_REVIEWED',
        answered_by=f['evidence_files'][-5:],closed_scope='80exactpairorder/otheroverlap,actualprimitive/normanddirectindexold-latent/cachepublicationcontracts.',
        not_closed='Historicalpointers/strides/eventgenerations,backendprecision/allcacheobservers/fullmodulecoverage/gain.',stop_rule=consumer['stop_rule']);write(O/'PROFILE-CODE-QUESTION-GATE-v27.json',gate)
    report='''本轮按profile优先推进F33，复核保存的80个graph block20中的C1/SWA双投影。两个5120→512 BF16 MatMul本身在80组中均无重叠；C1启动晚于SWA结束89.742–98.362微秒，中位94.771微秒。这个间隔包含其他计算与缓存工作，不是空闲时间或可删除延迟。进一步逐图核对发现，SWA投影与Q norm、SWA512 norm与Q_b、C1投影与index-query量化在全部80组中都有交叠，中位交叠分别4.260、11.990、5.480微秒。联合投影必须比较完整排程，不能只数两次调用。

每组保留两个独立24×512 norm候选，原始TASK行键、stream、顺序、COMM和全部block计算记录均保留。角色由已有源码和唯一SWA辅助流约束，不冒充历史tensor pointer证明。2382条原生来源TASK分为target SWA投影/norm各960、target C1投影/norm各24、draft SWA投影/norm各207。target C1的24条投影分别为6400、6528、6668行各8条；没有把所有SWA或draft算子当作第20层，也没用最近时间配出native同层关系。

Trace不能独自说明latent别名和具体副作用，所以仅定点读取DSA压缩源/写缓存/旋转方法，以及实际native帧指向的F.linear、CV matmul和RMSNorm入口。SWA当前记录命中unquantized_gemm→torch.nn.functional.linear，norm命中torch_npu.npu_rms_norm。C1的既有constructor/forward资料复用。GEMM输出与norm输入均为BF16；融合可研究保留舍入边界而不物化到HBM，但不能直接把FP32 accumulator输入norm当作等价。精确后端算术与历史stride仍未知。

DSA先调用indexer.update_keys(latent)，随后view同一latent并原位做long-KV RoPE。沿profile已有indexer.py:110 caller和这一个直接callee补查update_keys，确认它先将未做long-RoPE的512维latent投到128维index key，再独立norm、旋转新key，并选择packed/folded缓存发布，或INT8 key与FP16 scale分别scatter。因此必须保留index投影读完旧latent→原位long-RoPE写latent的WAR；若提前旋转而不另存原值，会改变index输入。当前整个index发布在long-RoPE之前是源码控制顺序，不能自动升级为所有实现都必须等待的语义屏障。F33单独变体保留它，F19式重排另审。

SWA有packed kind=win与scatter分支，long KV有kind=cmp与scatter分支；indexer还有!owns_k/empty早退与folded缓存分支。请求的_write_index_cache/_write_long_cache方法名在DSA不存在，实际发布为inline代码和update_keys，不据缺少名字推断没有写缓存。形状相同也不证明SWA/long缓存为同一物理地址；保留分配、block所有权、generation和未来读者条件。

上述WAR约束按被覆盖的旧值成立，可按行或tile细分流水，不要求所有新实现都等待整个index投影算子结束。

数学/布局记录区分三个问题。第一，共享输入联合计算X*A^T和X*B^T不要求A=B，必须保留两个参数矩阵；同形状不能支持把结果去重。第二，两个512宽norm不可改成一个1024宽norm，精确有理数反例展示全1/全2两组具有不同分母。第三，连续24×1024的两半视图行stride为1024，与连续24×512的512不同；可用支持stride的消费者、独立输出或显式转换，不能断言任何实现都必须多一次copy。

另列24/6400/6528/6668行的逻辑operand计数。它只说明在“各operand读一次”的示例域中，联合读入可少一份输入；没有把字节当HBM、带宽下界或收益。F32的先本地投影后row allgather在精确算术、同rank参数/相同行序条件下成立，但实际跨rank权重、精度、padding、位置、全局hidden消费者与缓存所有权仍须满足；特征宽度比5不是网络提速比。

F33新保存原始source图与条件联合图，明确移除需要重新审查的SWA发布→C1启动控制边，避免把旧控制边错误保留为joint节点自依赖。保留index旧latent WAR、两个norm与全部后续发布。新增20项条件记录，补齐M09/M10/M22/M24的F33跨边界链接，并新增自包含candidates/F33.json。

本轮4项CPU-only取证均成功，无新模型导入、NPU负载、服务修改或健康检查。34项模型、7项框架候选不变，整模块闭合数仍0，目标继续active。F33直接消费者问题已回答，应转入尚未闭合的有限模块/条件边界，避免为已可定性的条件候选继续扫整库。历史地址/事件、后端精度、所有缓存观察者及全范围理论证据仍按未知保留。
'''
    (O/'PROFILE-FINDINGS-v27.md').write_text(report)
    action=('继续Loop004，v27实质进展。先读F33-consumer-summary-v27.json、F33-consumer-and-numerical-contract-v27.json、F33-joint-projection-variants-v27.json、module-coverage-gaps-v27.json。'
        '80C1/SWAprojectionpair本身0overlap；C1晚89.742–98.362us启动不是idlegain。全部80有SWAproj/Qnorm、SWA512norm/Qb、C1proj/indexqueryquant交叠，中位4260/11990/5480ns。'
        '原生2382TASK=2*(960targetSWA+24targetC1+207draftSWA)，C1nativeM6400/6528/6668各8；勿按形状把全SWA归layer20或neartime配对。'
        '实际SWA F.linear(opslinear49–54,SHA7654777d9252fc6a3cc2c95f92b67a42410a0de24a5912ef20e807cf64ab74e6)，cvmatmul172，norm99 npu_rms_norm SHA d2fda42ad99935725c98b9505a8d2e94696d9cdb2906e5cb2f1e37bf21fb27bc。'
        'DSA544 update_keys先读latent，550view/551inplacelongRoPE；indexer106–125 SHA d5d6a438bbced4e63d729dac94c08869df30dab4a71ceacecdea211386b3db33：wk(latent)128norm→keyRoPE→packed/folded或INT8key/FP16scale；必须旧latentread完成再inplace写，整个indexpublish->longRoPE只是currentcontrol，F33alone保留，F19另域。'
        '两个512norm/参数/epsilon/可选bias及BF16边界保留；joint1024输出halves stride1024，非当前连续512。LocaljointGEMM保留A,B，不要求A=B；跨rank参数复制仅F32移动投影前gather组合需证。'
        'R/candidates/F33.json现已保存source/jointDAG；joint移除当前SWAjoincontrol而非保留自依赖。20conditions；M09/10/22/24补F33links。'
        'M/F33-C1-SWA-projection-order-and-consumer-surfaces-v27 SHAda20a123d7cf2c113efbc93d82979e2a404e337319062819160ae60f26711321；consumerprimitive SHAce40420e1af55fe6fe5a9f9d07e888aa0866147d9c61012338baa51515261109；audit SHA67853043943963cd320557038738cdd67bad34c4852f30051a1f13e309657820；indexcallee SHAb269a2b30f3bb2c4a8023d2bd32bd3bf1bf350e441dd3c6d5ce94c0db1ae9b13。'
        '下一步从当前26模块有限phase/branch矩阵与原始图边覆盖台账选择未审语义边界，优先填真实覆盖缺口，而非反复对F33/H07/F34已回答局部问题加码。先profile/fullcaller/capture metadata，缺关键语义再点代码；明确未执行/不适用/未知/已审条件，不能空表当全覆盖。'
        'F33仍alias/events/后端precision/未来cacheowner等条件，非已验证可执行优化；本任务不实现。保留H07v26 exact64+24long+8empty/16unbound、maxnormNone与futurebuffer guard条件；F34v25SWAanchor不变下独立ringhalf allocationgain排除。'
        '34model/7framework/0wholemodule；3136candidate-set资源/849missingPMU/16unboundgraphs/26theoryINCOMPLETE不变。EngineCore/producerready/prefixhash若决定性，既有finite诊断授权需freshhealth/ownership/frozenworkload/clone/restore，不能用旧PID或假定idle。'
        '4remoteCPUjobs全terminalpass，无newmodel/NPU/service/health，lasthealthv12post；Goalactive整体未完成。')
    jobs=[('profile','All80block20source-streamprojectionpairs andnativeprimitivecallerTASKs extractedfrompinned savedartifacts withoutnewsource/devicework.',[N/'query_F33_projection_consumers_v27_166.py',N/'F33-profile166-v27.json']),
          ('review','NamedDSApublication/rotary andnativeF.linear/CV/RMS入口 sourceextracted,missinghelpernames retainedwithoutinventingabsence.',[N/'read_F33_consumer_and_primitive_contract_v27_166.py',N/'F33-source166-v27.json']),
          ('review','Exact80ordersequences/otheroverlap and2382nativecallerroles auditedwithoutneartime/pointerclaims.',[N/'audit_F33_norm_and_schedule_v27_166.py',N/'F33-audit166-v27.json']),
          ('review','Profile/dsa-callraised indexer.update_keys singlecallee bindsold-latentreader andpacked/folded/keyscale publicationbranches.',[N/'read_F33_indexer_update_keys_v27_166.py',N/'F33-index-consumer166-v27.json']),
          ('review','Mainsemanticreview bindsWAR/independentnorm/numericalboundaries,correctsweight-equalityscope andrecords20conditionalbranches withvalidatedacyclic source/jointgraphs.',[O/'F33-consumer-and-numerical-contract-v27.json',O/'F33-projection-algebra-and-layout-v27.json',O/'F33-joint-projection-variants-v27.json',O/'F33-condition-ledger-v27.json',R/'candidates/F33.json',O/'PROFILE-FINDINGS-v27.md'])]
    runs=[]
    for k,(kind,desc,arts)in enumerate(jobs):
        rid='run-'+(now-timedelta(seconds=len(jobs)-k)).strftime('%Y%m%dT%H%M%SZ')
        run=ctl('new-run','--loop-dir',str(L),'--run-id',rid,'--case-key','c8-input32k-output1k-prefix90-166','--kind',kind,'--command','Preparedv27retrospectiveregistration: '+desc,'--code-ref','Saved166profile andspecificconsumer source;targetunchanged;noNPU/model/servicework','--working-directory',str(N))
        args=['finish-run','--run-dir',run,'--status','pass','--correctness','not-applicable','--summary',desc]
        for q in arts:assert q.exists(),q;args+=['--artifact',str(q)]
        ctl(*args);runs.append(dict(run_dir=run,status='pass',description=desc))
    cp=dict(status='PROGRESS',previous_turn_classification=summary['previous_turn_classification'],
        current_verified_changes=['80pairsexactorder/no projectionpair overlap but3otheroverlapsets reviewed;2382nativeTASKcontextssplit.',
            'Directindexold-latentWAR,nativeF.linear/RMSnorm,BF16/two512norm/packedstride andpacked/folded/keyscale publicationcontracts bound.',
            'F33source/jointDAGand20conditionssaved;localtwo-matrixequality requirementcorrected;4missingmoduleboundarylinksadded.'],
        summary=str(O/'F33-consumer-summary-v27.json'),candidate_ledger=str(O/'candidate-profile-ledger-v27.json'),framework_candidate_ledger=str(O/'framework-candidate-ledger-v27.json'),
        module_ledger=str(O/'module-profile-ledger-v27.json'),branch_matrix=str(O/'module-phase-branch-matrix-v27.json'),report=str(O/'PROFILE-FINDINGS-v27.md'),
        source_gate=str(O/'PROFILE-CODE-QUESTION-GATE-v27.json'),coverage_gaps=str(O/'module-coverage-gaps-v27.json'),pending_full_scope_domains=remaining,
        remote_job_status='4CPU-onlyremotejobs terminalpass;noNPU/model/service/healthwork.',model_candidates=34,framework_candidates=7,next_action=action,whole_modules_closed=0,overall_complete=False,updated_utc=stamp)
    write(O/'REVIEW-CHECKPOINT-v27.json',cp)
    for q in [L/'state.json',T/'index.json']:
        d=read(q);assert d['status']in['EVALUATING','ACTIVE'];d.update(next_action=action,updated_at=stamp);write(q,d)
    for q in [L/'status.md',T/'status.md']:
        q.write_text('\n'.join(('- 下一步：'+action)if x.startswith('- 下一步：')else('- 更新时间：`'+stamp+'`')if x.startswith('- 更新时间：')else x for x in q.read_text().splitlines())+'\n')
    (O/'profile-summary-before-v27.json').write_bytes((R/'profile-summary.json').read_bytes())
    prof=read(R/'profile-summary.json');prof['supplementary_F33_consumers_v27']=dict(summary=cp['summary'],graph_pairs=80,old_latent_WAR_bound=True,gain=None,whole_modules_closed=0,overall_complete=False);write(R/'profile-summary.json',prof)
    for q in [N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:
        d=read(q);d.update(latest_profile_first_review=cp,overall_complete=False,whole_modules_closed=0,updated_utc=stamp);d['next_read_only_action'if q==R/'STATE.json'else'next_action']=action
        if q==W/'STATE.json':d.update(profile_candidate_ledger=cp['candidate_ledger'],profile_module_ledger=cp['module_ledger'])
        if q in[N/'RECOVERY.json',R/'PARALLEL-RESUME-PACK.json']:d['read_first']=[cp['summary'],str(O/'F33-consumer-and-numerical-contract-v27.json'),str(O/'F33-joint-projection-variants-v27.json'),cp['coverage_gaps']]
        write(q,d)
    compact=ctl('compact','--task-dir',str(T));validation=ctl('validate','--task-dir',str(T));write(O/'task-run-registration-v27.json',dict(runs=runs,compact_output=compact,validation_output=validation,whole_modules_closed=0,overall_complete=False))
    arts=[q for q in O.glob('*v27*')if q.is_file()and not q.name.startswith('checkpoint-manifest')]
    arts += [N/(x+'.json')for x in ['F33-profile166-v27','F33-source166-v27','F33-audit166-v27','F33-index-consumer166-v27']]
    arts += [N/x for x in ['query_F33_projection_consumers_v27_166.py','read_F33_consumer_and_primitive_contract_v27_166.py','audit_F33_norm_and_schedule_v27_166.py','read_F33_indexer_update_keys_v27_166.py','finalize_F33_consumers_v27.py']]
    arts += [q for q in N.glob('zcode-F33-*v27*20261031*')if q.is_file()]
    arts += [R/'candidates/F33.json',N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json',R/'profile-summary.json',L/'state.json',T/'index.json']
    arts=sorted(set(arts));write(O/'checkpoint-manifest-v27.json',dict(generated_utc=stamp,artifacts=[dict(path=str(q),bytes=q.stat().st_size,sha256=hashlib.sha256(q.read_bytes()).hexdigest())for q in arts],whole_modules_closed=0,overall_complete=False))
    print(json.dumps(dict(checkpoint=str(O/'REVIEW-CHECKPOINT-v27.json'),manifest_files=len(arts),pass_runs=len(runs),condition_branches=len(conditions),model_candidates=34,framework_candidates=7,validation=validation,whole_modules_closed=0,overall_complete=False),ensure_ascii=False))


if __name__=='__main__':main()
