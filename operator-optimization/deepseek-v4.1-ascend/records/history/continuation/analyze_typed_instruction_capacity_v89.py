"""Saved-profile instruction-domain capacity cases; no native execution or framework audit."""
from pathlib import Path
from html.parser import HTMLParser
from collections import Counter
from math import prod, ceil
import hashlib, re
from finalize_origin_coverage_v9 import N,O,read,write
from analyze_state_engram_v62 import dimensions


class Tables(HTMLParser):
    def __init__(self):
        super().__init__();self.tables=[];self.table=None;self.row=None;self.cell=None
    def handle_starttag(self,t,a):
        if t=='table':self.table=[]
        elif self.table is not None and t=='tr':self.row=[]
        elif self.row is not None and t in ['td','th']:self.cell=''
    def handle_data(self,s):
        if self.cell is not None:self.cell+=s
    def handle_endtag(self,t):
        if t in ['td','th'] and self.cell is not None:self.row.append(' '.join(self.cell.split()));self.cell=None
        elif t=='tr' and self.row is not None:self.table.append(self.row);self.row=None
        elif t=='table' and self.table is not None:self.tables.append(self.table);self.table=None


class MathTree(HTMLParser):
    def __init__(self):super().__init__();self.stack=[];self.root=[]
    def handle_starttag(self,t,a):
        x=dict(attrs=dict(a),children=[]);(self.stack[-1]['children']if self.stack else self.root).append(x);self.stack.append(x)
    def handle_endtag(self,t):
        if self.stack:self.stack.pop()
    def handle_data(self,s):
        if self.stack and s.strip():self.stack[-1]['children'].append(s)


def math_text(x):
    if isinstance(x,str):return x
    a=x['attrs'];typ=a.get('data-mml-node');c=a.get('data-c');children=x['children']
    if c:return chr(int(c,16))
    href=a.get('xlink:href','')
    if href and 'TEX-'in href:
        try:return chr(int(href.rsplit('-',1)[1],16))
        except ValueError:pass
    if typ in ['mtr','mfrac','msub','msup']:
        return typ+'('+','.join(math_text(k)for k in children)+')'+('\n'if typ=='mtr'else'')
    return ''.join(math_text(k)for k in children)


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not(O/'REVIEW-CHECKPOINT-v89.json').exists()
    receipts=read(O/'typed-capacity-primary-receipts-v89.json')
    # Receipts retain original download times and the distinction from installed CANN.
    for r in receipts['primary_sources']:
        p=Path(r['path']);assert p.stat().st_size==r['bytes']and digest(p)==r['sha256']
    cube=(O/'Cube-theoretical-primary-v89.html').read_text();vector=(O/'Vector-theoretical-primary-v89.html').read_text()
    c=Tables();c.feed(cube);v=Tables();v.feed(vector);assert len(c.tables)==8 and len(v.tables)==18
    write(O/'Cube-primary-tables-v89.json',c.tables);write(O/'Vector-primary-tables-v89.json',v.tables)
    assert c.tables[0][5][1:]==['float','float','16','16','4','8']
    assert c.tables[0][6][1:]==['float','float','16','16','8','8']
    assert c.tables[1][9][1:]==['float','float','16','16','1','8']
    assert all(r[4]!='float'for r in c.tables[6][1:])
    first_math=re.search(r'<mjx-container.*?</mjx-container>',cube,re.S).group()
    mt=MathTree();mt.feed(first_math);decoded=''.join(math_text(x)for x in mt.root)
    assert decoded.count('mtr(')==4 and 'Δ' in decoded and 'mfrac' in decoded
    footers=[]
    for r in receipts['primary_sources']:
        s=Path(r['path']).read_text();commit=re.search(r'asc-devkit/commit/([0-9a-f]{40})',s)
        footers.append(dict(path=r['path'],observed_footer_commit=commit.group(1)if commit else None,
          uncommitted_changes_footer_present='包含未提交修改'in s,
          snapshot_SHA_is_authority_for_received_document_not_clean_commit_attestation=True))
    write(O/'Cube-formula-decoded-v89.json',dict(svg_glyph_tree_text=decoded,
        normalized_formula='ceil_m=16*ceil(m/16);ceil_n=16*ceil(n/16);ceil_k=k0*ceil(k/k0); utilization=(m*n*k/P)/(delta_t+ceil_m*ceil_n*ceil_k/P);P=cube_m*cube_n*cube_k.',
        delta_t_cycles_unknown=True,meaning='Official hardware parallelism is a throughput service cap,not a measured duration. Shape/head overhead reduces useful utilization.',
        source_html='Cube-theoretical-primary-v89.html'))
    rates={}
    for row in c.tables[0][1:7]:
        interface,a,b,m,n,k,k0=row
        name='FLOAT_HF32_ON'if 'HF32'in interface else {'bfloat16_t':'BF16','half':'FP16','float':'FLOAT_HF32_OFF','int8_t':'INT8','int4b_t':'INT4'}[a]
        rates[name]=dict(cube_m=int(m),cube_n=int(n),cube_k=int(k),k0=int(k0),MAC_per_AIC_cycle=int(m)*int(n)*int(k),
          ops_2perMAC_per_AIC_cycle=2*int(m)*int(n)*int(k),source_table=1,architecture='2201')
    selected_vector=[]
    for ti in [6,7,8,9,10,11,13,14,15]:
        for row in v.tables[ti-1][1:]:
            if row[0]in ['Add','Sub','Mul','Exp','Ln','Sqrt','Rsqrt','Div','Adds','Muls','FusedMulAdd','MulAddDst','Compare','Select','Cast',
              'ReduceDataBlock','ReduceRepeat','ReducePairElem','ReduceMax/ReduceMin/ReduceSum','Duplicate','Sort32','MrgSort','Gather','Gatherb']:
                selected_vector.append(dict(table=ti,columns=v.tables[ti-1][0],row=row))
    write(O/'typed-capacity-primary-contract-v89.json',dict(
        receipt_file='typed-capacity-primary-receipts-v89.json',document_build_footers=footers,cube_A3_2201=rates,vector_selected_A3_A2_rows=selected_vector,
        architecture_guard='Only Cube table1/3/7 (2201),Vector tables6..18 (AtlasA3/A2). 3510 FLOAT k1 and 950 UB/Fixpipe routes are not imported.',
        source_version_scope='Current official asc-devkit public documentation snapshot. Not an installed CANN9.2 binary,active native mode,physical maximum clock or all-SKU attestation.',
        per_die_reference=dict(AIC=24,AIV=48,core_count_scope='Existing 910C per-die reference model from v80;not newly measured or a multi-die aggregation.'),
        Cube_formula='Cube-formula-decoded-v89.json',L0C_to_L1_documented_2201_rows=c.tables[6],
        float_L0C_to_L1_lossless_route_documented_in_this_table=False,
        absence_is_universal_unsupported_proof=False,
        bridge_gate='The listed FLOAT→BF16/half routes round values. FLOAT C2 state and HC coefficients cannot use them as a lossless bridge without a new numeric contract. This main-instruction summary is not proof that every other API/path is unsupported.',
        HF32_contract=dict(only_FLOAT_inputs=True,rounds_inputs=True,output_remains_FLOAT=True,
          sticky_until_explicit_change=True,round_mode_requires_configuration=True,
          persistence_across_kernel_context_boundaries_proved=False,native_active_mode_proved=False,
          numeric_literal_1_in_old_SetHF32TransMode_mapped_to_new_C_API_enum=False),
        bound_rule='For one declared necessary instruction-work cut,cycles>=work/(core_count*documented_rate). Across distinct cuts use max. No weighted joint Cube-mode or Vector opcode issue simplex is proved by separate table rows.',
        reductions='ReduceRepeat FLOAT64/7 differs from ReduceDataBlock FLOAT64. High-level ReduceSum/Max/Min are soft emulation;no whole RMS/softmax reduction rate is supplied.',
        sort='Sort32/MrgSort FLOAT2 elements/cycle is an instruction case,not a universal TopK algorithm bound.',
        no_frequency_scaling_or_physical_MFU=True,actual_wallclock_upper_or_speedup_proved=False,overall_complete=False))

    matrix=[];skipped=[]
    source=read(O/'operator-shape-ceiling-ledger-v55.json')['observed_regimes']
    dtmap={'DT_BF16':['BF16'],'INT8':['INT8'],'FLOAT':['FLOAT_HF32_OFF','FLOAT_HF32_ON']}
    for r in source:
        if r['operator']not in ['MatMulV2','MatMulV3','QuantBatchMatmulV3']:continue
        try:a,b,*_=dimensions(r['signature']['inputShapes_text']);out=dimensions(r['signature']['outputShapes_text'])[0]
        except(ValueError,IndexError):skipped.append(r['regime_id']);continue
        dt=r['signature']['inputDataTypes_text'].split(';')
        if not(len(a)==len(out)==2 and a[0]==out[0] and prod(b)==a[1]*out[1] and dt[0]in dtmap and dt[0]==dt[1]):
            skipped.append(r['regime_id']);continue
        M,K,C=a[0],a[1],out[1];ops=2*M*C*K;W=K*C*{'DT_BF16':2,'INT8':1,'FLOAT':4}[dt[1]]
        cases=[]
        for name in dtmap[dt[0]]:
            q=rates[name];capacity=24*q['ops_2perMAC_per_AIC_cycle'];padded=2*(16*ceil(M/16))*(16*ceil(C/16))*(q['k0']*ceil(K/q['k0']))
            cases.append(dict(mode=name,documented_ops_capacity_per_die_cycle=capacity,
              useful_dense_MAC_work_cycle_floor=ops/capacity,
              independently_padded_whole_product_service_reference_cycles=padded/capacity,
              padding_reference_is_actual_tiling_or_all_fusions_lower_bound=False))
        B=max(0,W-24*65536)/(24*128);both=max(0,W-24*2*65536)/(24*384)
        matrix.append({k:r[k]for k in ['regime_id','operator','module_candidates','role','origin','context','executions','signature']}|dict(
            M=M,N=C,K=K,input_dtype=dt[0],dense_ops_2perMAC=ops,distinct_uncompressed_weight_bytes=W,mode_cases=cases,
            L1_only_fixed_B_weight_floor_cycles=B,L1_only_flexible_orientation_relaxed_weight_floor_cycles=both,
            joint_with_fixed_B_weight_cut_cycle_floors={x['mode']:max(B,x['useful_dense_MAC_work_cycle_floor'])for x in cases},
            native_mode_or_instruction_lowering_proved=False,new_profile_regime_review_promotion=False,actual_wallclock_floor_us=None))
    assert len(matrix)==118 and not skipped and sum(r['executions']for r in matrix)==35334
    assert Counter(r['input_dtype']for r in matrix)=={'DT_BF16':66,'INT8':31,'FLOAT':21}
    write(O/'A3-matrix-typed-capacity-cases-v89.json',dict(regimes=matrix,regime_count=len(matrix),native_executions=sum(r['executions']for r in matrix),
        input_dtype_regime_counts=dict(Counter(r['input_dtype']for r in matrix)),FLOAT_regimes=21,FLOAT_executions=5304,
        BF16_INT8_preexisting_v80_regimes=97,BF16_INT8_preexisting_v80_executions=30030,
        population='Every captured MatMulV2/MatMulV3/QuantBatchMatmulV3 regime with this two-dimensional dense geometry and same-typed A/B. Not all matrix work inside attention/MC2/HcPre or every possible algorithm.',
        compute_domain='Conventional dense Mmad on 2201 with indicated precision/format and 24AIC maximum reference capacity. Modes are cases,not permission to change original precision. Alternative algorithms,compression,sparsity,memoization and mixed decompositions require new work models.',
        ingress_domain='As v80: mandatory distinct uncompressed weights,declared L1→L0 path,fixed B128 or relaxed A256+B128;single initial L0 capacity grant. GM→L0 bypasses this cut. Not actual GM/HBM traffic.',
        padding_domain='Official independent-product m/n16,k0 rounding,ideal 24-AIC work distribution,zero head overhead. A reference;no native tiling or fused packing proof.',
        global_uncertainties_changed=False,whole_module_MFU_ranking_proved=False,overall_complete=False))

    mixed=[]
    old=read(O/'mixed-SWA-C2-cost-models-v88.json')
    for x in old['models']+[{ 'N':3,'H':5120,'D':512,'exact_new_native_triple_join':False,'owner_local_transformed_case':True}]:
        m,h,d=x['N'],x['H'],x['D'];ops=2*m*h*d
        useful=dict(SWA_BF16=ops/(24*8192),C2_two_FLOAT_HF32_OFF=2*ops/(24*2048),
          C2_two_FLOAT_HF32_ON=2*ops/(24*4096),three_BF16_reencoded=3*ops/(24*8192))
        mp=16*ceil(m/16)
        pad=dict(SWA_BF16=2*mp*h*d/(24*8192),C2_two_FLOAT_HF32_OFF=4*mp*h*d/(24*2048),
          C2_two_FLOAT_HF32_ON=4*mp*h*d/(24*4096))
        cuts={r['representation']:r for r in old['weight_ingress_cuts']if r['C2_source_windows']==1}
        orig=cuts['original_typed_planes']['fixed_B_min_aggregate_elapsed_cycles'];rec=cuts['conditional_exact_BF16_reencoding']['fixed_B_min_aggregate_elapsed_cycles']
        mixed.append(dict(rows=m,H=h,D=d,old_native_triple_geometry=x.get('exact_new_native_triple_join',False),
          transformed_owner_local=x.get('owner_local_transformed_case',False),useful_dense_service_cycle_floors=useful,
          independent_padded_matrix_service_references=pad,original_fixed_B_weight_cut_cycles=orig,reencoded_fixed_B_weight_cut_cycles=rec,
          original_fixed_B_combined_max_cycle_floors={mode:max(orig,useful['SWA_BF16'],useful['C2_two_FLOAT_'+mode])for mode in ['HF32_OFF','HF32_ON']},
          exact_reencoded_fixed_B_combined_max_cycle_floor=max(rec,useful['three_BF16_reencoded']),
          original_hypothetical_lossless_BF16_reencoding_proved=False))
    assert mixed[0]['original_fixed_B_weight_cut_cycles']==8022 and mixed[-1]['rows']==3
    write(O/'mixed-projection-capacity-repricing-v89.json',dict(models=mixed,
      native_N24_to_ownerlocal_M3=dict(useful_FLOAT_pair_work_ratio=8,
        independent_product_padding_reference_ratio=2,padding_reference_is_actual_speedup=False,
        unchanged_weight_cut_cycles=8022,
        conclusion='In fixedB/L1/uncompressed/oneinitialgrant domain,8022 is larger than individual useful compute cuts for bothN24 andM3,in bothFLOAT modes. Raising compute rate alone doesnot lower this particular max-cut floor. This doesnot identify the actual bottleneck or make two executions equally fast.'),
      mixed_mode_combined_compute_simplex_proved=False,
      reencoded_domain='Exact C2 weight representability,original reduction/rounding and retained C2FLOAT outputs required;v88 counterexamples remain. BF16 rate isnot an allowed substitute for arbitrary FLOAT operands.',
      no_independent_shared_gather_or_cast_credit_added=True,new_native_work=False,actual_wallclock_or_speedup_proved=False,overall_complete=False))

    router=[]
    for r in read(O/'router-vector-regime-models-v63.json')['regimes']:
        if r['operator']!='MoeGatingTopKHash':continue
        x=r['work_model'];n=x['M']*x['E'];k=x['M']*x['k'];active=min(48,r['signature']['blockNum'])
        instructions=[('Exp',n,32),('Ln',n,32),('Sqrt',n,32),('Adds_one',n,64),('Add_bias',n,64),('Div_selected_weights',k,32),('Muls_selected_scale',k,64),('Gather_selected_unbiased_weights',k,5)]
        cuts=[dict(instruction=name,source_elements=count,documented_elements_per_AIV_cycle=rate,
          full48AIV_cycle_floor=count/(48*rate),recorded_block_count_launch_reference_cycle_floor=count/(active*rate))for name,count,rate in instructions]
        # High-level Sort is pinned,its actual Sort32/MrgSort binary lowering is not.
        sort_elements=x['M']*32*ceil(x['E']/32)
        router.append(dict(regime_id=r['regime_id'],module_candidates=r['module_candidates'],executions=r['executions'],signature=r['signature'],
          M=x['M'],E=x['E'],k=x['k'],normType2_source_instruction_cuts=cuts,
          max_individual_direct_instruction_cycle_floor=max(q['full48AIV_cycle_floor']for q in cuts),
          full_sort_Sort32_lowering_case=dict(necessary_first_pass_elements=sort_elements,rate_per_AIV_cycle=2,
            full48AIV_first_pass_cycle_floor=sort_elements/96,merge_rounds_and_sync_excluded=True,
            native_Sort32_MrgSort_lowering_attested=False,applies_to_all_exact_TopK_algorithms=False),
          high_level_reduction_capacity_not_invented=True,independent_opcode_cut_sum_used=False,
          recorded_blocks_is_all_possible_fusions_parallelism_cap=False,native_loaded_binary_or_physical_time_proved=False))
    assert len(router)==7 and sum(r['executions']for r in router)==4472
    write(O/'router-Vector-instruction-cuts-v89.json',dict(regimes=router,regime_count=7,native_executions=4472,
      source='router-vector-numerical-contract-v63.json',
      domain='Pinned without-group normType2 FP32 Exp→+1→Ln→Sqrt,bias only for rank,sort/gather unbiased weights,selected reduction+eps/Div/scaling. Instruction caps price this declared lowering;not all equivalent sigmoid/softplus/TopK algorithms.',
      exact_sort32_lowering_is_conditional=True,full_E_reduction_or_fusion_communication_costs_retained=True,
      heterogeneous_instruction_rates_cannot_be_summed_without_joint_capacity=True,
      current_loaded_native_binary_or_instruction_counts_attested=False,physical_time_MFU_or_speedup_proved=False,overall_complete=False))

    hcfile=N/'HC-operator-source-v58/vllm-workspace/vllm-ascend/csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute.h'
    source_lines=hcfile.read_text().splitlines();slices=[]
    for first,last in [(128,135),(267,338),(418,437),(457,478)]:
        slices.append(dict(path=str(hcfile),sha256=digest(hcfile),first_line=first,last_line=last,source=source_lines[first-1:last]))
    hc=[]
    for x in read(O/'HC-work-traffic-and-reference-bounds-v58.json')['models']:
        q=x['source_conditioned_tiling']
        hc.append(dict(M=x['M'],semantic_projection_HF32_cycle_floor=x['pre_dense_projection_HF32_ops_2perMAC']/(24*4096),
          declared_source_requested_AB_HF32_work_cycle_floor=q['HF32_AB_requested_ops']/(24*4096),
          source_Gram_requested_work_HF32_OFF_case_cycle_floor=q['Gram16x16_requested_ops']/(24*2048),
          source_Gram_requested_work_HF32_ON_case_cycle_floor=q['Gram16x16_requested_ops']/(24*4096),
          source_Gram_mode_first_entry_unattested=True,
          post_individual_Mul_cycle_floor=x['post_FP32_multiplications']/(48*64),
          post_individual_Add_cycle_floor=x['post_FP32_additions']/(48*64),
          post_mul_add_joint_issue_sum_not_assumed=True))
    write(O/'HC-precision-mode-capacity-cases-v89.json',dict(models=hc,source_parts=slices,
      previously_reviewed_source_fact_first_version=58,new_mode_or_HF32_optimization_discovery_claimed=False,
      modes='A2 and AB buffers FLOAT. MmadAB explicitly enables HF32,sets transform mode1,and disables HF32 after its Mmad. ComputeDecode interleaves A2 thenAB. A2 hasno explicit mode setter in its body;native first-entry state isnot attested. Following AB returns modeoff in the declared source sequence.',
      fusion_gate='A fusion/split/reorder must preserve each Mmad precision mode,rounding,accumulation and required resets. New C_API sticky-mode semantics support this hazard;old SetHF32TransMode literal1 isnot assigned a new enum meaning.',
      source_requested_ops_not_actual_issued_instruction_count=True,
      source_Gram_lower_bound_not_universal_sumsq_lower_bound=True,
      mode_capacity_cuts_combined_by_max_only=True,
      actual_native_binary_mode_frequency_or_speedup_proved=False,overall_complete=False))
    print(dict(matrix_regimes=len(matrix),matrix_executions=35334,FLOAT_regimes=21,FLOAT_executions=5304,
      router_regimes=7,router_executions=4472,mixed_models=len(mixed),HC_models=len(hc),overall_complete=False))


if __name__=='__main__':main()
