"""Source-conditioned routing quantization reuse and work/interface models; offline only."""
from collections import Counter,defaultdict
from pathlib import Path
import hashlib,json,math
from finalize_origin_coverage_v9 import N,O,R,W,read,write
from mirror_saved_profile_core_v53 import CACHE
from analyze_state_engram_v62 import dimensions

BASE=W/'frozen-source-snapshot/vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_kernel/moe_init_routing_quant_v2'

def chunks(M,k,A):
    total=M*k;p=math.ceil(total/A)
    return [(a,min(a+p,total))for a in range(0,total,p)]

def row_evaluations(intervals,k):
    return sum((b-1)//k-a//k+1 for a,b in intervals)

def align32(x):return (x+31)//32*32

def case(M,k,E):
    A=40;H=5120;ub=196352;total=M*k
    sort_limit=(ub//32)//32*32
    full_space=align32(total)*4*9+align32(E*4)+13*H+64
    full=total<=sort_limit and H<=8192 and full_space<ub
    intervals=chunks(M,k,A);R=row_evaluations(intervals,k)
    row_size=align32(intervals[0][1]*4)*4;col_space=align32(H)*21+64
    assert row_size+col_space<ub and H!=7168
    alt=chunks(M,k,48);aligned_token_rows=math.ceil(M/48);aligned=[(r,min(r+aligned_token_rows,M))for r in range(0,M,aligned_token_rows)]
    return dict(M=M,H=H,topK=k,global_expert_namespace_including_sentinel=E,source_AIV_budget=A,
      expected_source_key=21000 if full else 11000,full_load_space_bytes=full_space,source_UB_bytes=ub,sort_limit_elements=sort_limit,
      per_core_assignment_rows=intervals[0][1],used_quant_AIV_partitions=len(intervals),column_loops=1,
      core_assignment_intervals=intervals,quant_row_evaluations=R,extra_row_evaluations_from_split_tokens=R-M,
      canonical_all_input_row_reference=M,source_BF16_input_value_bytes=2*H*R,
      alternate_48_assignment_partition_quant_rows=row_evaluations(alt,k),
      alternate_48_assignment_partition_count=len(alt),
      alternate_row_aligned_48_budget=dict(partitions=len(aligned),quant_rows=M,max_token_rows_per_partition=aligned_token_rows,max_assignment_rows_per_partition=aligned_token_rows*k,
        is_current_source_schedule=False,global_protocol_and_numeric_legality_proved=False),
      common_prequantized_q_and_scale_input_value_bytes=R*(H+4),
      same_quantizer_shared_producer_conditional_input_byte_reduction=R*(H-4),
      full_GM_dispatch_record_bytes=H+32,semantic_q_scale_bytes=H+4,
      native_tiling_or_instruction_execution_attested=False,
      scope='Frozen host constants and current descriptor;all input rows retained. Same-quantizer external producer plus new MC2 ABI,matching row generations/masks/lifetimes required. No HBM/gain or necessary-work claim from redundant source loops.')

def main():
    assert not(O/'REVIEW-CHECKPOINT-v95.json').exists()
    src=read(O/'MC2-operator-inputs-v56.json')['operator_source']
    host=Path(next(r['local_path']for r in src if r['path'].endswith('op_host/dispatch_ffn_combine_tiling.cpp')))
    dispatch=Path(next(r['local_path']for r in src if r['path'].endswith('dispatch_ffn_combine_kernel.hpp')))
    spans={host:[(14,24),(46,49),(246,286)],dispatch:[(793,811)],
      BASE/'moe_init_routing_quant_v2_tiling.h':[(16,29),(94,152),(359,427)],
      BASE/'moe_init_routing_v2_tiling.h':[(34,45),(163,180),(202,237)],
      BASE/'moe_init_routing_quant_v2.cpp':[(32,65),(120,135)],
      BASE/'moe_v2_fullload_dynamic_quant.h':[(221,308),(318,341)],
      BASE/'moe_v2_gather_dynamic_quant.h':[(116,152),(158,208)]}
    parts=[]
    for p,ss in spans.items():
        lines=p.read_text().splitlines();parts.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
          segments=[dict(first_line=a,last_line=b,source=lines[a-1:b])for a,b in ss]))
    assert 'constexpr uint32_t BLOCK_NUM = 20;'in host.read_text()
    parent_matches=[]
    for row in src:
        p=W/'frozen-source-snapshot'/row['path'].removeprefix('/vllm-workspace/')
        match=p.exists()and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
        assert match
        parent_matches.append(dict(path=str(p),original166_stage_sha256=row['sha256'],same_parent_source_hash=True))
    write(O/'route-quant-operator-source-contract-v95.json',dict(parts=parts,
      frozen_parent_matches_original166_stage_files=parent_matches,
      facts=[
        'Actual frozen host passes quantMode1,scaleDim0=0(dropPadMode0/activeNum0),UB196352,and2*BLOCK_NUM=40AIV. This is implementation selection,not48AIV physical capacity.',
        'Outer context SetBlockDim uses platform GetCoreNum/CalcTschBlockDim independently of the inner40 routing budget. Recorded24/48 launch descriptors need not equal40;they do not establish a source/binary conflict or an inner48 budget.',
        'Current captured target descriptors use topK6 and draft topK3. Neither a generic topK8 nor24/48core reference substitutes for these source work cases.',
        'Full-load21000 and gathered11000 no-smoothing full-column paths quantize each token once per core/loop interval,then copy the record to its mapped assignments. Core boundaries can split a token and recompute it.',
        'Routing quantizer has BF16→FLOAT,absmax/127,FLOATdivision,FP16CAST_TRUNC,INT8CAST_ROUND plus embeddedFLOATscale. Shared DynamicQuantV2 numerical implementation identity is not established.',
        'Native MC2 input remains BF16; substituting an INT8 pointer is not a valid new prequantized ABI. All route maps/counters/masks/readiness/receiver layouts must remain.',
        'Source quantization loop counts are not necessary lower work,measured inner instruction counts or time. Row-aligned scheduling changes parallelism and may introduce shared GM handoff.'
      ],public_or_native_code_executed=False,new_target_source_files=0,loaded_native_binary_attested=False,overall_complete=False))
    profile=read(O/'MC2-operator-profile-v56.json');models=[]
    for cohort in profile['shape_timings']:
        sh=dimensions(cohort['input_shapes']);M,H=sh[0];k=sh[3][1];localE=sh[1][0]
        assert H==5120 and k in [3,6]and localE in [8,24]
        model=case(M,k,localE*16+1);model['saved_executions']=cohort['executions'];model['saved_input_shapes']=cohort['input_shapes'];models.append(model)
    assert sum(m['saved_executions']for m in models)==4472
    write(O/'route-quant-tiling-and-work-models-v95.json',dict(cases=models,case_count=5,saved_executions=4472,
      source_conditioned_total_quant_row_evaluations=sum(m['quant_row_evaluations']*m['saved_executions']for m in models),
      one_quant_per_input_row_reference=sum(m['M']*m['saved_executions']for m in models),
      source_path_not_native_binary_binding=True,hardware_capacity_not_inferred_from_source_AIV_count=True,overall_complete=False))
    rows=profile['root_bound_graph_layers']+profile['phase_unbound_graph_layers'];bydb=defaultdict(list)
    for row in rows:bydb[row['database_index']].append(row)
    rootkeys={(r['database_index'],r['graph_ordinal'],r['block_ordinal'])for r in profile['root_bound_graph_layers']}
    fronts=[];used=set()
    for db,rs in bydb.items():
        d=read(CACHE/f'analysis/native-execution-identity-v14/db-{db:02d}.json');ts={t['task_record_rowid']:t for t in d['TASK_records']};defs={t['globalTaskId']:t for t in d['COMPUTE_definitions']}
        for r in rs:
            route=ts[r['route_task_rowid']];quant=ts[r['shared_stages'][0]['task_rowid']];rd=defs[route['globalTaskId']];qd=defs[quant['globalTaskId']]
            shapes=dimensions(rd['inputShapes_text']);assert shapes[0]==[3,5120]and shapes[3]==[3,6]
            assert dimensions(qd['inputShapes_text'])==[[3,5120]]and qd['inputDataTypes_text']=='DT_BF16'and qd['outputDataTypes_text']=='INT8;FLOAT'
            assert route['endNs']<=quant['startNs']and route['streamId']==quant['streamId']
            used|={(db,route['task_record_rowid']),(db,quant['task_record_rowid'])}
            fronts.append(dict(database_index=db,graph_ordinal=r['graph_ordinal'],block_ordinal=r['block_ordinal'],
              root_bound=(db,r['graph_ordinal'],r['block_ordinal'])in rootkeys,
              MC2_TASK=route['task_record_rowid'],shared_DQ_TASK=quant['task_record_rowid'],same_recorded_stream=True,
              recorded_MC2_blockNum=rd['blockNum'],recorded_MC2_mixBlockNum=rd['mixBlockNum'],
              native_internal_key_or_same_quantizer_or_pointer_identity_proved=False,
              hypothetical_shared_producer_before_MC2_was_measured=False))
    write(O/'route-quant-native-frontiers-v95.json',dict(frontiers=fronts,native_frontiers=len(fronts),root_bound=sum(r['root_bound']for r in fronts),
      reused_unique_native_TASKs=len(used),no_native_profile_or_regime_promotions=True,
      observation='SharedDQcurrentlysubmittedafterMC2onthesamerecordedstream. MovingproducerbeforeMC2isachangeddependency/resourcegraph,notsimplydeletingalatestandalonekernel.',overall_complete=False))
    # Compare one unchanged graph domain; do not mix sender quant loops with receiver expert GEMM assignments.
    M,H,k,R0=3,5120,6,18;A=M*k
    base=dict(shared_BF16_input_read=2*M*H,shared_q_scale_write=M*(H+4),shared_GateUp_q_scale_read=M*(H+4),
      routing_BF16_input_reads=2*R0*H,routing_expanded_record_writes=A*(H+32))
    common=dict(shared_BF16_input_read=2*M*H,shared_q_scale_write=M*(H+4),shared_GateUp_q_scale_read=M*(H+4),
      routing_common_q_scale_reads=R0*(H+4),routing_expanded_record_writes=A*(H+32))
    dual=dict(shared_BF16_input_read=2*M*H,shared_q_scale_write=M*(H+4),shared_GateUp_q_scale_read=M*(H+4),
      extra_routing_q_scale_write=M*(H+4),routing_common_q_scale_reads=R0*(H+4),routing_expanded_record_writes=A*(H+32))
    write(O/'route-quant-reuse-interface-models-v95.json',dict(current_graph_reference=dict(M=M,H=H,k=k,source_routing_quant_evaluations=R0,shared_quant_input_rows=M,
      baseline_components=base,common_same_quantizer_components=common,dual_preserved_quantizers_components=dual,
      baseline_value_bytes=sum(base.values()),common_value_bytes=sum(common.values()),dual_value_bytes=sum(dual.values()),
      conditional_common_input_byte_reduction=sum(base.values())-sum(common.values()),conditional_dual_input_byte_reduction=sum(base.values())-sum(dual.values())),
      conditional_ceiling=dict(domain='Freshper-assignmentphysicallydistinctuncompressedGMq+FLOATscalerecords,allAvalidassignmentsdemanded,nopreexistingvalidoutputorrepresentationchange.',
        mandatory_expanded_value_write_bytes=A*(H+4),current_padded_record_write_bytes=A*(H+32),
        time_lower_bound='T >= A*(H+4)/B_upper_GM_publication; combine max with other necessary-resource bounds unless a shared-resource sum is proved.',
        B_upper_GM_publication=None,physical_numeric_ceiling_status='SYMBOLIC_CEILING_IN_STATED_REPRESENTATION_DOMAIN',
        not_HBM_or_network_or_all_algorithms_bound=True),
      domain='Nominal all-row/all-assignment graph value model;unchanged shared/GateUp interface,route record and row map. Existing shared q output is the canonical producer only if both quantizers match. Dual-output variant preserves bothquantizers and shares only justified finite-nonzero amax/input work. No HBM or network cut from these GM values.',
      unchanged=['Route-map/sort/counters','H+32record writes and assignment copies','Receiver dispatch/GMM1/SwiGLU/GMM2/combine','Distinct shared/routed expert weights','All padding masks/epochs/aliases/output precision'],
      excluded=['Actual async visibility/record_stream/handoff','Native quantizer and zero/nonfinite/rounding equality','Receiver-side token/expert distribution and wire protocol','Inactive row padding/adapters and capacity overflow'],
      global_sender_and_receiver_counts_not_mixed=True,actual_gain_or_HBM_saving=None,overall_complete=False))
    variants=[
      dict(id='RQ95-CANONICAL-PREQUANT',decision='CONDITIONAL_NEW_ABI',contract='Share existing shared INT8q/FLOATscale with routing only after quantizer identity,all-row masks/generation/readers and new prequantized MC2 contract. New producer precedes MC2 and may change critical path.'),
      dict(id='RQ95-DUAL-QUANT',decision='CONDITIONAL_INPUT_AND_STATISTIC_REUSE',contract='If quantizers differ,produce both outputs with exact distinct arithmetic. Common finite nonzero amax requires compatible reduction semantics;pay extra routing q+scale storage/reads. It does not prove native sharing or remove row statistics.'),
      dict(id='RQ95-ROW-ALIGNED',decision='CONDITIONAL_PARTITION_ALTERNATIVE',contract='Align assignment intervals to full token route groups to eliminate cross-core repeat quantization;may reduce available quant partitions from18to3 forM3K6. Sort/mapping/events/banks/counters still preserved,not a measured improvement.'),
      dict(id='RQ95-EXPERT-SMOOTH',decision='OUTSIDE_CURRENT_HOST_SMOOTH0;SEPARATE_DOMAIN',contract='Perexpert smoothing can change quantized codes andscales even for the same token. Generic row-cache key cannot omit expert/smooth generation. Current hostscaleDim0=0 selects no smooth;other branches not promoted to current.'),
      dict(id='RQ95-QUANT-THEN-DISPATCH',decision='BASELINE_SOURCE_REUSE_RETAINED',contract='Current full/gather no-smooth path already reuses q inside a core interval;do not assume independent per-route quantization everywhere or treat work ratios as time ratios.')]
    write(O/'route-quant-variants-v95.json',dict(variants=variants,new_maincandidate=False,implementation=False,native_test=False,overall_complete=False))
    print(json.dumps(dict(native_frontiers=len(fronts),native_tasks_reused=len(used),source_cases=5,saved_executions=4472,
      source_quant_row_evaluations=sum(m['quant_row_evaluations']*m['saved_executions']for m in models),
      row_once_reference=sum(m['M']*m['saved_executions']for m in models),conditional_M3_common_GM_difference=sum(base.values())-sum(common.values()),overall_complete=False)))

if __name__=='__main__':main()
