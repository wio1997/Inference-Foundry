"""Sparse-index multisets, same-request broadcast descriptors and reuse boundaries."""
from collections import Counter,defaultdict
from copy import deepcopy
from fractions import Fraction
from itertools import combinations,product
from pathlib import Path
import ast,hashlib,json,math,struct
from finalize_origin_coverage_v9 import N,O,R,W,read,write
from reconcile_operator_graph_coverage_v88 import connected_sets


def pin(p):
    p=Path(p);b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def part(p,*ranges):
    lines=Path(p).read_text().splitlines()
    return dict(**pin(p),segments=[dict(first_line=a,last_line=b,source=lines[a-1:b])for a,b in ranges])


def bf16(x):
    bits=struct.unpack('<I',struct.pack('<f',float(x)))[0]
    bits=(bits+0x7fff+((bits>>16)&1))&0xffff0000
    return struct.unpack('<f',struct.pack('<I',bits))[0]


def encode(i):
    return (0x81800000-i)&0xffffffff if i<(1<<24)else i-(1<<23)


def decode(bits):
    return (0x81800000-bits)&0xffffffff if bits&(1<<31)else bits+(1<<23)


def prepare_reference(selected,position,ratio):
    visible=(position+1)//ratio;sent=(1<<31)-1
    valid=[x for x in selected if 0<=x<visible]
    return sorted(valid)+[-1]*(len(selected)-len(valid)),len(valid)


def graph_template(tid,nodes,edges,module):
    return dict(id=tid,module=module,nodes=[dict(id=i,kind=k,resource='SOURCE_OPERATOR_OR_STATE',candidate_associations=c)for i,k,c in nodes],
        edges=[dict(producer=a,consumer=b,value=v,dependency_kind='SOURCE_DATA_CONTROL_OR_PUBLICATION',
            condition='Only the declared logical/noncausal/PA source branch;not all arms coexecute.',source='sparse-index-source-contract-v124.json')for a,b,v in edges],
        complete=False,native_runtime_parameters_or_all_lowerings_attested=False)


def build_graphs():
    templates=[]
    p=['F12'];m=['F31'];both=['F12','F31']
    templates.append(graph_template('PREPARE_VISIBLE_MULTISET',[
        ('selected','SELECTED_INT32_INDEX_SLOTS',p),('positions','QUERY_POSITIONS',m),('empty','ZERO_ROWS_RETURN',p),
        ('load','MASKED_SELECTED_LOAD',p),('visible','POSITION_PLUS_ONE_FLOORDIV_RATIO',both),
        ('valid','GE_ZERO_AND_LT_VISIBLE',p),('sentinel','WHERE_VALID_ELSE_INT32_MAX',p),
        ('encode','INT32_TO_NORMAL_FLOAT_BIT_KEYS',p),('sort','SORT_KEYS_PERMUTATION',p),
        ('decode','FLOAT_BITS_TO_ORIGINAL_INT32_KEYS',p),('negative','SENTINEL_TO_MINUS_ONE',p),
        ('indices_out','GRAPH_STABLE_INDEX_PUBLICATION',both),('length_sum','SUM_VALID_SLOT_OCCURRENCES',both),
        ('length_out','OPTIONAL_STABLE_LENGTH_PUBLICATION',both),('consumer','SMLA_INDEX_AND_LENGTH_INPUT',['F13','F31'])],
        [('selected','empty','T==0'),('selected','load','slots'),('positions','visible','positions'),
         ('load','valid','selected'),('visible','valid','visibility'),('load','sentinel','original IDs'),('valid','sentinel','mask'),
         ('sentinel','encode','multiset'),('encode','sort','bit keys'),('sort','decode','sorted bit keys'),('decode','negative','indices'),
         ('negative','indices_out','chronological valid IDs and -1 suffix'),('valid','length_sum','slot validity,not uniqueness'),
         ('length_sum','length_out','count'),('indices_out','consumer','all current readers'),('length_out','consumer','optional lengths')],'M09'))
    templates.append(graph_template('DSPARK_LOGICAL_WINDOW_BROADCAST',[
        ('qstart','PER_REQUEST_QUERY_STARTS',m),('seq','PER_REQUEST_SEQUENCE_LENGTHS',m),('diff','QUERY_LENGTH_DIFFERENCES',m),
        ('prefix','SEQ_MINUS_QUERY_LENGTH',m),('start','MAX_PREFIX_MINUS_WINDOW_ZERO',m),('visible','SEQ_MINUS_START',m),
        ('columns','ARANGE_FIXED_INDEX_WIDTH',m),('positions','START_PLUS_COLUMNS',m),('mask','COLUMN_LT_VISIBLE',m),
        ('int32','LOGICAL_POSITIONS_INT32',m),('where','INVALID_SUFFIX_MINUS_ONE',m),
        ('repeat_indices','REPEAT_REQUEST_INDEX_ROW_BY_QUERY_LENGTH',m),('repeat_lengths','REPEAT_REQUEST_VISIBLE_LENGTH',m),
        ('stable_indices','PERSISTENT_INDICES_COPY',m),('stable_lengths','PERSISTENT_LENGTH_COPY',m),
        ('consumer','NATIVE_SMLA_PER_QUERY_INDEX_READER',['F13','F31'])],
        [('qstart','diff','adjacent difference'),('seq','prefix','length'),('diff','prefix','query count'),
         ('prefix','start','context'),('seq','visible','length'),('start','visible','start'),('start','positions','first logical position'),
         ('columns','positions','offset'),('columns','mask','column'),('visible','mask','valid limit'),('positions','int32','logical branch'),
         ('int32','where','IDs'),('mask','where','validity'),('where','repeat_indices','one row per request'),('diff','repeat_indices','repeat count'),
         ('visible','repeat_lengths','one count per request'),('diff','repeat_lengths','repeat count'),
         ('repeat_indices','stable_indices','fresh values'),('repeat_lengths','stable_lengths','fresh values'),
         ('stable_indices','consumer','capture-stable bank'),('stable_lengths','consumer','capture-stable bank')],'M21'))
    templates.append(graph_template('SMLA_PA_INDEX_OCCURRENCES',[
        ('offset','SPARSE_LOGICAL_SLOT_OFFSET',['F13']),('blocksize','SPARSE_BLOCK_SIZE',['F13']),
        ('topk','TOPK_INDEX_BANK',['F12','F31']),('which','SLOT_DIV_BLOCKSIZE',['F13']),
        ('read','READ_ONE_INDEX_OCCURRENCE',['F13']),('real','INDEX_TIMES_BLOCKSIZE_PLUS_OFFSET',['F13']),
        ('valid','REAL_POSITION_RANGE_CHECK',['F13']),('pages','REQUEST_BLOCK_TABLE',['F31']),
        ('address','PAGED_ADDRESS_FROM_LOGICAL_BLOCK_AND_OFFSET',['F13','F31']),('gather','VALID_KV_COPY_RETAINING_OCCURRENCE',['F13'])],
        [('offset','which','offset'),('blocksize','which','size'),('topk','read','index slots'),('which','read','selected slot'),
         ('read','real','index'),('blocksize','real','size'),('offset','real','in-block remainder'),('real','valid','position'),
         ('valid','address','valid or -1'),('real','address','block and offset'),('pages','address','request map'),('address','gather','valid address')],'M10'))
    graph=dict(id='SPARSE_INDEX_SOURCE_G9_V124',templates=templates,
        scope='Three selected source-level index producer/publication/consumer branches,not the complete QLI or SMLA lowering.',
        complete=False,overall_complete=False)
    write(O/'operator-graph-G9-v124.json',graph)
    nn=[];ee=[];counts={}
    for t in templates:
        nodes={n['id']:n for n in t['nodes']};sets=connected_sets(nodes,t['edges']);nc=defaultdict(Counter);ec=defaultdict(Counter)
        counts[t['id']]={str(k):len(v)for k,v in sets.items()}
        for k,subsets in sets.items():
            for ss in subsets:
                for node in ss:nc[node][k]+=1
                for i,e in enumerate(t['edges']):
                    if e['producer']in ss and e['consumer']in ss:ec[i][k]+=1
        nn.extend(dict(graph='G9',template=t['id'],node_id=n['id'],kind=n['kind'],module=t['module'],
            connected_set_counts=dict(nc[n['id']]),candidate_source_node_associations=n['candidate_associations'],
            structural_scope='ALL_CONNECTED_SETS_OF2_TO4_WITHIN_SELECTED_SOURCE_TEMPLATE',
            semantic_coverage='SELECTED_SOURCE_CONTRACT_WITH_EXPLICIT_NATIVE_GAPS',native_lowering_and_all_runtime_variants_closed=False)for n in t['nodes'])
        ee.extend(dict(graph='G9',template=t['id'],edge_index=i,edge=e,connected_set_counts=dict(ec[i]),
            producer_candidate_associations=nodes[e['producer']]['candidate_associations'],consumer_candidate_associations=nodes[e['consumer']]['candidate_associations'],
            semantic_coverage='SELECTED_SOURCE_INTERFACE_REVIEWED',dependencies_are_not_all_same_invocation_DAG_edges=True)for i,e in enumerate(t['edges']))
    for name,rows in [('node',nn),('edge',ee)]:
        d=deepcopy(read(O/f'operator-graph-{name}-coverage-v101.json'));d['rows']+=rows
        d.update(parent=f'operator-graph-{name}-coverage-v101.json',row_count=len(d['rows']),overall_complete=False)
        write(O/f'operator-graph-{name}-coverage-v124.json',d)
    bind=deepcopy(read(O/'operator-graph-candidate-bindings-v101.json'))
    rel=read(O/'candidate-relation-clauses-v123.json')['clauses']
    for cid in ['F12','F13','F31']:
        next(c for c in bind['candidates']if c['id']==cid)['sparse_index_source_v124']=dict(templates=[t['id']for t in templates],all_variants_mapped=False)
    hints={tuple(sorted((h['a'],h['b']))):h for h in bind['structural_pair_hints']}
    old_hints=len(hints)
    for row in ee:
        for a in row['producer_candidate_associations']:
            for b in row['consumer_candidate_associations']:
                if a==b:continue
                key=tuple(sorted((a,b)))
                if key not in hints:
                    h=dict(a=key[0],b=key[1],locations=[],existing_conditional_clause_ids=[r['id']for r in rel if {r['a'],r['b']}==set(key)],structure_does_not_prove_same_call_or_joint_legality=True)
                    bind['structural_pair_hints'].append(h);hints[key]=h
                hints[key]['locations'].append(dict(graph='G9',template=row['template'],edge_index=row['edge_index'],kind='SELECTED_SOURCE_INDEX_PUBLICATION_OR_CONSUMER'))
    bind.update(parent='operator-graph-candidate-bindings-v101.json',new_pair_hints_added=len(hints)-old_hints,overall_complete=False)
    write(O/'operator-graph-candidate-bindings-v124.json',bind)
    prior=read(O/'operator-graph-coverage-audit-v101.json');catalog=prior['graph_catalog']|{'G9':str(O/'operator-graph-G9-v124.json')}
    audit=dict(parent='operator-graph-coverage-audit-v101.json',graph_catalog=catalog,
        graph_sha256={k:pin(v)['sha256']for k,v in catalog.items()},new_source_node_rows=len(nn),new_source_edge_rows=len(ee),
        node_rows=prior['node_rows']+len(nn),edge_rows=prior['edge_rows']+len(ee),current_structural_pair_hints=len(hints),
        G9_templates=len(templates),G9_connected_sets=counts,source_rows_not_unique_native_operators_or_new_executions=True,whole_modules_closed=0,overall_complete=False)
    write(O/'operator-graph-coverage-audit-v124.json',audit)
    return audit


def main():
    assert not(O/'REVIEW-CHECKPOINT-v124.json').exists()
    frozen=W/'frozen-source-snapshot/vllm-ascend'
    src=N/'attention-operator-source-v60/vllm-workspace/vllm-ascend'
    prep=src/'vllm_ascend/ops/triton/prepare_indexer_indices.py'
    vector=src/'csrc/attention/quant_lightning_indexer_v2/op_kernel/arch22/quant_lightning_indexer_v2_service_vector_arch22.h'
    gather=src/'csrc/attention/sparse_flash_mla/op_kernel/arch22/sparse_flash_mla_csa_block_vector.h'
    indexer=frozen/'vllm_ascend/models/deepseek_v41/indexer.py'
    dsa=frozen/'vllm_ascend/attention/dsa_v41.py';dsa1=frozen/'vllm_ascend/attention/dsa_v1.py';binding=frozen/'csrc/torch_binding.cpp'
    docs=[O/'selection-attention-native-frontiers-v103.json',O/'attention-indexer-work-traffic-models-v60.json',
        O/'attention-indexer-fusion-reorganization-variants-v60.json',O/'attention-headwise-merge-and-linear-state-v100.json',
        O/'operator-graph-coverage-audit-v101.json',O/'operator-graph-node-coverage-v101.json',
        O/'operator-graph-edge-coverage-v101.json',O/'operator-graph-candidate-bindings-v101.json']
    write(O/'sparse-index-inputs-v124.json',dict(inputs=[pin(p)for p in docs+[prep,vector,gather,indexer,dsa,dsa1,binding]],
        operator_questions=['Does prepare remove duplicate valid indices or count occurrences?',
            'What producer/range/offset conditions support a valid-index uniqueness certificate?',
            'When do identical per-request DSpark windows permit descriptor or KV-load reuse without reusing query-dependent outputs?'],
        no_new_remote_device_Torch_or_model_work=True,overall_complete=False))
    tree=ast.parse(indexer.read_text());fn=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='select_projected')
    call=next(n for n in ast.walk(fn)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='npu_quant_lightning_indexer_v3')
    assert 'output_idx_offset'not in [k.arg for k in call.keywords]
    common=next(n.value for n in ast.walk(fn)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='common'for t in n.targets))
    assert isinstance(common,ast.Call)and 'output_idx_offset'not in [k.arg for k in common.keywords]
    write(O/'sparse-index-source-contract-v124.json',dict(parts=[part(prep,(35,56),(68,78)),
        part(vector,(240,246),(777,804),(860,888),(890,914),(1008,1099),(1105,1129)),
        part(indexer,(181,204),(252,300)),part(binding,(3158,3169)),part(dsa1,(428,460),(481,511)),
        part(dsa,(1069,1096)),part(gather,(524,555),(573,632))],
        prepare='Filter each slot independently to INT32_MAX,sort by reversible FLOAT bit keys,decode to original ID or-1. Length is number of valid slots,not number of distinct IDs. No deduplication.',
        key_domain='Normal source domain:nonoverflowing position+1 and0<=visible<=INT32_MAX;valid IDs0..visible-1. Negative selected values and out-of-range IDs become invalid. Sentinel cannot count as valid in this domain.',
        local_QLI_uniqueness='Within one source S2 tile,ArithProgression lanes plus cuBaseS2Idx give distinct valid IDs. Global valid-ID uniqueness additionally needs disjoint per-query source tile/LD partitions,record-preserving sort/merge and consistent output offsets;not established merely by TopK shape.',
        offset='Pinned V41 ordinary caller omits output_idx_offset;schema default None. Generic nonzero offset branches add to all extracted slots including -1,so a broad uniqueness/invalid-sentinel claim must retain that branch or prove its domain. No such nonzero offset is asserted active here.',
        dspark='In selected use_logical_indices=True noncausal source domain,each request builds a contiguous valid logical interval then repeats exactly that row and length for all query rows of the request. Persistent bank copies are current behavior.',
        consumer='SMLA expands each sparse index occurrence and range-checks the resulting logical position;the selected PA path maps it through the request block table. Repeated valid IDs are not made a set by this code;invalid -1 expands only negative positions for positive sparseBlockSize.',
        pair_gather_order='Selected CopyInKv compares physical offsets,uses absolute stride/minimum starting address when paired copying is legal,and falls back to two CopyInSingleKv calls for negative stride/overlap or range limits. A changed gather/cache must preserve the actual emitted-row/numeric order contract,not assume logical chronological index order alone describes every physical pair.',
        physical_pointer_identity_is_not_logical_key_or_epoch_identity=True,
        historical_native_ID_values_or_duplicate_rates_attested=False,whole_source_scheduler_partition_proved=False,overall_complete=False))
    # Exhaustive bounded multiset reference;not native execution or a whole-int32 exhaustive proof.
    count=duplicates=0
    for selected in product([-1,0,1,2,5],repeat=4):
        for position in [-1,0,1,2,5,8]:
            for ratio in [1,2]:
                out,length=prepare_reference(selected,position,ratio)
                expected=Counter(x for x in selected if 0<=x<(position+1)//ratio)
                assert Counter(x for x in out if x>=0)==expected and length==sum(expected.values())
                duplicates+=any(v>1 for v in expected.values());count+=1
    probes=[0,1,(1<<23)-1,1<<23,(1<<24)-1,1<<24,(1<<24)+1,(1<<31)-2,(1<<31)-1]
    floats=[struct.unpack('<f',struct.pack('<I',encode(x)))[0]for x in probes]
    assert all(math.isfinite(v)and v!=0 for v in floats)and floats==sorted(floats)
    assert all(decode(encode(x))==x for x in probes)
    w=Fraction(257,512);repeated=Fraction(3,2)/(1+3*w);folded=(3*w)/(1+3*w)
    write(O/'sparse-index-multiset-and-rounding-v124.json',dict(
        reference_cases=count,reference_cases_with_valid_duplicates=duplicates,
        prepare_duplicate_witness=dict(selected=[1,1,-1,5],position=3,ratio=1,prepared=[1,1,-1,-1],valid_length=2,unique_valid_count=1),
        offset_sentinel_witness=dict(raw=[0,-1,-1],offset=2,unmasked_offset_result=[2,1,1],visible=4,prepared=[1,1,2],scope='Generic optional offset source branch only;not the selected V41 call or a measured bug.'),
        exact_real_coalescing='For each fixed query/head,only if all occurrences in group g have identical effective score and value under mask/bias/position/bank/epoch semantics,keep count c_g:denominator=exp(sink-m)+sum_g(c_g*exp(score_g-m));numerator=sum_g(c_g*exp(score_g-m)*v_g). Sink occurs once. Dropping multiplicity changes the function.',
        sink_counterexample=dict(duplicate_key_score=0,duplicate_value=1,multiplicity=2,sink_score=0,original_output='2/3',wrong_set_output='1/2'),
        rounding_counterexample=dict(exact_positive_exponential=str(w),BF16_exponential=bf16(w),multiplicity=3,
            three_rounded_exponentials=3*bf16(w),rounded_aggregate_exponential=bf16(3*w),
            background_or_sink_exponential=1,source_style_output=str(repeated),real_log_count_fold_output=str(folded),
            BF16_final_source_style=bf16(repeated),BF16_final_fold=bf16(folded),
            scope='Finite declared reference arithmetic with source-style BF16 exponential/PV boundary. It disproves a general real-identity-implies-native-bit-identity rule;not actual native exp/log/accumulation measurement.'),
        across_query_counterexample=dict(same_indices=[0,1],keys=[0,1],values=[0,1],q0=0,q1='log(3)',real_outputs=['1/2','3/4'],scope='No sink example;equal index rows alone do not imply equal attention outputs.'),
        key_encoding_checks=dict(boundary_IDs=probes,normal_finite_nonzero_ordered=True,bijection_in_declared_piecewise_int32_domain=True),
        native_bitwise_duplicate_collapse_proved=False,overall_complete=False))
    costs=[]
    for B in [1,4,8]:
        for q in [2,3,5]:
            I=256;T=B*q;expanded=4*T*I+4*T;row=4*B*I+4*B+4*T;interval=8*B+4*T
            costs.append(dict(requests=B,query_rows_per_request=q,query_rows=T,index_width=I,
                expanded_current_index_and_length_content_bytes=expanded,
                selected_per_request_row_plus_query_owner_content_bytes=row,
                selected_interval_plus_query_owner_content_bytes=interval,
                per_request_row_reference_delta=expanded-row,interval_reference_delta=expanded-interval,
                native_observers_or_new_descriptor_ABI_bound=False,physical_HBM_or_allocation_credit=False))
    write(O/'sparse-index-request-reuse-and-cost-v124.json',dict(models=costs,
        per_request_interval='start=max(seq_len-query_len-window,0),visible=seq_len-start;all query rows of that request share [start,start+visible) in the selected noncausal logical branch.',
        no_output_reuse='Query values/position transforms,sink/scale and all mask/cache generations still determine every query output. Identical index rows support a descriptor and possible KV-load reuse question,not automatic QK/PV/softmax/output reuse.',
        descriptor_variants='Selected new ABI references:one explicit index row+length/request and INT32 owner/query,or INT32(start,length)/request+INT32 owner/query. Existing query_start_loc could derive ownership only with its own cost/ABI proof. Allocation/alignment/epochs/other metadata are additional.',
        KV_reuse='For one unchanged request-bank epoch and identical visible set of n vectors,logical set union is n across q query rows rather than q*n. v60 already states union-sensitive ideal IO;this is its new source-qualified same-request specialization,not a second independent IO saving.',
        baseline_unknown='Current native SMLA may already reuse KV via tiling or caches. No per-query gather-count or physical-traffic gain is assigned without its loaded tiling/cache evidence.',
        padded_invalid='Source selected gather already rejects negative/out-of-range positions. Repeated -1 descriptors are not repeated valid-key payloads;do not bill their removal as free valid-key QK/PV deletion.',
        memory_and_lifetime='Descriptor/KV cache keys must include request/sequence logical namespace,bank type,head/layout/compression and version,block-table/page-owner generation and query-mask generation. Keep capture bank addresses,all old readers/queued writers and stable length/index publication coherent.',
        theory_state='SYMBOLIC_CEILING',numeric_T_lower_or_throughput_upper=None,actual_speedup_or_MFU=False,overall_complete=False))
    front=read(O/'selection-attention-native-frontiers-v103.json')
    write(O/'sparse-index-profile-scope-v124.json',dict(reused_QLI_prepare_SMLA_frontiers=front['count'],root_bound=front['root_bound'],phase_unbound=front['phase_unbound'],
        reused_unique_TASKs=front['reused_unique_TASKs'],modes=front['modes'],families=front['families'],
        scope='Existing ordered profile frontiers verify the producer/prepare/consumer boundary. They contain no selected ID payload,duplicate count,per-query gather count or counterfactual readiness.',
        DSpark_noncausal_descriptor_scenario_not_attested_by_QLI_chain=True,new_regime_promotions=0,overall_complete=False))
    audit=build_graphs()
    print(json.dumps(dict(status='ANALYZED',prepare_multiset_cases=count,descriptor_models=len(costs),
        source_graph_node_rows=audit['new_source_node_rows'],source_graph_edge_rows=audit['new_source_edge_rows'],
        partial_graph_rows=[audit['node_rows'],audit['edge_rows']],structural_hints=audit['current_structural_pair_hints'],new_device_work=False,overall_complete=False)))


if __name__=='__main__':main()
