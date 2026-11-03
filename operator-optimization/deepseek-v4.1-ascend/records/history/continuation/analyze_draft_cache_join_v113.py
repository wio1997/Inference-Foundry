"""Analyze DSpark context/query cache WAW cuts and backwards row demand; CPU only."""
from pathlib import Path
from collections import Counter
from itertools import permutations
from fractions import Fraction
import hashlib,json
from finalize_origin_coverage_v9 import N,O,W,read,write

ROOT=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend'
CACHE=N/'original-profile-cache-v53/analysis/native-execution-identity-v14'

def pin(p):
    b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def part(p,a,b):return dict(**pin(p),first_line=a,last_line=b,source=p.read_text().splitlines()[a-1:b])
def owner_local(p,d,i):return (p%(d*i))//i,(p//(d*i))*i+p%i
def inverse_local(local,rank,d,i):return (local//i)*(d*i)+rank*i+local%i
def slots(positions,rank,d,i,block_size,table):
    out=[]
    for p in positions:
        owner,local=owner_local(p,d,i)
        out.append(table[local//block_size]*block_size+local%block_size if owner==rank else -1)
    return out

def validate_maps():
    cases=0;colliding=0;max_overlap=0;digest=hashlib.sha256()
    for n in range(1,9):
        for r in range(n+1):
            for q in range(1,7):
                for prefix in [0,7]:
                    seq=prefix+n;c=list(range(prefix,seq));z=list(range(seq-r,seq-r+q))
                    expected=set(range(seq-r,min(seq,seq-r+q)))
                    assert set(c)&set(z)==expected and len(expected)==min(r,q)
                    for d in [1,2,3,4]:
                        for interleave in [1,2,4]:
                            for block in [1,2,4,8]:
                                # A non-identity injective physical-page permutation.
                                table=[1000+17*k for k in range(64)];total=0
                                for rank in range(d):
                                    cs={v for v in slots(c,rank,d,interleave,block,table)if v>=0}
                                    qs={v for v in slots(z,rank,d,interleave,block,table)if v>=0}
                                    es={v for v in slots(sorted(expected),rank,d,interleave,block,table)if v>=0}
                                    assert cs&qs==es;total+=len(es)
                                    for p in c+z:
                                        own,local=owner_local(p,d,interleave)
                                        assert inverse_local(local,own,d,interleave)==p
                                assert total==min(r,q)
                                cases+=1;colliding+=bool(total);max_overlap=max(max_overlap,total)
                                digest.update(f'{n},{r},{q},{prefix},{d},{interleave},{block}:{total}\n'.encode())
    return dict(cases=cases,colliding_cases=colliding,max_overlap=max_overlap,fixture_digest=digest.hexdigest(),
        domain='CPU finite reference:n1..8,r0..n,q1..6,prefix0/7,DCP1..4,interleave1/2/4,block1/2/4/8,injective pages;not native runtime parity.')

def main():
    assert not(O/'draft-cache-join-inputs-v113.json').exists()
    paths=[O/'REVIEW-CHECKPOINT-v112.json',O/'coverage-current-obligations-v112.json',O/'candidate-relation-clauses-v112.json',
        O/'candidate-pair-navigation-v112.json',O/'M19-layout-ownership-and-context-query-reorganization-v35.json',
        O/'M19-current-forward-patterns-and-dispatch-v35.json',O/'M18-current-aux-context-and-publication-profile-v34.json',
        O/'index-context-native-chains-v64.json',O/'index-context-operator-contract-v64.json']
    ds=ROOT/'models/deepseek_v41/dspark.py';sp=ROOT/'spec_decode/dspark_proposer.py'
    df=ROOT/'spec_decode/dflash_proposer.py';ll=ROOT/'spec_decode/llm_base_proposer.py'
    ut=ROOT/'ops/triton/spec_decode/utils.py';at=ROOT/'attention/dsa_v41.py'
    parts=[part(ds,223,309),part(sp,31,44),part(sp,247,311),part(sp,313,355),part(df,301,319),
        part(ll,1047,1083),part(ll,1475,1495),part(ut,69,206),part(at,296,308),part(at,337,389),part(at,690,717)]
    paths+=list(dict.fromkeys(Path(p['path'])for p in parts));front=[];dbpaths=[]
    old=read(O/'index-context-native-chains-v64.json')
    for db in range(32):
        p=CACHE/f'db-{db:02d}.json';dbpaths.append(p);data=read(p)
        defs={d['globalTaskId']:d for d in data['COMPUTE_definitions']}
        ts={t['task_record_rowid']:t for t in data['TASK_records']}
        tiles=[]
        for t in ts.values():
            de=defs.get(t['globalTaskId'],{})
            if de.get('opType_text')=='Tile'and de.get('outputShapes_text')in ['"10,4,5120"','"20,4,5120"']:
                tiles.append(t)
        tiles.sort(key=lambda t:t['startNs'])
        contexts=sorted([r for r in old['context_KV_source_order_cycles']if r['database_index']==db],key=lambda r:r['stages'][0]['startNs'])
        mains=[r for r in old['main_projection_source_order_chains']if r['database_index']==db]
        assert len(contexts)==3*len(tiles)
        for k,t in enumerate(tiles):
            cs=contexts[3*k:3*k+3];assert len({c['rows']for c in cs})==1 and len({c['source_context']for c in cs})==1
            for c in cs:
                for stage in c['stages']:
                    saved=ts[stage['TASK_rowid']]
                    assert(saved['startNs'],saved['endNs'],saved['streamId'])==(stage['startNs'],stage['endNs'],stage['streamId'])
            assert all(cs[j]['stages'][-1]['endNs']<=cs[j+1]['stages'][0]['startNs']for j in range(2))
            assert cs[-1]['stages'][-1]['endNs']<=t['startNs']
            previous_end=tiles[k-1]['endNs']if k else -1
            mm=[m for m in mains if previous_end<m['full5120_norm']['endNs']<=cs[0]['stages'][0]['startNs']]
            assert len(mm)<=1
            de=defs[t['globalTaskId']]
            front.append(dict(database_index=db,source_order_ordinal=k,context_rows=cs[0]['rows'],query_global_rows=int(de['outputShapes_text'].strip('"').split(',')[0]),
                source_context=cs[0]['source_context'],context_cycles=[dict(first_TASK=c['stages'][0]['TASK_rowid'],last_TASK=c['stages'][-1]['TASK_rowid'],startNs=c['stages'][0]['startNs'],endNs=c['stages'][-1]['endNs'])for c in cs],
                draft_repeat=dict(TASK_rowid=t['task_record_rowid'],streamId=t['streamId'],startNs=t['startNs'],endNs=t['endNs']),
                prior_main=mm[0]if mm else None,
                grade='EXACT_SAVED_TASK_KEYS_AND_TYPED_SOURCE_ORDER_SCHEMA;NOT_SAME_REQUEST_BANK_OR_POINTER_PROOF',
                all_context_before_repeat_observed=True,physical_alias_or_counterfactual_critical_path_proved=False))
    assert len(front)==104 and sum(f['prior_main']is not None for f in front)==88
    paths+=dbpaths
    write(O/'draft-cache-join-inputs-v113.json',dict(inputs=[pin(p)for p in paths],
        source_question='Raw saved profile lacks slot operands/accepted counts. Pin only copy/queryslot/cachewriter and proposed backwards-demand boundaries.',
        prior_v34_v35_pipeline_and_rowpack_are_existing_answers=True,new_remote_source_or_device_work=False,overall_complete=False))
    write(O/'draft-cache-join-source-contract-v113.json',dict(parts=parts,facts=[
        'The merged proposer currently combines target aux states before set_inputs_first_pass,then precomputes all context-layer KV before DSpark embedding/decoder. The source control order is not a proof of necessary data dependence.',
        'set_inputs_first_pass copies the complete context prefix into its bank. The metadata kernel copies every incoming context slot unchanged;rejection modifies query effective_seq_len,not that context copy.',
        'For requestj,query logical cache positions are seq_len_j-r_j+q_idx;RoPE positions instead derive from target_positions at the last valid context row. They are distinct domains.',
        'A DCP owner/local compact mapping is applied before physical block lookup. Rejected context rows may map to exactly the same physical slots as early query rows,including on cache-only/no-localQ ranks.',
        'Context precompute and query cache writers select the same layer swa_cache_layer.kv_cache[0] expression. Loaded object/actualbank/pointer identity remains a runtime gate.',
        'At construction sample_from_anchor makes num_query_per_req=configuredK rather than1+configuredK;the alternate branch uses1+configuredK. The proof uses actual q,not a universal current-scheduled-K+1 assumption.',
        'v35 already proposed per-layer lazy preparation and same-WKV row packing with context-before-query write order. New work here is exact rejection-induced conflict/demand algebra and its main-projection backwards slice.'],
        current_loaded_binary_guards_or_actual_slot_values_bound=False,overall_complete=False))
    write(O/'draft-cache-join-native-frontiers-v113.json',dict(frontiers=front,source_order_frontiers=104,context_cycles_reused=312,
        preceding_main_chains=88,contexts=dict(Counter(f['source_context']for f in front)),
        original_TASK_values_rechecked=True,new_functional_or_phase_promotions=0,
        timing_scope='Observed existing serial schema;not causal recoverable latency or all streams dependence. No slots/r_j/allreader/allocator epoch data is present.',overall_complete=False))
    validation=validate_maps()
    formula=dict(status='CONDITIONAL_CEILING',normal_domain=[
        'For each request n_j>=1 current context rows occupy contiguous logical cache interval[s_j-n_j,s_j).',
        '0<=r_j<=n_j;query countq_j>=1;correct shared per-group slot mapping/CP owner transform is applied to both sides.',
        'Valid injective page maps within each request andexclusive ownership or resolved COW across requests;negative/masked slots excluded.',
        'Source current context slots really correspond to these logical rows and loaded cachebank/writer format agrees. No such runtime operand assertion is made.'],
        C_j='[s_j-n_j,s_j)',Q_j='[s_j-r_j,s_j-r_j+q_j)',
        intersection='[s_j-r_j,min(s_j,s_j-r_j+q_j))',O_j='min(r_j,q_j)',
        physical_per_rank='Apply owner/local/page map;sum across the DCP owner ranks recovers O_j. Do not multiply a per-owner row count by all ranks.',
        arbitrary_actual_maps='O_l=unique valid physical slot intersection(C_l,Q_l);use actual group/bank/byteextent. Formula min(r,q) need not survive invalid slots,noninjective aliases or mismatched table generations.',
        request_sum='O=sum_j min(r_j,q_j) only under cross-request disjointness;otherwiseusephysicalsets and writer/read order.',
        DCP_inverse='p=(local//I)*(D*I)+rank*I+local%I for an owned local slot,so logical compact map is injective.',
        no_reuse_of_context_KV_as_query_KV='Even at identical slot/RoPEposition,main_norm context and current query layernorm have different hidden provenance;same W does not imply same value.',
        runtime_overlap_count=None,overall_complete=False)
    write(O/'draft-cache-slot-conflict-algebra-v113.json',formula)
    schedules=[]
    for order in permutations(['Wc','Wq','Read']):
        if order.index('Wq')>order.index('Read'):continue
        cache='old';observed=None
        for op in order:
            if op=='Wc':cache='context'
            elif op=='Wq':cache='query'
            else:observed=cache
        schedules.append(dict(order=list(order),observed=observed,final=cache,source_equivalent=observed==cache=='query'))
    witnesses=[dict(id='REJECTED_TAIL_OVERLAP',n=3,seq_len=10,rejected=2,q=5,block_size=4,
        context_logical=[7,8,9],query_logical=[8,9,10,11,12],overlap=[8,9],
        context_values=['C7','C8','C9'],query_values=['Q8','Q9','Q10','Q11','Q12'],
        consequence='Unordered lateC8/C9 can overwrite Q8/Q9. No input or weight value equality is available.'),
        dict(id='CROSS_REQUEST_ALIAS_WITH_R_ZERO',requestA_context_slot=20,requestB_query_slot=20,rejected=[0,0],
            consequence='Per-request logical nonoverlap alone doesnotprove physical disjointness across request/page epochs.'),
        dict(id='ROPE_IS_NOT_LINEAR_CACHE_INDEX',context_RoPE_positions=[100,100],seq_len=20,rejected=1,
            first_query_RoPE_position=101,first_query_linear_slot_position=19,
            consequence='Building the conflict predicate from RoPEposition instead of linear slots can be wrong.'),
        dict(id='OVERWRITE_IS_NOT_ENOUGH_TO_DELETE',order=['Wc','ExtraRead','Wq'],extra_reader_needs='context',
            consequence='Finalquerywinner doesnotmake Wc dead when an intervening reader orerror/state observer exists.'),
        dict(id='EMPTY_LOCAL_QUERY_STILL_WRITES',local_query_rows=0,global_query_KV_rows=10,
            consequence='Current savedcache-only rankcases stillpublishglobalKV;noQ isnot noWq/nohazard.')]
    write(O/'draft-cache-order-and-witnesses-v113.json',dict(schedules=schedules,witnesses=witnesses,CPU_reference=validation,
        rule='For conflicting materialized bytes preserve Wc→Wq and required RAW to readers. Nonconflicting stores need no invented whole-batch barrier;new scheduling/publication protocol must be priced.',
        no_native_test_or_unconditional_bug_claim=True,overall_complete=False))
    write(O/'draft-cache-demand-and-readiness-v113.json',dict(
        current_graph=['aux_selected→MainGEMM→full5120RMS→contextbank','metadata→context_slots/query_slots/IDs/positions',
            'contextbank→ContextKV_l→Wc_l','IDs→embedding/HC/querynorm→QueryKV_l,Q_l','Wq_l→Attention_l','Attention_l→later_decoder_l+1'],
        split='Separate the independent metadata/ID production from the current method context-bank copy;actual output,mutation,error,seed,stream andcapture epochs remain explicit. Do not move the whole mutating method blindly.',
        required_edges=['Wc_l(s)→Wq_l(s) on conflicting bytes if retaining both writes',
            'Last required cache writer→each attention reader ofthat slot','all required first-layerKVreadiness→that attention tile,not necessarily every later-layer context write',
            'layerlqueryoutput→nextlayerqueryinput;layer-specific WKV weights/norms andpositions remain distinct'],
        work_elision='If Wc_l(s) has no observer before a guaranteed full Wq_l(s) overwrite,its private producer can be removed only after all other producer readers are excluded. Keep required final cache bytes andpreserve allnumeric boundaries.',
        context_demand='D_l = context source rows with any remaining demanded KVoutput/cachewrite orotherobserver atlayerl.',
        main_demand='D_main = union_l D_l union D_other_contextbank_readers. Only the complement ofthis union is eligible for removing context main projection/full5120norm/bankwork.',
        cross_layer_rule='A row overwritten atone layer is not enough to prune F27. Differentgroups/slots/weights/masks/readers can keep it live elsewhere. Physical masks must map back to original ordered context rows.',
        no_full_layer_barrier_or_free_overlap_assumed=True,backend_forced_collectives_not_deleted=True,
        native_no_context_dummy_andzeroQ_cases_remain_distinct=True,
        observers_andepochs=['intermediateKV/cache readers','persistentcontextbank prefix/tail/capturedaddresses','query/context groups andCOWpagegenerations',
            'duplicate-slot sourceorder','nextiteration queuedreaders/writers/retirement','normalmetadata/errors/RNG/seedstate','numericalcontraction/rounding andallTP/CPparticipation'],
        actual_allreader_or_epoch_closure_proved=False,overall_complete=False))
    examples=[]
    for ns,rs,qs in [([6,6],[1,2],[5,5]),([6]*4,[0,1,2,5],[5]*4)]:
        nc=sum(ns);nq=sum(qs);overlap=sum(min(r,q)for r,q in zip(rs,qs));union=nc+nq-overlap
        examples.append(dict(context_rows=nc,query_rows=nq,rejected=rs,query_per_request=qs,conditional_overwrite_rows=overlap,
            source_context_plus_query_cache_row_write_bytes=(nc+nq)*1024,conditional_final_union_row_write_bytes=union*1024,
            conditional_write_difference_per_layer_bytes=overlap*1024,
            full_ordinary_WKV_reference_OPS=2*(nc+nq)*5120*512,
            after_private_dead_context_row_pruning_WKV_OPS=2*union*5120*512,
            conditional_WKV_OPS_difference_per_layer=2*overlap*5120*512,
            if_same_dead_rows_atall3layers_andno_other_main_readers_main_OPS_difference=2*overlap*15360*5120,
            actual_profile_rejected_counts=False))
    write(O/'draft-cache-joint-work-and-ceiling-v113.json',dict(status='SYMBOLIC_CEILING',examples=examples,
        declared_dense_domain='Full ordinary named products,arbitrary row values,no precomputed outputs,numericstage equivalence. Not an allalgorithms/GEMM lower bound.',
        full_work='Q_WKV=2*H*D*sum_l(|D_l|+q_l),withH5120,D512. Q_main=2*15360*5120*|D_main|. Fullrowmain5120 andKV512RMS remain independent numeric boundaries.',
        output_work='Under an explicit fullrow final-cache ABI,arbitrary values differingfrominitial,andno compression/implicit representation,distinctfinal demanded slots require1024B each. This is a cache-interface value-store bound,not HBMtraffic.',
        time='T>=max(Q_BF16/(N_AIC*R_BF16*F_AIC),B_store/U_same_cache_interface,othernecessarycuts) onlywithvalidscope andcapacityupperbounds. Preserve sharedresources and all newmask/map/compaction/publication costs.',
        representation_costs=['conflictmask andinverse source-row map','sparse/dense row packing andextra launch/publication','complete same-WKV/norm numericcontract',
            'lostinputreuse orlater queryreadydelay','remaining contextbank/collective costs','actualcriticalpath/enginecompetition'],
        no_duplicate_credit='F27deadrow,contextKVdeadrow,cachewrite elision andv35rowpacking are one selectedcomposedgraph. Do not add independent intermediate oroverlap credits.',
        actual_MFU_physical_numeric_floor_or_speedup_proved=False,overall_complete=False))
    evidence=['draft-cache-join-source-contract-v113.json','draft-cache-join-native-frontiers-v113.json','draft-cache-slot-conflict-algebra-v113.json',
        'draft-cache-order-and-witnesses-v113.json','draft-cache-demand-and-readiness-v113.json','draft-cache-joint-work-and-ceiling-v113.json']
    write(O/'draft-cache-composition-v113.json',dict(clause=dict(id='REL113-01',a='F27',b='F28',
        relation='QUERY_CACHE_FINAL_WRITES_BACKPROPAGATE_CONTEXT_MAIN_ROW_DEMAND',
        operator_claim='Split metadata/ID readiness fromcontextbankcopy;querycacheoverwrite may remove private contextKVrows andonly all-layer/all-reader deadrows fromF27 mainprojection/norm. Preserve typed WAW/RAWcachepublication.',
        applicable_domain='Loadedmatching standardcache ABI,normalvalid row/slot metadata,actualsamebank/fullbyteoverwrite,allintermediate/future readers/epochs and numeric/allrank protocol. Rejectionformula needsinjectiveconsistentpages and actualqbranch.',
        evidence=evidence,conditional_compatibility=True,all_runtime_gates_proved=False,speedup_additive=False,critical_path_gain_us=None),
        refine_group='HG85-VERIFY-CONTEXT-DRAFT',refine_existing_F29_v35_context_query=True,
        old_rowpack_andlazy_perlayer_opportunities_not_new=True,new_pairs=1,new_groups=0,new_main_candidates=0,
        whole_pair_all_variants_closed=False,overall_complete=False))
    print(json.dumps(dict(status='ANALYZED',native_source_order_frontiers=len(front),CPU_mapping_cases=validation['cases'],new_conditional_pair=1,
        original_regime_promotions=0,new_device_work=False,overall_complete=False)))

if __name__=='__main__':main()
