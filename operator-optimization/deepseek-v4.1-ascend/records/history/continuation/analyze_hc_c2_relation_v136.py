"""HC/C2 composition with original logical pairs and future ring-state duties."""
from pathlib import Path
from collections import Counter
from itertools import product
import hashlib,json
from finalize_origin_coverage_v9 import N,O,W,read,write


def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def original(s,n,base,valid):
    if not valid:return [],[]
    pairs=[]
    for q in range(s//2,(s+n)//2):
        left=2*q;right=left+1
        pairs.append(dict(q=q,left_kind='CURRENT'if left>=s else'OLD_RING',left_position=left,
            left_row=base+left-s if left>=s else None,right_row=base+right-s,out_row=base+right-s,
            branch='NONRESIDUAL'if left>=s else'RESIDUAL',old_slot=left%32 if left<s else None))
    tail=[(base+j,s+j,(s+j)%32)for j in range(max(0,n-32),n)]
    return pairs,tail


def pair_case(s,n,cut,base,valid):
    ps,tail=original(s,n,base,valid);a,at=original(s,cut,base,valid);b,bt=original(s+cut,n-cut,base+cut,valid)
    naive={p['out_row']:p for p in a+b};mismatch=[p['out_row']for p in ps if naive[p['out_row']]!=p]
    X=sum(p['left_kind']=='CURRENT'and (p['left_row']<base+cut)!=(p['right_row']<base+cut)for p in ps)
    return dict(s=s,n=n,cut=cut,base=base,valid=valid,E=len(ps),B=sum(p['left_kind']=='OLD_RING'for p in ps),U=len(tail),
        cross_owner_current_pair_halos=X,naive_changed_current_vs_old_pairs=len(mismatch),
        naive_old_reads=sum(p['left_kind']=='OLD_RING'for p in a+b),
        naive_tail_extra_writes=len(at)+len(bt)-len(tail),original_completion_rows=[p['out_row']for p in ps],
        metadata_and_owner_example_only=True)


def timing(r):
    # Independent resources and one ordered message domain are explicit assumptions.
    s,k,h,c,p,t,cp,ct,cj,u=[r[x]for x in ['s','k','h','c','p','t','cp','ct','cj','u']]
    pool=max(k,h)+p;packet=max(s,pool);snapshot=k+t
    ja=max(packet,snapshot)+cj;pa=packet+cp;ta=max(snapshot,pa)+ct
    return dict(pool_ready=pool,joint_current=max(c,max(ja+4,2)+2),split_current=max(c,max(pa+4,2)+2),
        joint_ring=max(pool,ja)+u,split_ring=max(pool,ta)+u,
        joint_current_and_future=max(max(c,max(ja+4,2)+2),max(pool,ja)+u),
        split_current_and_future=max(max(c,max(pa+4,2)+2),max(pool,ta)+u))


def main():
    assert not(O/'REVIEW-CHECKPOINT-v136.json').exists()
    prof=read(O/'hc-c2-native-consumers-v136.json');root=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend';parts=[]
    for rel,a,b in [('models/deepseek_v41/model.py',887,920),('models/deepseek_v41/compressor.py',16,24),('models/deepseek_v41/compressor.py',55,67),
        ('attention/dsa_v41.py',523,565),('models/deepseek_v41/indexer.py',106,125)]:
        q=root/rel;parts.append(dict(**pin(q),first_line=a,last_line=b,source=q.read_text().splitlines()[a-1:b]))
    q=N/'state-engram-operator-source-v62/vllm_ascend/ops/triton/compressor/compressor_triton.py'
    for a,b in [(85,193),(225,265),(266,320),(324,364),(647,721)]:parts.append(dict(**pin(q),first_line=a,last_line=b,source=q.read_text().splitlines()[a-1:b]))
    facts=[
        'Reuse v24/v62 logical pairing, even-history observability, pool/update WAR and token-aligned zero/norm answers; v88 already covers original-typed SWA+C2 FLOAT carrier. New scope is composition with HC fork and owner partition versus original request-segment selectors.',
        'HC current coefficient consumes raw4H/FLOAT-HF32 X; C2 consumes normalized rounded BF16 Y widened to FLOAT, with two distinct FLOAT weight matrices and FLOAT outputs. FLOAT storage alone does not bind the C2 math mode or permit a shared-input wide GEMM with HC coefficient. Current coefficient, post/comb, following pre4 and raw-X residual remain live.',
        'Each original valid request segment (s,n,base,page) completes floor((s+n)/2)-floor(s/2) pairs. Only its first odd-start pair reads OLD_RING at the even logical position; all other partners are CURRENT segment projections. SP owner boundaries do not create new logical history boundaries.',
        'Pair owner must preserve original branch and FP32 arithmetic/softmax then BF16 pooled rounding. Fetch a missing current partner from its current producer, not stale ring; genuine OLD_RING reads need matching page and generation. A numerically plausible halo injected through the residual branch is not a native bit-parity proof.',
        'Keep existing full32-slot replicated FP32 ring ABI: latest min(n,32) current K+score rows per valid segment must reach every required state replica before future old-ring reads, with original write order/duplicates/epochs. Pool-only BF16 endpoints cannot reconstruct that state. F34 even-only pruning remains a separate conditional observer-closure contract.',
        'Current source zeroes all T pooled BF16 rows, stores E completions at original completion-token rows, then calls 512-wide RMSNorm on all T rows. Preserve zero initialization, output map, BF16 boundary and original norm semantics. Noncompletion post-norm/index/store work is not dead from shapes; actual slot/readers/nonfinite behavior are unbound.',
        'Pool old-ring read precedes overlapping ring overwrite; pooled BF16 precedes norm. Update->norm is observed current same-stream order, not a universal value dependency under disjoint buffers/effects. Deferred state work still meets future-read and projected-input/metadata/bank retirement deadlines.',
        'Owner projections require actual W replica/epoch/newM precision and normalized row/rank maps; pooled owner execution also needs original request segments, pages, positions, residual generations and state placement. Distinct allocated source objects do not prove loaded equality or inequality.',
        'A state-owner redesign may route future residual reads instead of replicating tails, but requires a new ownership/migration/rollback/capture protocol. It cannot inherit unchanged local replicated-ring reader behavior.',
        'Current positive fallback INDEX_WK reads unrotated rounded normalized latent before overlapping long-RoPE mutation. Keep per-row/range old-value WAR, INT8 key and FP16 scale rounding/stores; not whole index publication barrier. Source pending/packed/empty/!owns_k alternatives are not closed.'
    ]
    write(O/'hc-c2-source-and-state-contract-v136.json',dict(parts=parts,facts=facts,original_E='sum_valid floor((s+n)/2)-floor(s/2)',original_B='count_valid(s odd and E_segment>0)',original_U='sum_valid min(n,32)',
        page_gate='0<page<NUM_CACHE_BLOCKS; original metadata row bounds/request ownership remain gates',new_device_work=False,overall_complete=False))
    cases=[];witnesses={}
    for s,n,base,valid in product(range(12),list(range(10))+[31,32,33,63,64,65],[0,3],[False,True]):
        for cut in range(n+1):
            r=pair_case(s,n,cut,base,valid);cases.append(r)
            if r['naive_changed_current_vs_old_pairs']:witnesses.setdefault('SP_BOUNDARY_IS_NOT_HISTORY',r)
            if r['naive_tail_extra_writes']:witnesses.setdefault('SHARD_LOCAL_TAIL_IS_NOT_ORIGINAL_TAIL',r)
    scalar=dict(original_segment=dict(s=0,n=2,base=0),split_at=1,current_K=[1,3],current_scores=[0,0],stale_ring_even_K=9,
        correct_pooled=2,wrong_shard_history_pooled=6,meaning='Scalar equal-score finite example; BF16 2 and 6 remain distinct. No native exp/reduction parity claim.')
    state=dict(current_K_A=[1,3],current_K_B=[0,4],scores=[0,0],current_pooled_both=2,
        rollback_next_segment=dict(s=1,n=1,current_K=3,score=0),future_pooled_A=2,future_pooled_B=1.5,
        meaning='Identical current BF16 pool output can conceal different observable FP32 even ring positions. Endpoint-only communication loses future state under this admissible helper-domain rollback-to-position1; actual workload acceptance/rollback policy is not bound. Preserving only latest even also lacks the earlier even observer proof from v24.')
    write(O/'hc-c2-pair-owner-and-state-witnesses-v136.json',dict(cases=cases,case_count=len(cases),witnesses=witnesses,stale_ring_scalar=scalar,endpoint_only_loses_future_state=state,
        owner_aware_rule='Keep original (request,generation,pair,branch) and assign completion owner; fetch CURRENT left halo if owner differs, genuine old history otherwise. Preserve original tail set and one legal update order for collisions.',
        native_or_actual_request_metadata_proof=False,new_device_work=False,overall_complete=False))
    geoms=Counter();work=[]
    for r in prof['records']:
        p=r['eager_HC_SP_parent'];m=p['SP_allGather']['COMM']['count']//5120;n=int(p['C2_KV']['definition']['inputShapes_text'].strip('"').split(',')[0]);q=int(p['QA']['definition']['inputShapes_text'].strip('"').split(',')[0]);geoms[m,n,q]+=1
    for (m,n,q),count in sorted(geoms.items()):
        work.append(dict(sender_rows=m,global_rows=n,QA_rows=q,frontiers=count,uniform_AG_rows=8*m,
            original_hidden_sender_content=10240*m,original_typed_SWA_and_C2_sender_content=5120*m,with_QA_typed_sender_content=7680*m,
            C2_pair_current_dense_ops=4*n*5120*512,C2_pair_owner_dense_ops=4*m*5120*512,
            BF16_to_FLOAT_cast_logical_value_IO=6*n*5120,C2_FLOAT_weights=20971520,SWA_BF16_weights=5242880,HC_FLOAT_coeff_weights=1966080,
            union_HC_SWA_C2_weight_contents=28180480,with_QA_union_weight_contents=41287680,
            metadata_E_B_U_and_actual_wire_HBM_or_new_native_mathmode_bound=False))
    # Reference only: two distinct even-start 3200-row requests; contiguous 8-way ownership.
    T=6400;P=8;n=800;segs=[(0,3200,0),(0,3200,3200)];es=[0]*P;us=[0]*P;us_even=[0]*P;X=0
    for s,length,base in segs:
        ps,tail=original(s,length,base,True)
        for p in ps:
            es[p['right_row']//n]+=1;X+=p['left_kind']=='CURRENT'and p['left_row']//n!=p['right_row']//n
        for row,pos,slot in tail:
            us[row//n]+=1;us_even[row//n]+=pos%2==0
    content=1024*sum(es)+4096*sum(us);fixed=1024*max(es)+4096*max(us);evenfixed=1024*max(es)+4096*max(us_even)
    write(O/'hc-c2-delivery-cuts-and-work-v136.json',dict(profile_geometry_models=work,
        cuts=[dict(id='ORIGINAL_TYPED_PROJECTIONS',sender_content='SWA BF16 512 + C2 K FLOAT512 + score FLOAT512 = 5120*m; optional QA BF16 1280 adds2560*m.',receiver='Original global pool/zero/norm and replicated ring update; global row trimming and all raw readers must close.'),
            dict(id='PAIR_OWNER_POOL_WITH_REPLICATED_STATE',application_content='C2 pooled BF16:1024*E plus K+score FP32 tails:4096*U; SWA1024*T and optional QA2560*T additional. Halo source data4096*X and remote genuine history4096*B_remote if needed.',receiver='Reconstruct original T token-aligned zero buffer and completion map, original norm/reader pipeline; install original ring tail before future demand. Delivery fanout/header/alignment/metadata, raw readers and residency additional.'),
            dict(id='PAIR_OWNER_POOL_WITH_STATE_OWNER',application_content='Pooled endpoints plus current-pair halos and routed history/state-owner work; no declared numeric byte minimum.',receiver='New state-owner routing/migration/page-generation/rollback/capture protocol required; current local ring ABI cannot be reused unchanged.')],
        fixed_count_rule='For separate fixed-count collectives: pooled per-sender cap1024*max_p e_p; tail cap4096*max_p u_p. Independent sum of caps is an upper provisioning reference; a jointly packed per-sender buffer may instead use max_p(1024*e_p+4096*u_p), plus metadata/alignment. E/P and U/P are not safe caps without distribution proof.',
        reference=dict(T=T,P=P,sender_rows=n,segments=[dict(s=s,n=l,base=b)for s,l,b in segs],E=sum(es),U=sum(us),per_sender_completed=es,per_sender_tail=us,
            per_sender_even_tail=us_even,cross_owner_current_pair_halos=X,aggregate_C2_value_content=content,fixed_count_C2_content_per_sender=fixed,
            fixed_count_C2_even_only_content_per_sender=evenfixed,with_SWA_fixed_per_sender=fixed+1024*n,with_SWA_QA_fixed_per_sender=fixed+3584*n,
            wrong_mean_tail_cap=4096*(sum(us)//P),full_tail_cap=4096*max(us),original_C2_pair_sender_content=4096*n,original_hidden_sender_content=10240*n,
            actual_profile_metadata_or_even_observer_closure_proved=False),
        retention='After pool, deferred update can retain an explicit FP32 tail snapshot rather than all projected rows if all other readers and metadata generations are closed; copy/queue/visibility costs remain. Even-only F34 savings cannot be added to unchanged full-state ABI.',
        theory='SYMBOLIC_CEILING: T>=sup_lambda(lambda.W_necessary)/beta_joint_upper only with a mandatory servicedomain, one initial warm grant and valid joint-capacity upper. These logical plane contents/dense references do not establish HBM traffic, physical MFU, latency or optimality.',
        new_device_work=False,overall_complete=False))
    ts=[];classes=Counter();wits={}
    for s,k,h,c in product([1,5],repeat=4):
        for p,t,cp,ct,cj,u in product([1,4],[0,8],[1,3],[1,3],[1,3,7],[1,4]):
            r=dict(s=s,k=k,h=h,c=c,p=p,t=t,cp=cp,ct=ct,cj=cj,u=u);r.update(timing(r));ts.append(r)
            cl='JOINT_CURRENT_FASTER'if r['joint_current']<r['split_current']else'SPLIT_CURRENT_FASTER'if r['split_current']<r['joint_current']else'TIED_CURRENT';classes[cl]+=1;wits.setdefault(cl,r)
    write(O/'hc-c2-current-versus-future-DAG-models-v136.json',dict(cases=ts,case_count=len(ts),classes=dict(classes),witnesses=wits,
        domain='Hypothetical services in arbitrary units, independent noncontending resources except one ordered communicator. Chosen whole-pool publication frontier also marks completion of all its old-ring reads; gating ring update by that frontier is a sufficient chosen schedule, not a necessary whole-pool barrier. Real WAR applies only to overlapping ranges. SWA/pool packet and FP32 tail packet are joint or ordered separate. Norm needs pooled output, not ring write completion under declared disjointness. Current endpoint is attention HCPost, plus separate future ring completion. Later pre4/FFN and actual resource contention not priced.',
        symbols='s:SWAready,k:C2projectionsready,h:oldhistoryready,c:HCcoeffready,p:poolservice,t:tail-snapshotservice,cp/ct/cj:pool/tail/jointmessageservices,u:ringwrite. Receiver norm/index/publication combined4; queryready2; attention/output2.',
        no_actual_device_overlap_or_speedup_or_universal_barrier_proved=True,new_device_work=False,overall_complete=False))
    write(O/'hc-c2-joint-graph-and-clauses-v136.json',dict(pair=['F01','F17'],new_pairs=1,new_groups=0,groups_to_extend=[['HG93-HC-ATTENTION-FORK-DEADLINE','F17'],['HG73-C2','F01'],['HG88-SWA-C2-TYPED-PROJECTION','F01']],
        required_value_edges=[['OLD_MIX','Y_BF16'],['X','Y_BF16'],['X','HC_COEFF'],['Y_BF16','NORM_H_BF16'],['NORM_H_BF16','C2_INPUT_FLOAT'],['C2_INPUT_FLOAT','K_FLOAT'],['C2_INPUT_FLOAT','SCORE_FLOAT'],['K_FLOAT','POOL_BF16'],['SCORE_FLOAT','POOL_BF16'],['ORIGINAL_OLD_RING_READS','POOL_BF16'],['POOL_BF16','NORM512_BF16'],['ORIGINAL_OLD_RING_READS','OVERLAPPING_RING_UPDATE'],['K_FLOAT_TAIL','OVERLAPPING_RING_UPDATE'],['SCORE_FLOAT_TAIL','OVERLAPPING_RING_UPDATE'],['OVERLAPPING_RING_UPDATE','FUTURE_OLD_RING_READS'],['NORM512_BF16','INDEX_WK_OLD_READ'],['INDEX_WK_OLD_READ','OVERLAPPING_LONG_ROPE'],['INDEX_WK_OLD_READ','INDEX_PUBLICATION'],['OVERLAPPING_LONG_ROPE','LONG_PUBLICATION'],['INDEX_PUBLICATION','ATTENTION_OUTPUT'],['LONG_PUBLICATION','ATTENTION_OUTPUT'],['ATTENTION_OUTPUT','HC_POST'],['HC_COEFF','HC_POST'],['X','HC_POST']],
        next_edges=[['HC_COEFF','CURRENT_PRE4'],['CURRENT_PRE4','NEXT_FFN_PRE'],['HC_POST','NEXT_FFN_PRE']],
        not_universally_required_edges=[['OVERLAPPING_RING_UPDATE','NORM512_BF16'],['INDEX_PUBLICATION','OVERLAPPING_LONG_ROPE']],
        edge_scope='Per-range/request/page/generation value dependencies, not universal whole-op barriers. Owner transport/zero reconstruction/retirement and all raw readers add chosen-variant edges. Native 72 records are current execution evidence, not historical tensor-alias or transformed-native legality proof.',
        native_joint_fusion_or_all_subsets_proved=False,one_HC_C2_baseline=True,new_device_work=False,overall_complete=False))
    names=['hc-c1-postcheck-result-v135p.json','candidate-relation-clauses-v135.json','candidate-pair-navigation-v135.json','hc-c2-native-consumers-v136.json','hc-sp-eager-native-frontiers-v134.json',
        'C2-state-opportunity-contract-v24.json','C2-ring-observability-integer-review-v24.json','C2-state-and-pool-costs-v62.json','state-engram-operator-contract-v62.json',
        'mixed-SWA-C2-source-contract-v88.json','mixed-SWA-C2-project-before-gather-v88.json','hc-sp-reader-generation-and-row-witnesses-v133.json','hc-sp-eager-typed-carrier-and-padding-models-v134.json','hc-c1-joint-graph-and-clauses-v135.json']
    paths=[O/n for n in names]+[Path(p['path'])for p in parts]
    write(O/'hc-c2-inputs-v136.json',dict(inputs=[pin(p)for p in sorted(set(paths))],profile_first=True,
        questions=['Which exact native C2 zero/pool/update/norm/old-latent consumers follow the 72 HC/SP C2 parents?',
            'Can owner partition preserve original request pair/residual selection and every observable future ring generation?',
            'Which pooled/state delivery cuts remove projection replication without forcing current attention to wait for future-only state work?'],new_device_work=False,overall_complete=False))
    print(json.dumps(dict(status='HC_C2_COMPOSITION_CONDITIONALLY_REVIEWED',frontiers=72,roles=16,geometry_cases=len(geoms),pair_cases=len(cases),DAG_cases=len(ts),DAG_classes=dict(classes),new_scoped_pair=1,new_device_work=False)))


if __name__=='__main__':main()
