"""HC with C2 pool/norm and even-state alternatives, using original request demand."""
from pathlib import Path
from collections import defaultdict,Counter
from itertools import product
from zipfile import ZipFile
import hashlib,json,sqlite3,math,struct
from finalize_origin_coverage_v9 import N,O,W,read,write


def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def bf(x):
    u=struct.unpack('>I',struct.pack('>f',x))[0];u=(u+0x7fff+((u>>16)&1))&0xffff0000
    return struct.unpack('>f',struct.pack('>I',u))[0]


def reference_norm(v,eps=1e-6):
    scale=1/math.sqrt(sum(x*x for x in v)/len(v)+eps)
    return [bf(x*scale)for x in v]


def native_norm_wrappers():
    old=read(O/'hc-c2-native-consumers-v136.json');refs={r['database_index']:r for r in old['databases']};groups=defaultdict(list);records=[]
    for r in old['records']:groups[r['database_index']].append(r)
    for di,rows in sorted(groups.items()):
        ref=refs[di];db=sqlite3.connect(Path(ref['raw_database']['path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row
        ss={r['id']:r['value']for r in db.execute('SELECT *FROM STRING_IDS')};wanted={r['consumers']['POOL_NORM']['API']['callchainId']for r in rows};fs=defaultdict(list);bias_chains=set()
        for f in db.execute('SELECT *FROM PYTORCH_CALLCHAINS'):
            v=ss[f['stack']]
            if f['id']in wanted:fs[f['id']].append(dict(f,value=v))
            if '/vllm_ascend/ops/layernorm.py(101):'in v:bias_chains.add(f['id'])
        # Only source-line-positive candidates; no nearest-time tensor or branch inference.
        aps=[dict(a,name_text=ss[a['name']])for a in db.execute('SELECT rowid AS api_rowid,*FROM PYTORCH_API')if a['callchainId']in bias_chains]
        for r in rows:
            norm=r['consumers']['POOL_NORM'];wk=r['consumers']['INDEX_WK'];a=norm['API']
            frames=sorted(fs[a['callchainId']],key=lambda f:f['stackDepth'])
            assert any('/vllm_ascend/ops/layernorm.py(99): forward_oot'in f['value']for f in frames)
            hits=[x for x in aps if x['globalTid']==a['globalTid']and int(a['endNs'])<=int(x['startNs'])<=int(x['endNs'])<=int(wk['API']['startNs'])]
            records.append(dict(database_index=di,forward_API_rowid=r['forward_API_rowid'],source_block_ordinal=r['source_block_ordinal'],norm_API=a,norm_TASK=norm['TASK'],norm_definition=norm['definition'],
                next_index_WK_API=wk['API'],full_norm_callchain=frames,recorded_bias_line101_APIs_before_index=hits,
                runtime_bias_loaded_or_all_external_rstd_observers_bound=False))
        db.close()
    write(O/'hc-c2-norm-wrapper-profile-v137.json',dict(records=records,frontiers=72,databases=old['databases'],
        source_positive_no_residual_forward99=72,recorded_bias_line101_APIs=sum(len(r['recorded_bias_line101_APIs_before_index'])for r in records),
        absence_of_recorded_Add_is_not_loaded_bias_false_proof=True,new_native_promotions=0,new_device_work=False,overall_complete=False))


def request_case(s,n,valid):
    current=set(range(s,s+n))if valid else set();pool=set();old=[];complete=[]
    if valid:
        for q in range(s//2,(s+n)//2):
            complete.append(2*q+1)
            for t in [2*q,2*q+1]:
                if t>=s:pool.add(t)
                else:old.append(t)
    full=set(range(s+max(0,n-32),s+n))if valid else set();even={t for t in full if t%2==0}
    return dict(s=s,n=n,valid=valid,E=len(complete),B=len(old),U=len(full),U_even=len(even),current_rows=len(current),
        current_pool_rows=len(pool),current_pool_or_full_tail_rows=len(pool|full),current_pool_or_even_tail_rows=len(pool|even),
        missing_from_pool=sorted(current-pool),missing_from_pool_and_even_tail=sorted(current-(pool|even)),
        even_state_retention_does_not_cut_current_projection_rows=(pool|even)==current)


def main():
    assert not(O/'REVIEW-CHECKPOINT-v137.json').exists()
    native_norm_wrappers();root=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend';parts=[]
    for rel,a,b in [('models/deepseek_v41/model.py',887,920),('models/deepseek_v41/compressor.py',16,33),('models/deepseek_v41/compressor.py',55,67),('attention/dsa_v41.py',523,565)]:
        q=root/rel;parts.append(dict(**pin(q),first_line=a,last_line=b,source=q.read_text().splitlines()[a-1:b]))
    q=N/'state-engram-operator-source-v62/vllm_ascend/ops/triton/compressor/compressor_triton.py'
    for a,b in [(225,245),(305,319),(348,364),(647,721)]:parts.append(dict(**pin(q),first_line=a,last_line=b,source=q.read_text().splitlines()[a-1:b]))
    supp=W/'sampler-source-supplement';wrapper=supp/'frozen-source-snapshot/vllm-ascend/vllm_ascend/ops/layernorm.py';archive=supp/'frozen-sources.zip';receipt=supp/'source-collection-receipt.json';member='vllm-ascend/vllm_ascend/ops/layernorm.py'
    assert pin(archive)['sha256']==read(receipt)['archive_sha256']
    with ZipFile(archive)as z:
        possible=[x for x in z.namelist()if x.endswith(member)];assert len(possible)==1
        assert z.read(possible[0])==wrapper.read_bytes();member=possible[0]
    parts.append(dict(**pin(wrapper),first_line=49,last_line=103,source=wrapper.read_text().splitlines()[48:103]))
    write(O/'hc-c2-norm-state-source-contract-v137.json',dict(parts=parts,wrapper_archive=pin(archive),wrapper_member=member,wrapper_receipt=pin(receipt),
        prior_answers=['v24/v62pool-normBF16/tokenmap/oldreadWAR/evenobserver','v79SWA-anchoredpool/privatebank/ABA','v136originalrequestpair-vs-SPowner/current-futurepacket'],
        facts=[
            '72source/CIDnativeRMSfrontiers point to AscendRMSNorm.forward_oot line99 no-residual branch. This source computes npu_rms_norm then optional in-place bias add line101. The locally named rstd/residual is not used/returned by this direct branch; external hooks/native ABI/debug/allocator observers are a separate unclosed gate.',
            'RMS result is BF16 before optional x.add_(bias), whose BF16 output boundary must be retained unless numeric equivalence is proved. Moving norm to owner requires gamma/epsilon and loaded beta/branch replicas; projection-only delivery can leave norm at receiver.',
            'Token-aligned helper starts T pooled rows at +0 and writes E completion rows. A fused pool+norm may generate original final noncompletion rows directly, but cannot assume +0 final under signed weights, nonfinite values, epsilon or loaded beta. Preserve completed-row BF16 pool round then RMS and optional BF16 bias add.',
            'Original valid request current pool demand includes every current token except at most the final unpaired even token. That token is always in the current latest32 tail and retained-even tail. Thus D_pool_current union D_even_tail equals all current request rows, even in the optimistic retained-even observer domain.',
            'Consequently F34 odd-write pruning and F18 completed-row norm cuts do not remove current valid C2 projection/Y input rows by themselves. Invalid page/unused padding/other-reader closure may change the active domain; fused materialization/recomputation choices are still possible.',
            'HC raw4H coefficient, current post/comb, nextFFNpre4 and raw-X Post residual remain distinct from normalized Y/C2 FLOAT input. No C2 completion or norm event retires current X while Post still reads it.',
            'Pool-norm fusion and norm-before-communication choose one delivery graph and one latent/pooled interface credit. Retaining pooled or normalized latent for another reader, fallback, graph output or alias reduces deletable storage. FusedNorm must keep all512 lanes and real gamma/eps/beta, not flip TOKEN_ALIGNED with placeholder arguments.',
            'State representations are alternatives: unchanged32, in-place even, private32/private16 or shared paired16. Existing SWA-anchored divisor540928 does not shrink from even compaction. New private-bank content and additional HCY/Z, pooled and tail copies share one device live-memory budget. Existing persistent pooled allocation is already profiled before cache budgeting; retaining it is neither new allocation nor freed-byte credit.',
            'Physical aliasing can require reuse/drain but must not create invented semantic update->norm or whole-index-publication barriers. Old ring reads precede overlapping writes; source/metadata/owner/page generations retire after last readers AND queued writers; per-range partial publication requires a new verified native contract.'
        ],loaded_bias_or_new_native_precision_or_all_readers_closed=False,new_device_work=False,overall_complete=False))
    cases=[request_case(s,n,v)for s,n,v in product(range(128),range(129),[False,True])]
    write(O/'hc-c2-current-row-demand-proof-v137.json',dict(cases=cases,case_count=len(cases),
        symbolic_proof=[
            'Each completed q uses logical tokens2q,2q+1. The sole possible old token is2floor(s/2)<s at first odd start; every other pair member is current.',
            'For current interval[s,s+n), each token before a completed odd endpoint is in one pair. The only possible uncovered current token is final token s+n-1 when that position is even.',
            'For n>0 the final token belongs to last min(n,32) tail, and if uncovered it is even, hence in retained-even tail. For n=0 all sets empty. Valid pages therefore satisfy D_pool_current union D_even_tail=[s,s+n). Invalid pages use empty current operator demand in this selected helper domain.',
            'This is exact logical row demand of the preserved direct operators, not a universal arithmetic/HBM lower bound for all equivalent algorithms. Actual metadata and other readers remain unbound.'
        ],singletons=[request_case(0,1,True),request_case(1,1,True)],source_native_or_actual_metadata_test=False,new_device_work=False,overall_complete=False))
    p=[0.84375,0.849609375];r=[bf(x)for x in p];a=[1,0];b=[0,2];pooled=[(x+y)/2 for x,y in zip(a,b)];na=reference_norm(a,0);nb=reference_norm(b,0)
    write(O/'hc-c2-norm-round-and-zero-witnesses-v137.json',dict(
        rounding=dict(pooled_FLOAT=p,pooled_BF16=r,epsilon=1e-6,preserve_round=reference_norm(r),skip_round=reference_norm(p),embedding='Repeat the two feature values256 times to obtain512features; same reference mean square.',
            scope='Finite scalar FP64-reference RMS followed by BF16 rounding of FP32-rounded input; not actual CANN reduction/native bit-parity. Demonstrates missing-round substitution is not a general numerical identity.'),
        noncommuting=dict(inputs=[a,b],pooled=pooled,norm_after_pool=reference_norm(pooled,0),pool_after_each_norm=[(x+y)/2 for x,y in zip(na,nb)],epsilon=0,
            scope='Finite reference example; RMS512 after pool is not input RMS5120 and cannot migrate through weighted pool/projection by matching names/widths.'),
        zeros=dict(pooled_initial_plus_zero=True,finite_positive_eps=1e-6,negative_gamma_channel=-1,negative_zero_sign=-1,
            loaded_beta_example=1,final_noncompletion_value_with_loaded_beta=1,epsilon0_or_nonfinite_branch_requires_original_semantics=True),
        rstd_direct_caller='No direct returned rstd in selected no-residual wrapper source. This does not prove no external/native/capture content or ABI observer.',new_device_work=False,overall_complete=False))
    geoms=Counter();models=[]
    for r in read(O/'hc-c2-native-consumers-v136.json')['records']:
        p=r['eager_HC_SP_parent'];m=p['SP_allGather']['COMM']['count']//5120;t=int(p['C2_KV']['definition']['inputShapes_text'].strip('"').split(',')[0]);geoms[m,t]+=1
    for (m,t),count in sorted(geoms.items()):
        models.append(dict(sender_rows=m,T=t,frontiers=count,profile_E_or_active_rows_not_bound=True,
            full_token_norm_value_IO=4*512*t,full_token_norm_rstd_descriptor_value_content=4*t,
            pool_norm_interface_budget='2*512*(E+T); if no intermediate observer; logical only',
            with_full_final_generation_budget='2*512*E+4*512*T; alternative total, not additive to preceding budget',
            pool_BF16_delivery_content='1024*E',normalized_BF16_delivery_content='1024*E if same row map, all noncompletion outputs reconstructed/closed and receiver-specific norm parameters match',
            optional_private_Y_and_normedY_content=4*5120*m,HC_X_value_content=2*4*5120*m,
            C2_full_current_projection_dense_ops_per_active_row=4*5120*512,C2_projection_row_deletion_from_even_state_or_E_only_norm=0,
            old_state_representation_choice_not_free_pool_capacity=True))
    cuts=[dict(id='PRESERVE_ALL_T_NORM',native='Original token-aligned zeroT/poolE, receiver fullT RMS512 plus optional bias. Projection-only owner path leaves norm params local.'),
        dict(id='LOCAL_POOL_NORM_FUSION_FULL_T_FINAL',native='Compute exact BF16 pool, RMS512 and optional bias for completions; generate all final noncompletion rows under original semantics. No global pooled buffer if every other observer closed; preserve latent old-index fork.'),
        dict(id='OWNER_POOL_THEN_RECEIVER_NORM',native='Send BF16 pooledE, reconstruct original zeroT and completion map, retain fullT receiver norm/bias. No owner gamma equality requirement beyond unchanged projection/pool gates.'),
        dict(id='OWNER_POOL_NORM_WITH_RECONSTRUCTION',native='Send normalized BF16 completions; receiver must reconstruct every demanded noncompletion final value or prove it dead. Requires loaded gamma/eps/beta/native-M math and all row/position/state/capture gates; ring state still separate.')]
    write(O/'hc-c2-pool-norm-delivery-and-work-v137.json',dict(models=models,cuts=cuts,
        state_maintenance_references='FullU and retained-evenUe writes change state service, not valid current projection row union. Ue<=U; neither per-request tail count nor E alone prices active valid rows.',
        one_graph_credit='Choose one baseline/transformed boundary graph. Pool-norm interface deletion, moving norm to owner and eliminating a retained pool bank are not independent additive bytes. Price required original final rows, optional bias, rstd observer, output stores and new transport/snapshots once.',
        resource_cut='Necessary value/operation demand only in declared operator domain. T>=sup_lambda lambda.W_necessary/beta_joint_upper remains symbolic: mixed HC/C2/SWA Cube/Vector/SFU/link joint capacity, physical traffic and actual native services are unbound.',
        new_device_work=False,overall_complete=False))
    budgets=[]
    for rows,U,S,z,m,needY,needZ,poolkeep,tailkeep in product([16,32],[1,4,16],[0,1,4],[0,1],[800,834],[False,True],[False,True],[False,True],[False,True]):
        if S>U:continue
        bank=(U+z)*3*rows*1024*4;private=(2*5120*m)*(int(needY)+int(needZ));pool=2*512*8*m if poolkeep else 0;tail=8*512*32*2 if tailkeep else 0
        D=bank+private+pool+tail;q,t=divmod(D,540928);budgets.append(dict(rows=rows,U=U,S=S,z=z,m=m,privateY=needY,privateZ=needZ,additional_pooled_snapshot=poolkeep,additional_tail_snapshot=tailkeep,
            state_bank_bytes=bank,HC_optional_private_YZ_bytes=private,pooled_snapshot_bytes=pool,tail_snapshot_bytes=tail,additional_live_content_bytes=D,
            min_global_ID_loss=q,max_global_ID_loss=q+int(t>0),min_net_free_ID_change=S-q-int(t>0),max_net_free_ID_change=S-q,
            HC_X_value_content_retired_by_C2_completion=False,metadata_alignment_and_transport_content_additional=True))
    # U=resident-bank capacity, not updated tail U in v136. Distinct symbol namespace is explicit.
    write(O/'hc-c2-state-and-HC-live-budget-v137.json',dict(cases=budgets,case_count=len(budgets),
        symbols='U here=reserved private bank OWNER capacity; S=resident state owners; z=invalid owner slot. Not v136 U=sum updated tail rows. m=HC sender rows,chosen T=8m reference; actual T trim/owner capacity/lifetimes unbound.',
        domain='One selected NPU arena, unchanged normal shared pool divisor C540928, three C2 layers, nooverride/unpacked source domain. All listed NEW buffers assumed overlap at one named time; alternatives/real event intervals can reduce peak. Baseline B already accounts for existing HC and persistent pooled buffers; only additional distinct copies are charged. State release S counted once, all additional content charged once.',
        formula='D_total=privateBank+additionalHCY/Z+additionalPooledCopy+additionalTailSnapshot+other; DeltaFree=S-ceil((D_total-r)/C) for original B=qC+r,0<=r<C, equivalently S-(floor(B/C)-floor((B-D_total)/C)). Bounds S-ceil(D_total/C)..S-floor(D_total/C). Require budget B>=D_total; metadata/alignment/transport may worsen.',
        nonclaims=['Existing HC X is live until its original current coefficient/Post residual readers retire; not extra private-bank allocation credit.',
            'Existing ring16 view does not change SWA-anchored divisor or free bytes. Full private32/even private16/paired16 are alternatives, not stacked savings.',
            'This models a selected concurrent peak; no per-kernel free-memory sum, no cross-rank pooled budget, no actual resident bound or allocator snapshot.'],
        new_device_work=False,overall_complete=False))
    write(O/'hc-c2-norm-state-composition-v137.json',dict(new_pairs=[['F01','F18'],['F01','F34']],new_pair_clauses=2,new_groups=0,
        group_extensions=[['HG93-HC-ATTENTION-FORK-DEADLINE','F18'],['HG93-HC-ATTENTION-FORK-DEADLINE','F34']],
        clauses=[dict(id='REL137-01',a='F01',b='F18',contract='HCexternalY/Hnorm/C2pool->BF16->512norm are distinct nonlinear/rounding domains. Pool-norm/fullT-final or owner-norm variants preserve completion map, zero/bias/FP16indexscale/oldlatent readers, currentHCcoeff/Post/pre4 and actual shared resource/readiness. No sum of old interface/launch/privatebank credits.'),
            dict(id='REL137-02',a='F01',b='F34',contract='HC current rawX/coeff/Post lifetime survives C2 state or norm completion. Even-state tail plus current pool still demands every current valid C2 input row; no proportional HC Y or C2 projection deletion. Statebank and optionalHCcopies/snapshots share one live budget; originalSWAanchoreddivisor/owner-generation/last-reader-and-writer/rollback/nativeABI gates remain.')],
        existing_F18_F34_relation_reused=True,all_runtime_gates_proved=False,critical_path_gain_us=None,whole_pair_or_all_subsets_closed=False,new_device_work=False,overall_complete=False))
    names=['hc-c2-postcheck-result-v136p.json','candidate-relation-clauses-v136.json','hc-c2-native-consumers-v136.json','hc-c2-source-and-state-contract-v136.json','hc-c2-joint-graph-and-clauses-v136.json',
        'hc-c2-delivery-cuts-and-work-v136.json','F18-pool-update-norm-domains-v24.json','C2-state-and-pool-costs-v62.json','ring-storage-capacity-models-v79.json','ring-state-layout-and-lifetime-proof-v79.json','hc-c2-norm-wrapper-profile-v137.json']
    paths=[O/n for n in names]+[Path(p['path'])for p in parts]+[archive,receipt]
    write(O/'hc-c2-norm-state-inputs-v137.json',dict(inputs=[pin(p)for p in sorted(set(paths))],profile_first=True,
        questions=['Which exact norm wrapper and optional bias/rstd behavior bound pool-norm/owner-norm alternatives?',
            'Can retained-even future state plus sparse completed pooling delete current valid HC Y/C2 projection rows?',
            'How do optional HC private copies and retained pool/tail snapshots change private state-bank capacity on the same live arena?'],new_device_work=False,overall_complete=False))
    print(json.dumps(dict(status='HC_C2_NORM_STATE_CONDITIONALLY_RECONCILED',native_wrapper_frontiers=72,demand_cases=len(cases),budget_cases=len(budgets),new_scoped_pairs=2,new_device_work=False)))


if __name__=='__main__':main()
