"""HC/query-RoPE join and key/old-latent versioned fanout alternatives."""
from pathlib import Path
from collections import Counter,defaultdict
from itertools import product
import hashlib,json,statistics
from finalize_origin_coverage_v9 import N,O,W,read,write


def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def query_time(r,fork,hoist,reprice=False):
    u=(r['y']if fork else max(r['y'],r['c']))+r['n'];qb=u+r['qb'];sw=u+r['sw'];coeff=r['c']
    if hoist:
        ready=max(qb+r['rope']+(r['dr']if reprice else 0),sw+(r['ds']if reprice else 0));coeff+=r['dc']if reprice else 0
    else:ready=max(qb,sw)+r['rope']
    return max(coeff,ready+r['tail'])


def latent_times(r):
    wk=r['l']+r['w'];idx=wk+r['i'];oldlong=idx+r['g'];qidx=r['qi']
    def post(long,qli):return max(r['c'],max(r['q'],long,qli)+r['out'])
    return dict(baseline=post(oldlong,max(qidx,oldlong)+r['k']),
        QLI_before_long_join=post(oldlong,max(qidx,idx)+r['k']),
        long_after_WK_read=post(wk+r['g']+r['extra'],max(qidx,idx)+r['k']),
        readonly_long_direct=post(r['l']+r['g']+r['extra'],max(qidx,idx)+r['k']),
        saved_old_tail_segmented_WK=post(r['l']+r['extra']+r['g'],max(qidx,r['l']+r['extra']+r['w']+r['i'])+r['k']))


def main():
    assert not(O/'REVIEW-CHECKPOINT-v140.json').exists()
    prof=read(O/'hc-rope-latent-native-frontiers-v140.json');idx={(r['database_index'],r['forward_API_rowid'],r['source_block_ordinal']):r for r in read(O/'hc-sparse-eager-native-frontiers-v138.json')['records']};root=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend';parts=[]
    for rel,a,b in [('models/deepseek_v41/model.py',887,920),('attention/dsa_v41.py',290,308),('attention/dsa_v41.py',351,388),('attention/dsa_v41.py',477,495),
        ('attention/dsa_v41.py',523,565),('attention/dsa_v41.py',580,585),('models/deepseek_v41/indexer.py',106,125),('models/deepseek_v41/compressor.py',55,71)]:
        p=root/rel;parts.append(dict(**pin(p),first_line=a,last_line=b,source=p.read_text().splitlines()[a-1:b]))
    write(O/'hc-rope-latent-source-contract-v140.json',dict(parts=parts,
        prior_answers=['v27old-latentperrangeWAR/notwhole-index-publicationbarrier','v62pool/norm/C2stateandC1-distinctmode','v82QB_BF16→tail64RoPEwithoutpostQBnorm512','v87currentcachepublication/directproducerbaseline','v135/v136nativeindexkey/scale/longconsumers','v137BF16norm/bias/zeroandstateproof','v139typedquerykeyquantandauxstream'],
        facts=[
            'Current main Query RoPE374 is already before prepare_indexer_inputs376. Do not claim a new gain from moving it ahead of IndexQ/Weight. It follows multistream_preprocess whose main waits aux at492 after QB491; the remaining candidate is moving Query RoPE before this SWA join or a QB epilogue,with correct query C/S inputs.',
            'Query RoPE must keep full512 output,only64tail rotation,BF16QB round then original RoPEBF16 round. Current path has no postQB RMS512. Separate early RoPE scheduling changes no GM interface; fused QB epilogue may remove only the declared tail interface under legal ownership/partner halves/native handoff.',
            'Query C/S must be consumed before compression source-table reuse as stated at375. Hoisting before SWA join still needs exact C/S publication/alias/lifetime and extra Vector contention repricing. Empty local Q still updates nonempty cache sources; this positive-query scope does not delete those writes.',
            'C1/C2 normed BF16 latent512 feeds IndexWk512→128 and longKV. IndexWk must read each overlapping old unrotated value before in-place longRoPE modifies it,or use an independent value version. IndexNorm128/quant/indexcache publication does not itself read that old512latent; a whole publication→longRoPE barrier is not generally required.',
            'IndexK128norm/RoPE and LongKV512RoPE are different values despite common tail64. Same sourceC/S expression may share table service only with matching row/position/epochs and ownership; it does not permit reusing rotated key coordinates. C2 source C/S may differ from current query/SWA positions.',
            'IndexKey uses native DynamicQuant then FP16 scaleCast;Query custom quant rounds scale toFP16 BEFORE quantization. Never replace key quant by query quant solely from shared128 width or finalFP16scale. Loaded native key quant rounding/FTZ/newM remains a gate.',
            'A full out-of-place512 long result needs unchanged448 prefix as well as rotated64tail. A tail-only64 bank cannot be passed to unchanged single contiguous512 ScatterNdUpdateSk. A segmented writer may read prefix from old latent and tail from its bank,extending old prefix lifetime;direct RoPE-cache writer may read old latent and leave it unmodified without a fullrotated bank.',
            'Direct long-cache production and IndexWk are readonly forks of old BF16 latent when new cachewriter/slot/destination-disjoint/post-rotated-latent-observer gates hold. Keep original rotate rounding and defined invalid/duplicate-slot behavior. Source direct-producer idea is preexisting; new scope prices HC deadline/version-granularity alternatives,not a new missing primitive.',
            'An alternative saves the OLD64tail before in-place long rotation and lets a new segmented IndexWk read unchanged448prefix plus oldtail. It needs a new two-source GEMM/access contract or full adapter and original accumulation order;ordinary contiguous512 IndexWk cannot read this split version unchanged. Copy readiness may delay both branches.',
            'QLI reads IndexK/FP16scale;SMLA compressed branch reads longKV. In a valid disjoint layout,QLI may proceed before long publication completes while SMLA waits actual demanded long rows. Neither allN publication nor all-index completion is universally necessary;the models use a chosen full-demand frontier and are not minimal row pipelines.',
            'SourceC/S lifetime extends through every IndexRoPE and LongRoPE reader;old latent through IndexWk and any direct/segmented writer. CurrentHCrawX still through coeff/Postresidual,pre4 to nextFFN;C2ring future readers/writers are separate. New stream/capture/storage reuse requires last reader AND queued writer retirement.'
        ],all_runtime_pointer_epoch_numeric_or_fusion_gates_closed=False,new_device_work=False,overall_complete=False))
    bygeom=defaultdict(list);diagbygeom=defaultdict(list);diagnostics=[];querycounts=Counter();latentcounts=Counter();qbefore=0;idxrows=[]
    for r in prof['query_records']:
        s=r['stages'];m=int(s['QB']['definition']['outputShapes_text'].strip('"').split(',')[0]);g=int(s['SWA_NORM']['definition']['inputShapes_text'].strip('"').split(',')[0]);bygeom[m,g].append(r['recorded_positive_SWA_wait_window_ns']);querycounts[m,g]+=1;qbefore+=r['query_RoPE_before_IndexQ_TASK']is True
        d=s['QUERY_ROPE']['TASK']['endNs']-s['QUERY_ROPE']['TASK']['startNs'];cap=min(d,r['recorded_positive_SWA_wait_window_ns']);diagbygeom[m,g].append(cap)
        diagnostics.append(dict(database_index=r['database_index'],forward_API_rowid=r['forward_API_rowid'],block=r['source_block_ordinal'],QB_end_ns=s['QB']['TASK']['endNs'],SWA_end_ns=s['SWA_STORE']['TASK']['endNs'],RoPE_recorded_duration_ns=d,fixed_service_join_hiding_reference_ceiling_ns=cap))
    for r in prof['latent_records']:
        key=r['database_index'],r['forward_API_rowid'],r['source_block_ordinal'];p=idx[key];s=r['consumers'];Nrows=int(s['LATENT_NORM']['definition']['inputShapes_text'].strip('"').split(',')[0]);latentcounts[r['mode'],Nrows]+=1
        idxrows.append(dict(key=list(key),mode=r['mode'],IndexWK_to_LongRoPE_control_span_ns=r['recorded_LongRoPE_start_minus_IndexWK_end_ns'],
            observed_index_stores_before_long=r['recorded_LongRoPE_after_both_index_stores'],observed_long_store_before_QLI=s['LONG_STORE']['TASK']['endNs']<=p['stages']['QLI']['TASK']['startNs'],
            actual_requested_cache_rows_or_disjoint_physical_ranges_bound=False))
    write(O/'hc-rope-latent-observed-frontiers-v140.json',dict(query_geometry=[dict(M=m,global_SWA_rows=g,frontiers=len(v),positive_SWA_wait_intervals=sum(x>0 for x in v),minimum_ns=min(v),median_ns=statistics.median(v),maximum_ns=max(v),fixed_service_join_hiding_reference_median_ns=statistics.median(diagbygeom[m,g]),fixed_service_join_hiding_reference_max_ns=max(diagbygeom[m,g]))for (m,g),v in sorted(bygeom.items())],
        fixed_service_join_diagnostics=diagnostics,fixed_service_join_scope='Reference three-node scheduling component only: fixed QB/SWA completion and fixed recorded RoPE duration,readycorrectCS,no added competition/queue/ownershipcost. max(QB,SWA)+R minus max(QB+R,SWA)=min(R,max(SWA-QB,0)). Not a physical upper bound on fusion/end-to-end gain;do not sum acrosslayers/cards/iterations.',
        Query_RoPE_before_IndexQ_frontiers=qbefore,all_query_RoPE_after_QB_and_SWA=all(r['query_RoPE_TASK_starts_after_QB_and_SWA_store']for r in prof['query_records']),latent_frontiers=idxrows,
        observed_intervals_not_saved_critical_time_or_isolated_services=True,new_device_work=False,overall_complete=False))
    qs=[];qclasses=Counter()
    for y,c,n,qb,sw,rope,tail in product([0,4],[0,4,16],[1,3],[1,5],[1,5,12],[1,3],[1,8]):
        for dr,ds,dc in product([0,2],[0,3],[0,5]):
            r=dict(y=y,c=c,n=n,qb=qb,sw=sw,rope=rope,tail=tail,dr=dr,ds=ds,dc=dc)
            r.update(baseline=query_time(r,False,False),HC_fork_only=query_time(r,True,False),Query_hoist_only=query_time(r,False,True),joint_ideal=query_time(r,True,True),joint_repriced=query_time(r,True,True,True));qs.append(r)
            tag='JOINT_FASTER'if r['joint_repriced']<r['baseline']else'JOINT_SLOWER'if r['joint_repriced']>r['baseline']else'TIED';qclasses[tag]+=1
    ls=[];lclasses=Counter()
    for l,w,i,g,k,q,qi,c,out,extra in product([0,4],[1,4],[1,4],[1,4],[1,5],[0,8],[0,8],[0,12],[1,3],[0,2,8]):
        r=dict(l=l,w=w,i=i,g=g,k=k,q=q,qi=qi,c=c,out=out,extra=extra);r.update(latent_times(r));ls.append(r)
        tag='READONLY_FASTER'if r['readonly_long_direct']<r['baseline']else'READONLY_SLOWER'if r['readonly_long_direct']>r['baseline']else'TIED';lclasses[tag]+=1
    write(O/'hc-rope-latent-readiness-DAGs-v140.json',dict(query_cases=qs,query_case_count=len(qs),query_classes=dict(qclasses),latent_cases=ls,latent_case_count=len(ls),latent_classes=dict(lclasses),
        query_symbols='u=(fork?y:max(y,c))+n;QB=u+qb,SWA=u+sw;oldQueryready=max(QB,SWA)+rope,newready=max(QB+rope+dr,SWA+ds);tail includes index/compression/QLI/attention/output continuation before HCcoeffjoin c+dc.',
        latent_symbols='l:latent512normready,w:IndexWk oldread,i:IndexNorm/RoPE/quant/stores,g:LongRoPE/store,k:QLI,q:mainQuery/SWAready,qi:IndexQuery/weight/metadataready,c:HCcoeff,out:SMLA/output continuation,extra:newlongwriter/resource overhead. Arbitraryunits.',
        scope='Chosen full-frontier atomic schedules,not mandatory whole-row barriers/native optimal-time. Query comparison retains source pre-index ordering and moves only queryRoPE beforeSWAwait. Latent schedules preserve old-read per chosen version and separate IndexK/Long readiness. Shared Vector/Cube/link contention actual unknown;repriced overheads hypothetical. EndpointHCPost inputready,commonPostservice and laterpre4/FFN omitted.',
        new_device_work=False,overall_complete=False))
    write(O/'hc-rope-latent-version-and-CS-witnesses-v140.json',dict(
        prefix_tail=dict(N=2,D=4,R=2,old_rows=[[1,2,3,4],[5,6,7,8]],full_rotated_rows=[[1,2,-4,3],[5,6,-8,7]],tail_only=[[-4,3],[-8,7]],
            IndexWk=[0,0,1,0],old_index=[3,7],wrong_after_inplace=[-4,-8],wrong_tail_as_contiguous_full_row=[-4,3,-8,7],
            scope='Finite90degree two-feature reference;not native512RoPE lane/rounding proof. Segmented reconstruction must preserve full prefix and tail.'),
        source_CS_lifetime=dict(original_CS=[1,0],reused_CS=[0,1],old_index_RoPE_read_done=1,queued_long_RoPE_read=3,reuse_time=2,vector=[1,0],correct_long=[1,0],wrong_long=[0,1],
            meaning='First branch completion does not retire tables read by another queued branch. Native pointers/generations not asserted.'),
        query_key_quantizer_distinction=read(O/'index-context-operator-contract-v64.json')['witnesses']['KeyScale_not_QueryScale'],
        source_current_query_order='QueryRoPE374 before_prepare_indexer_inputs376;not a new ordering gap. SWAwait492 is earlier candidate frontier.',
        native_numeric_or_actual_alias_proof=False,new_device_work=False,overall_complete=False))
    qm=[];lm=[]
    for (m,g),count in sorted(querycounts.items()):qm.append(dict(M=m,global_SWA_rows=g,frontiers=count,full_BF16_query_output=2*m*64*512,
        QB_RoPE_tail_interface_reference=4*m*64*64,separate_early_RoPE_interface_elision=0,no_postQB_RMS512=True))
    for (mode,rows),count in sorted(latentcounts.items()):lm.append(dict(mode=mode,N=rows,frontiers=count,full_outofplace_rotated_latent_content=2*rows*512,
        tail_only_rotated_content=2*rows*64,full_outofplace_unchanged_prefix_extra_value_IO=4*rows*(512-64),direct_long_cache_RoPE_tail_interface_elision=4*rows*64,
        saved_old_tail_copy_value_IO=4*rows*64,
        IndexNorm_RoPE_Quant_postnorm_interface_reference=4*rows*128+4*rows*64,IndexQuant_INT8_temporary_interface_reference=2*rows*128,
        FloatScale_cast_interface_reference=8*rows,FP16Scale_scatter_interface_reference=4*rows,
        one_shared_source_CS_pair_value_content=8*rows*64,actual_unique_index_or_long_dirty_rows_known=False))
    variants=[dict(id='CURRENT_INPLACE_LONG_AFTER_WK',contract='Preserve oldread-before-mutation,sourcewholeindex control can be relaxed only under effect/generation gates; no extra rotated bank.'),
        dict(id='FULL_OUTOFPLACE_LONG',contract='Read old latent,emit512 fullrotated output;retain448prefix andBF16round. Extra fullplane and1792N prefixread/write reference,unless other new handoff absorbs it.'),
        dict(id='TAIL_ONLY_WITH_SEGMENTED_WRITER',contract='Emit64rotated tail,read448prefixfromoldlatent innew2-inputwriter. Existingsinglecontiguous512Scattercannotconsumeitunchanged;oldprefixlifetimeextends.'),
        dict(id='DIRECT_READONLY_ROPE_TO_LONG_CACHE',contract='Newwriter readsold512latent,rotates64tailwithoriginalBF16round,storesfinalcachewithoriginalslotsemantics;no fullLrot bank ifallotherpostrotatedlatentreadersclosed. BothIndexWkandlongproducerreadonlyfork;destalias/cacheepoch/capture/numericgatesremain.'),
        dict(id='SAVE_OLD_TAIL_FOR_SEGMENTED_INDEX_WK',contract='Copyold64tail beforeinplacelongrotation;newIndexWkreadsunchanged448prefix+oldtailinoriginalK/accumulationorder. Ordinarysinglecontiguous512GEMMneedsadapterornewaccessABI. Same128Ntailbankbutadditional256NcopyvalueIO;oldtailcopyprecedesbothchosenbranches.'),
        dict(id='INDEX_NORM_ROPE_QUANT_CACHE',contract='Key128norm BF16→RoPE BF16→nativecompatibleKeyDynamicQuant/FP32scale→FP16cache-scale;directkey+scaleproducerkeepsdefinedrow/duplicate/slot/epoch policy andQLIpublication,notQueryquantizerreuse.')]
    write(O/'hc-rope-latent-version-work-and-lifetimes-v140.json',dict(query_models=qm,latent_models=lm,variants=variants,
        selected_lifetimes=['Old512latent until allIndexWk/direct/segmentedprefix readers complete;tail-onlyvariantdoesnotretireprefixatIndexWkend.',
            'SourceC/S until IndexRoPE and LongRoPE and any aliasedQuery readers complete;not firstbranchdone.',
            'IndexK andFP16scale requiredsame defined cachegeneration before QLIneededranges;LongKVbeforeSMLAneededranges. Current two stores/control is not invented tuple-atomic duplicate policy.',
            'C2ring old/future read/write retirement and HCrawX coeff/Post/pre4 remain separate;newfork doesnot close them.'],
        cache_store_content='In fixed uncompressed cache layout: Index key+scale130*U_index,Long1024*U_long. U are actual unique dirty demanded rows under defined duplicate/invalid policy,not descriptorN. Different slot/readsets preserved.',
        one_graph_credit='Choose one latent version andconsumer graph. Fullcopy/tailbank/directwriter are alternatives,not additive savings. Queryfusiontailcredit vs standalonehoist differs. IndexNorm/KeyQuant/LongRoPE/interface/CStable credits overlap andmust be unioned once;no physical HBM/wire/minimum/nativegain claim.',
        theory='SYMBOLIC_CEILING: only per-version mandatoryvalue/operationneedandvalid jointmixedCube/Vector/linkcapacityupper can bound time. Current interfaces/referencecontents are not all-algorithmphysicalwork;actualallocation/livepeak/criticalpath/lowest/MFU unbound.',
        new_device_work=False,overall_complete=False))
    write(O/'hc-rope-latent-composition-v140.json',dict(new_pairs=[['F01','F07'],['F01','F11'],['F01','F19']],new_pair_clauses=3,new_groups=0,
        group_extensions=[['HG93-HC-ATTENTION-FORK-DEADLINE','F07'],['HG93-HC-ATTENTION-FORK-DEADLINE','F11'],['HG93-HC-ATTENTION-FORK-DEADLINE','F19']],
        clauses=[dict(id='REL140-01',a='F01',b='F07',contract='Current mainQueryRoPEalreadybeforeIndexer. HCearlyY/QB/QueryRoPEcompositioncanhoistRoPEacrossSWAjoinorfuseQBepilogueonlywithfullBF16QB→tail64RoPEround/CStable/ownership/vectorcontention/currentHCcoeff/Post/pre4gates. StandalonehoistdeletesnoGMinterface;TASKwaitwindowsarenotspeedup.'),
            dict(id='REL140-02',a='F01',b='F11',contract='HCnormalizedY→C1/C2normedlatent512→IndexWk512→128→KeyNorm/RoPE/nativeDynamicQuant/FP16scale/cachestoreskeepsdistinctRMS/rounddomainsandcurrentQLIcachevisibility. Query-keyquantizersanddifferentRoPEoperandsarenotCSE;directcachefusionrequiresallslots/readers/epochs/nativeprecisionandoneinterfaceunion.'),
            dict(id='REL140-03',a='F01',b='F19',contract='ComposeHCforkwitholdlatentIndexWk/LongKVversions:inplaceWARperrangeorreadonlyfull/tail/directcachefork. Tail64bankneedsnewsegmentedconsumer/fullprefix;CS/oldprefix/readerslifetimesremain. QLIneedsIndexK+scale,SMLAneedsLongKV;currentfullindex→long→QLIcontrolnotuniversalsemanticbarrierunderproveddisjointdomains. PreserveHCX/coeff/Post/pre4/C2ring andfinite resourceprogress.')],
        all_runtime_gates_proved=False,critical_path_gain_us=None,whole_pair_or_all_subsets_closed=False,new_device_work=False,overall_complete=False))
    names=['hc-index-postcheck-result-v139p.json','candidate-relation-clauses-v139.json','hc-rope-latent-native-frontiers-v140.json','hc-sparse-eager-native-frontiers-v138.json',
        'hc-c1-native-consumers-v135.json','hc-c2-native-consumers-v136.json','F33-consumer-and-numerical-contract-v27.json','state-engram-fusion-variants-v62.json',
        'Q_B-index-epilogue-extension-v82.json','KV-publication-variants-v87.json','index-context-operator-contract-v64.json','attention-indexer-native-numerical-contract-v60.json','hc-c2-norm-state-source-contract-v137.json']
    paths=[O/n for n in names]+[Path(p['path'])for p in parts]
    write(O/'hc-rope-latent-inputs-v140.json',dict(inputs=[pin(p)for p in sorted(set(paths))],profile_first=True,
        questions=['Is main QueryRoPE actually delayed by Indexer,or by the earlier SWA aux join?',
            'What native old-latent/index/long consumers and per-head rounding/readiness constrain HC composition?',
            'Can full/tail-only/direct-readonly long versions remove WAR without silently discarding prefix,cache-read or C/S lifetime requirements?'],new_device_work=False,overall_complete=False))
    print(json.dumps(dict(status='HC_QUERY_ROPE_KEY_LATENT_COMPOSITION_REVIEWED',query_frontiers=960,latent_frontiers=96,query_DAGs=len(qs),latent_DAGs=len(ls),query_classes=dict(qclasses),latent_classes=dict(lclasses),new_scoped_pairs=3,new_device_work=False)))


if __name__=='__main__':main()
