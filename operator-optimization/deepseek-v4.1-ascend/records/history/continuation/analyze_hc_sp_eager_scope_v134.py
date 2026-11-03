"""Close the saved eager-frontier join, price typed carrier variants without speedup claims."""
from pathlib import Path
from collections import Counter,defaultdict
from statistics import median
import hashlib,json
from finalize_origin_coverage_v9 import N,O,W,read,write


def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def rows(stage):return int(stage['definition']['inputShapes_text'].strip('"').split(',')[0])


def stats(v):return dict(n=len(v),minimum=min(v),median=median(v),maximum=max(v))


def main():
    assert not(O/'REVIEW-CHECKPOINT-v134.json').exists()
    p=read(O/'hc-sp-eager-native-frontiers-v134.json');old=read(O/'hc-sp-native-frontiers-v133.json');allsp=[r for r in read(O/'communication-native-operator-population-v66.json')['operations']if r['semantic_role']=='ATTENTION_INPUT_SP_ALLGATHER']
    before={(r['database_index'],r['SP_allGather']['COMM_OP_rowid'])for r in old['records']};new={(r['database_index'],r['SP_allGather']['COMM']['COMM_OP_rowid'])for r in p['records']};total={(r['database_index'],r['COMM_OP_rowid'])for r in allsp}
    assert len(before)==3512 and len(new)==960 and not(before&new)and before|new==total and len(total)==4472
    write(O/'hc-sp-eager-coverage-reconciliation-v134.json',dict(previous_joined=3512,new_joined=960,all_saved_attention_SP=4472,remaining_in_this_saved_population=0,previous_new_keys_disjoint=True,
        new_keys=[list(k)for k in sorted(new)],scope='Allsavedv66ATTENTION_INPUT_SP_ALLGATHERrecordsnowhaveasource-scopedHC/norm/Postfrontier. Notallmodules/branches/newABIs/actualtensororcrossrankepisodeclosure.',
        new_pair_relations=0,no_native_operand_or_global_phase_promotion=True,prior_graph80_role_ambiguities_retained=True,new_device_work=False,overall_complete=False))
    bygeo=defaultdict(list);qa=Counter();guard=Counter();dur=defaultdict(lambda:defaultdict(list))
    for r in p['records']:
        n=r['SP_allGather']['COMM']['count']//5120;N=rows(r['SWA_KV']);q=rows(r['QA']);bygeo[n,N].append(r);qa[n,q]+=1
        guard['initial'if r['source_block_ordinal']==0 else'Engram'if r['Engram_positive_APIs']else'aux'if r['aux_mean']else'no_interposed_guard_record']+=1
        for name in ['HC','NORM','POST','FFN_PRE','FFN_POST']:
            t=r['stages'][name]['TASK'];dur[n,N][name].append(t['endNs']-t['startNs'])
        for name,field in [('AG','SP_allGather'),('RS','output_reduceScatter')]:dur[n,N][name].append(r[field]['COMM']['duration_ns'])
        for name in ['QA','SWA_KV']:
            t=r[name]['TASK'];dur[n,N][name].append(t['endNs']-t['startNs'])
    write(O/'hc-sp-eager-observed-timing-and-geometry-v134.json',dict(geometries=[dict(sender_rows=n,KV_reported_rows=N,uniform_gather_rows=8*n,executions=len(rs),QA_reported_rows=dict(Counter(rows(r['QA'])for r in rs)),internal_COMM_record_counts=dict(Counter(r['SP_allGather']['COMM']['internal_records']for r in rs)),native_duration_ns={k:stats(v)for k,v in dur[n,N].items()})for(n,N),rs in sorted(bygeo.items())],
        input_guard_counts=dict(guard),
        metric_scope='Originaloneprofilepopulation,nativeTASKorCOMMintervalstatisticsacrosslocalDBinstances. InternalCOMMrecordcounts193/196/207/210areretainedratherthanassuming196foreverycall;recordsinclude waits/transport,notuniversalbandwidth,exclusivecompute,criticalpathgainoracomparableoptimizationbenchmark.',
        missing_COMM_COMPUTE_TASK_is_not_zero_cost=True,no_crossrank_episode_or_client_request_promotion=True,no_native_time_clipping_to_CPU_forward_interval=True,
        all_module_MFU_lowest_or_gain_established=False,new_device_work=False,overall_complete=False))
    models=[]
    for n,N in sorted(bygeo):
        H,D,Q=5120,512,1280;plans=[]
        for name,extraC1,extraC2,includeQA in [('SWA_ONLY',False,False,False),('SWA_QA',False,False,True),('SWA_C1',True,False,False),('SWA_C1_QA',True,False,True),('SWA_C2_FLOAT',False,True,False),('SWA_C2_FLOAT_QA',False,True,True)]:
            bf=2*n*(D+(D if extraC1 else 0)+(Q if includeQA else 0));fp=8*n*D if extraC2 else 0
            plans.append(dict(name=name,BF16_value_content=bf,FLOAT_value_content=fp,total_sender_value_content=bf+fp,ratio_to_old_H_sender=(bf+fp)/(2*n*H),
                complete_all_reader_carrier_proved=False))
        models.append(dict(sender_rows=n,KV_reported_rows=N,padded_rows=8*n,old_BF16_H_sender_content=2*n*H,plans=plans,
            original_global_SWA_dense_ops_per_card=2*N*H*D,new_uniform_owner_SWA_dense_ops_per_card=2*n*H*D,
            original_C2_pair_FLOAT_dense_ops_per_source_card=4*N*H*D,new_uniform_owner_C2_pair_FLOAT_dense_ops=4*n*H*D,
            QA_cases=[dict(current_QA_reported_rows=q,executions=c,current_QA_dense_ops=2*q*H*Q,new_uniform_owner_QA_dense_ops=2*n*H*Q,extra_dense_ops=2*(n-q)*H*Q)for(n0,q),c in sorted(qa.items())if n0==n],
            logical_work_not_all_algorithm_necessary_or_HBM_or_wire=True))
    write(O/'hc-sp-eager-typed-carrier-and-padding-models-v134.json',dict(models=models,
        prior_reference='v88mixedSWA/C2N6400/6528/6668weregeometryreferences;v134adds72exacteagerSWA+twoC2FLOATsource-rolejoinsatthosesizes. Existingratio.5withoutQAisnotanewformula. NewQAroute/fullownerbudget.75appliesonlyifallremainingrawreadersclosed.',
        extra_planes='24C1BF16projectionsaredistinctsourcecallsfromSWA;nosameparameter/equalvalueCSEinferred. SWA_C1(.2)/SWA_C1_QA(.45)keepbothroleplanesunlessweight/norm/readeridentityproved. SWA_C2_FLOAT(.5)/plusQA(.75)keeptwoFLOAT512planes,notBF16substitutions.',
        all_reader_gate='QP/CP-SPmaps,rawIndexer/C1/C2/QA/rstd/otherreaders,positions/slot/history/cacheepochsandnumericsremain. IfrawHmuststillbeexchanged,projectedplanesareadditionalcontentratherthansavings.',
        padding='N6668isreportedglobalKVmatmulrowsversusuniformAG6672;QA830occurs40timesalongsideHC834. Uniformownerprojectioncomputesextra4QArowsinthosecases;trim/rowmaps/error/nativeMmustbepreserved. Countsnotclient-validtokens.',
        collective='Multipletypedcollectivesorbit-preservingpackedABIwithalignment/layout/queue/readycosts. OriginalRING-RING-NHR/largeHCCLrecordscannotbeextrapolatedfromsmallAIVTASKtimesorcontentratios.',
        actual_numeric_hardware_bound_latency_gain_or_new_implementation=False,theory_status='SYMBOLIC_CEILING',new_device_work=False,overall_complete=False))
    root=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend';parts=[]
    for rel,a,b in [('models/deepseek_v41/model.py',887,920),('models/deepseek_v41/model.py',1551,1597),('models/deepseek_v41/compressor.py',55,71),('attention/dsa_v41.py',350,382),('attention/dsa_v41.py',456,478),('attention/dsa_v41.py',506,546),('attention/context_parallel/dsa_v41_cp.py',188,196)]:
        path=root/rel;parts.append(dict(**pin(path),first_line=a,last_line=b,source=path.read_text().splitlines()[a-1:b]))
    write(O/'hc-sp-eager-operator-source-contract-v134.json',dict(parts=parts,questions=[
        'HowareCOMM-onlycallsboundwithoutinventingCOMPUTE_TASKnodes?',
        'WhichsameforwardHC/premix/PostandEngram/auxrolesarepositivelyrecorded?',
        'WhichtypedplanesandpaddingremainwhenprojectingbeforeSPacrossC1/C2/QA?'],
        facts=['SameglobalTidCPUforwardcontainmentand40orderedmodelsourcecalls,thenexactCONNECTION_IDS→TASK/COMMkeys. CPUorderisnotNPUwhole-barrierorrequest/crossrankidentityproof.',
            'SourceC1usescompressor.forward71BF16wkv/norm;C2widenshiddenBF16toFLOATthenusesseparatewkv530/wgate531beforepool/state/RoPE/cache. Do notreplaceFLOAToutputsbyBF16orassumeC1weightsequalSWA.',
            'CurrentHColdpremix/updatedPostgenerations,48positiveEngramguards/72auxreadersandallretirementrulesremain. Sourceordinaldoesnotattestloadedparameterorhistoricalpointeridentity.'],
        new_device_work=False,overall_complete=False))
    ins=[O/n for n in ['hc-sp-postcheck-result-v133p.json','candidate-relation-clauses-v133.json','candidate-pair-navigation-v133.json','hc-sp-eager-intake-v134.json','hc-sp-eager-native-frontiers-v134.json',
        'hc-sp-native-frontiers-v133.json','communication-native-operator-population-v66.json','embedding-initialization-native-chains-v67.json','mixed-SWA-C2-project-before-gather-v88.json','mixed-SWA-C2-cost-models-v88.json','hc-sp-eager-build-note-v134a.json']]+[Path(p['path'])for p in parts]
    write(O/'hc-sp-eager-inputs-v134.json',dict(inputs=[pin(x)for x in sorted(set(ins))],profile_first=True,new_device_work=False,overall_complete=False))
    print(json.dumps(dict(status='SAVED_HC_SP_FRONTIER_POPULATION_COVERED_IN_SOURCE_SCOPE',joined=4472,remaining_in_saved_population=0,source_guards=dict(guard),geometry_count=3,new_pairs=0,new_device_work=False)))


if __name__=='__main__':main()
