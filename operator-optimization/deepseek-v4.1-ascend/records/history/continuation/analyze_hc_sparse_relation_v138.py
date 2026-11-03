"""HC with sparse-index publication and early SWA/CMP attention joins."""
from pathlib import Path
from collections import Counter
from itertools import product
import hashlib,json,math
from finalize_origin_coverage_v9 import N,O,W,read,write


def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def finish(r,fork,split,extra=0,coeff_delay=0):
    release=r['y']if fork else max(r['y'],r['c']);s=release+r['ds'];i=max(release+r['di'],r['k'])
    attention=max(s+r['a'],i)+r['b']+extra if split else max(s,i)+r['a']+r['b']
    return max(r['c']+coeff_delay,attention+r['o'])


def state(scores,values):
    if not scores:return dict(m=None,l=0,a=0)
    m=max(scores);es=[math.exp(s-m)for s in scores]
    return dict(m=m,l=sum(es),a=sum(e*v for e,v in zip(es,values)))


def merge(x,y):
    if not x['l']:return y
    if not y['l']:return x
    m=max(x['m'],y['m']);sx=math.exp(x['m']-m);sy=math.exp(y['m']-m)
    return dict(m=m,l=sx*x['l']+sy*y['l'],a=sx*x['a']+sy*y['a'])


def main():
    assert not(O/'REVIEW-CHECKPOINT-v138.json').exists()
    native=read(O/'hc-sparse-eager-native-frontiers-v138.json');views=read(O/'hc-sparse-shared-topk-views-v138.json');root=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend';parts=[]
    for rel,a,b in [('models/deepseek_v41/model.py',887,920),('attention/dsa_v41.py',567,611),('attention/dsa_v41.py',613,683),('attention/dsa_v41.py',700,716),
        ('attention/context_parallel/dsa_v41_cp.py',194,227),('attention/context_parallel/dsa_cp.py',77,91),('models/deepseek_v41/indexer.py',272,298)]:
        p=root/rel;parts.append(dict(**pin(p),first_line=a,last_line=b,source=p.read_text().splitlines()[a-1:b]))
    old=read(O/'attention-indexer-native-numerical-contract-v60.json')
    for r in old['parts']:
        if 'sparse_flash_mla'in r['path']and r['source']:
            p=Path(r['path']);parts.append(dict(**pin(p),first_line=r['first_line'],last_line=r['last_line'],source=p.read_text().splitlines()[r['first_line']-1:r['last_line']]))
    write(O/'hc-sparse-source-and-join-contract-v138.json',dict(parts=parts,
        reused_answers=['v60SWA-CMPsplit(m,l,a)/BF16exp/PV/sink/noreturnedLSE','v83QLIscore-not-attentionscore/partialnormalized-output-informationloss','v93HCcoeff-YforkandPost/pre4deadlines','v124orderedmultiset/invalidprefix/physicalgatherordering'],
        facts=[
            'All960positiveeagerSMLAstages use original64heads/512BF16 output, inverse64-tailRoPE then head-layout transpose before unchanged restore_tp_heads/all_to_all/Oprojection/reduceScatter/HCPost continuation. The full callback/crossrank episode, actual gather ID payload and new partial-state ABI are not attested by source/CID.',
            'SWA data path needs current Q/position/SWA cache, not compressed selected indices. Current integrated SMLA includes both SWA and compressed work and online softmax/sink/PV in one kernel; earlier SWA is a new schedule with compatible partial state, not deletion of SWA/compression work.',
            'Current fallback caller explicitly requests return_softmax_lse=False; adapter/source constructs empty LSE. Two normalized BF16 outputs cannot reconstruct a joint softmax. FP32(m,l,a) is a chosen real-reference state format, not native parity proof; original BF16 exponential/PV/online-update/tile and output round/order remain separate gates.',
            'Partial attention partitions must preserve original occurrence multiset, masks/sentinel policy and physical emitted-row order. SWA and CMP occurrences need not be disjoint physical token addresses; deduplication changes weights. Include sink exactly once, with compatible empty-partition semantics.',
            'Per-head normalization weights vary. Merging after cross-head Oprojection generally loses required head contributions, even if one carries per-head masses. Merge before head mixing, or retain sufficient head-resolved projected contributions under a new priced ABI. Head exchange is a permutation only under correct row/head/epoch maps; precision/quantization boundaries cannot be commuted automatically.',
            '192sourceQLI/prepare/queryquantchains cover72MODE3,24MODE1,96MODE2. Current source20 candidate bank is copied once perforward and used by source24/28/32/36;720later source-positive CP206 slice APIs bind non-index source layers to shared topk view branches. Candidate membership can be prepared after producer epoch before a later consumer HCY, but current consumer query/score/TopK still depends on that layer input. Current source SMLA completion does not close the later shared TopK readers.',
            'Candidate bitmap/sorted-membership preparation is pure metadata only under same logical row/visible universe/generation/readers. It is not reuse of QLI scores, which change with query and use different INT8/DEQF16/ReLU/headreduce math from SMLA BF16 attention scores.',
            'QLI-finalprepare fusion must retain visibility filtering, chronological valid prefix, duplicate occurrences, -1 suffix, candidate source output and all shared bank readers. Partial ready publication needs native release/acquire/per-row ownership and finite progress; current whole kernel completion does not expose early rows.',
            'One HC coefficient/Y/attention graph is required for costs. Current full4H coeff/postcomb and Postresidual remain, pre4 still meets nextFFN. Earlier attention may end before coeffready and yield zero Post improvement; shared Cube/Vector/link contention and partial-state service can worsen it.',
            'Persistent SWA waiting for QLI while occupying all resources needed by QLI can have a wait cycle. Separate launch boundaries, verified reservation/yield/preemption or a jointly scheduled kernel must establish finite progress; none is assumed from fusion alone.'
        ],actual_loaded_native_fusion_partial_numeric_or_hardware_gain_proved=False,new_device_work=False,overall_complete=False))
    ideal=[];classes=Counter();witnesses={};priced=[];pricedclasses=Counter()
    for y,c,ds,di,k,a,b,o in product([0,4,12],[0,4,12],[0,3],[0,3,10],[0,8,20],[1,5],[1,5],[0,3]):
        r=dict(y=y,c=c,ds=ds,di=di,k=k,a=a,b=b,o=o)
        vals=dict(baseline=finish(r,False,False),HC_fork_only=finish(r,True,False),attention_split_only=finish(r,False,True),joint=finish(r,True,True));r.update(vals)
        hg=vals['baseline']-vals['HC_fork_only'];sg=vals['baseline']-vals['attention_split_only'];jg=vals['baseline']-vals['joint'];r.update(HC_gain=hg,split_gain=sg,joint_gain=jg)
        tag='SYNERGY'if jg>hg+sg else'OVERLAPPING_CREDIT'if jg<hg+sg else'ADDITIVE_IN_THIS_CASE';classes[tag]+=1;witnesses.setdefault(tag,r);ideal.append(r)
        for extra,cdelay in product([0,2,8],[0,5]):
            end=finish(r,True,True,extra,cdelay);cl='JOINT_FASTER'if end<r['baseline']else'JOINT_SLOWER'if end>r['baseline']else'TIED';pricedclasses[cl]+=1
            priced.append(dict(inputs={x:r[x]for x in ['y','c','ds','di','k','a','b','o']},extra_partial_state_launch_merge_service=extra,HC_coefficient_contention_delay=cdelay,baseline=r['baseline'],joint=end,classification=cl))
    write(O/'hc-sparse-fork-and-join-DAG-models-v138.json',dict(ideal_cases=ideal,ideal_case_count=len(ideal),nonadditivity_classes=dict(classes),witnesses=witnesses,repriced_cases=priced,repriced_case_count=len(priced),repriced_classes=dict(pricedclasses),
        symbols='y:externalYready,c:currentHCcoeffready,ds:Y-to-Q/SWAready,di:Y-to-CMPindex/cache ready,k:externalmetadata/candidate/historydeadline,a:SWAservice,b:CMPservice,o:inverseRoPE/headexchange/Oprojection/RS continuation. All hypothetical arbitrary units.',
        domain='Chosen same-service schedule with one full demand frontier per stage and otherwise noncontending resources. Baseline monolithic HC releases Y at max(y,c); fork releases at y. Integrated attention=max(s,i)+a+b; split/resume=max(s+a,i)+b. Endpoint is HCPost INPUT readiness, with common unchanged HCPost service omitted; later pre4/nextFFN not priced. Neither is actual native service decomposition. Repricing adds explicitly assumed state/launch/merge and coefficient contention delays.',
        fixed_fork_ideal_SWA_overlap_gain='min(a,max(i-s,0)); Postgain further clamped by coeffdeadline, not additive to independent HCfork gain.',
        no_actual_overlap_speedup_or_lowest_time_proved=True,new_device_work=False,overall_complete=False))
    # Reference real-valued statistics: preserve every occurrence; empty branches explicit.
    sm=[]
    for na,nb,cut,withsink in product(range(4),range(4),range(3),[False,True]):
        # Repeated logical IDs within a plane keep identical scores/values.
        idsA=[j%max(1,cut+1)for j in range(na)];idsB=[j%max(1,cut+1)for j in range(nb)]
        sa=[-1+0.5*j for j in idsA];sb=[0.25+0.5*j for j in idsB];va=[j%2 for j in idsA];vb=[1-j%2 for j in idsB]
        aa=state(sa,va);bb=state(sb,vb);mm=merge(aa,bb)
        if withsink:mm=merge(mm,state([0],[0]))
        full=state(sa+sb+([0]if withsink else[]),va+vb+([0]if withsink else[]))
        sm.append(dict(scoresA=sa,scoresB=sb,valuesA=va,valuesB=vb,logical_occurrencesA=idsA,logical_occurrencesB=idsB,sink_once=withsink,
            merged=mm,direct=full,output=mm['a']/mm['l']if mm['l']else None))
    write(O/'hc-sparse-partial-state-information-witnesses-v138.json',dict(real_reference_cases=sm,case_count=len(sm),
        normalized_output_information_loss=read(O/'attention-joint-interface-models-v83.json')['split_normalized_output_information_loss_witness'],
        head_mixing=dict(partition_A_mass=[1,3],partition_B_mass=[1,1],O_weights=[1,1],case1_A_normalized=[1,0],case2_A_normalized=[0,1],B_normalized=[0,0],
            both_A_projected=1,both_B_projected=0,same_per_head_masses=True,correct_final_projected_case1=0.5,correct_final_projected_case2=0.75,
            conclusion='Even head masses plus only a cross-head projected partial result lose the needed head-resolved contributions. Not a frozen-weight counterexample.'),
        sink_once=dict(SWA_mass=1,CMP_mass=1,sink_mass=1,CMP_numerator=1,correct=1/3,wrong_sink_each_partition=1/4),
        duplicate_occurrence=dict(logical_occurrences=[0,1,1],scores=[0,0,0],values=[0,1,1],original=2/3,wrong_set_deduplicated=1/2),
        liveness=dict(resources=['AIC_SLOT'],persistent_SWA_holds_all_AIC_while_waiting_QLI=True,
            wait_edges=[['QLI_DONE','SWA_RELEASE_SLOT'],['SWA_RELEASE_SLOT','QLI_ACQUIRE_SLOT'],['QLI_ACQUIRE_SLOT','QLI_DONE']],
            finite_progress_proved=False,scope='Hypothetical no-preemption exclusive resource model; actual device residency/preemption not asserted.'),
        native_precision_or_new_kernel_proof=False,new_device_work=False,overall_complete=False))
    geoms=Counter();models=[]
    for r in native['records']:
        m=int(r['stages']['SMLA']['definition']['outputShapes_text'].strip('"').split(',')[0]);geoms[m]+=1
    for m,count in sorted(geoms.items()):
        models.append(dict(M=m,frontiers=count,heads=64,D=512,current_BF16_output_value_content=2*m*64*512,
            one_declared_FP32_m_l_a_state_value_content=4*m*64*(512+2),two_state_value_contents=8*m*64*(512+2),
            one_state_private_write_plus_read_value_IO=8*m*64*(512+2),BF16_normalized_plus_FLOAT_LSE_value_content=m*64*(2*512+4),
            existing_returned_LSE_value_content=0,reference_format_not_mandatory_extra_allocation_or_HBM=True))
    orderchecks=sum(c['candidate_publish_TASK_ends_before_consumer_HC_TASK_starts']for f in native['forward_candidate_publish_frontiers']for c in f['consumers'])
    write(O/'hc-sparse-publication-and-work-cuts-v138.json',dict(models=models,candidate_publish_frontiers=24,source20_to_later_mode2_frontiers=96,source_topk_groups=views['producer_groups'],later_source_positive_TopK_view_APIs=views['source_positive_view_APIs'],
        observed_candidate_publish_TASK_ends_before_consumer_HC_TASK_starts=orderchecks,
        candidate_hoist='Under exact logical query-row/metadata/candidate epoch, build membership once after source20 candidate publish, before consumerHC24/28/32/36 as ready resources allow. This does not require future consumer hidden input. Keep current query-dependent QLI/TopK and chronological/multiset outputs.',
        group_shared_banks='Source2→blocks2..7,8→8..13,14→14..19,20→20..23,24→24..27,28→28..31,32→32..35,36→36..39. Native role/source ordering is not actual bank pointer/ID/crossrank episode proof.',
        partial_state_cut='A declared FP32(m,l,a512) private handoff can increase live value content and movement compared with BF16 output; original online workspaces might be reused only under actual lifetime/address gates. Do not count all state content as necessarily extra allocation or physical traffic.',
        before_vs_after_projection='Merge compatible per-head state before ordinary head mixing, or retain head-resolved projected state with per-head coefficients and original native precision. A normal mixed Oprojection/HCPost output alone is insufficient. Head permutation/exchange may move before merge only with all state/head/row/epoch routing and new datatype/collective/capture ABI costs.',
        one_resource_cut='Choose one HC/QLI/SMLA/transport graph, one cold/warm grant and shared Cube/Vector/GM/link capacity upper. Logical value content is not physical servicedomain necessity. No allmodule numeric MFU/lowest time; no independent sum of HC,prepare,queue,attention or bank credits.',
        new_device_work=False,overall_complete=False))
    write(O/'hc-sparse-composition-clauses-v138.json',dict(new_pairs=[['F01','F12'],['F01','F13']],new_pair_clauses=2,new_groups=0,
        group_extensions=[['HG93-HC-ATTENTION-FORK-DEADLINE','F12'],['HG93-HC-ATTENTION-FORK-DEADLINE','F13']],
        clauses=[dict(id='REL138-01',a='F01',b='F12',contract='HCconsumerY/currentquery-dependentQLI/TopK differs from already-produced shared candidate membership. Hoist pure membership after exactsource20epoch if possible;keep current QLI math/multiset/chronological prefix/stablebank/readers. Producer-prepare fusion/partial publication needs new ABI/progress,not early wholekernel value invention;currentHCcoeff/Post/pre4 remain.'),
            dict(id='REL138-02',a='F01',b='F13',contract='HCearlyY can expose SWA attention while CMP/QLI waits, but compare one HCcoeff-Y-SWA-CMP-Post graph. Keep compatible per-head partial normalizers/numerators,sink once,original occurrence order and BF16 exp/PV/output/rope/nativeprojection boundaries. Merge before head mixing or carry priced head-resolved state. Current Post may be coeffbound;state/communication/contention/liveness costs can erase gain.')],
        all_runtime_gates_proved=False,critical_path_gain_us=None,whole_pair_or_all_subsets_closed=False,new_device_work=False,overall_complete=False))
    names=['hc-c2-norm-state-postcheck-result-v137p.json','candidate-relation-clauses-v137.json','hc-sparse-eager-native-frontiers-v138.json','hc-sparse-shared-topk-views-v138.json','hc-sp-eager-native-frontiers-v134.json',
        'attention-indexer-native-numerical-contract-v60.json','attention-indexer-fusion-reorganization-variants-v60.json','attention-joint-interface-models-v83.json','index_attention-distinct-score-proof-v83.json',
        'sparse-index-source-contract-v124.json','sparse-index-profile-scope-v124.json','HC-attention-deadline-and-precision-models-v93.json','hc-c2-joint-graph-and-clauses-v136.json']
    paths=[O/n for n in names]+[Path(p['path'])for p in parts]
    write(O/'hc-sparse-inputs-v138.json',dict(inputs=[pin(p)for p in sorted(set(paths))],profile_first=True,
        questions=['How do native eager SMLA/QLI/prepare/candidate-bank roles connect to HC/SP frontiers?',
            'When can candidate membership hoist before consumer HCY, while current query scores must remain?',
            'Can HC early-Y and early SWA/CMP partial attention compose without losing normalization/head state, precision, finite progress or coefficient/Post deadlines?'],new_device_work=False,overall_complete=False))
    print(json.dumps(dict(status='HC_SPARSE_COMPOSITION_CONDITIONALLY_REVIEWED',frontiers=960,TASKs=3480,ideal_DAG_cases=len(ideal),repriced_cases=len(priced),nonadditivity_classes=dict(classes),repriced_classes=dict(pricedclasses),real_reference_cases=len(sm),new_scoped_pairs=2,new_device_work=False)))


if __name__=='__main__':main()
