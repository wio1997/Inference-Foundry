"""Independent binary/fanout/placement composition checks; no target execution."""
from collections import Counter,defaultdict
from itertools import combinations,product
from pathlib import Path
import argparse,gzip,hashlib,json,struct
from finalize_origin_coverage_v9 import N,O,T,L,read,write


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args();checks=[]
    def ck(name,v):
        assert v,name
        checks.append(name)
    def pinned(r):
        b=Path(r['path']).read_bytes()
        return len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
    def popcount(bits):
        return bin(bits).count('1')
    for r in read(O/'route-placement-inputs-v123.json')['inputs']:
        ck('input:'+r['path'],pinned(r))
    m=read(O/'route-hyperedges-source-index-v123.json')
    ck('CPU_only_finished',not m['Torch_imported'] and not m['new_device_work'] and not m['new_server_data_or_source_files_written'] and m['records']==1392 and m['read_bytes']==35632142)
    ck('SSH_closed',not(N/'ssh166-route-v123.sock').exists())
    compressed=(N/'route-hypergraph-source-v123/active-routes.bin.gz').read_bytes();b=gzip.decompress(compressed)
    ck('exact_saved_export',len(b)==7327280 and hashlib.sha256(b).hexdigest()==m['packed_sha256'] and hashlib.sha256(compressed).hexdigest()==m['compressed_sha256'])
    ck('magic',b[:8]==b'WIOEP123')
    raw=read(O/'route-fanout-saved-records-v122.json');old={r['source_index']:r for r in raw['records']}
    source_index=read(N/'route-snapshot-source-v122/tensor-index.json')
    with gzip.open(N/'MC2-V1-observer-source-v119/decoded-records.jsonl.gz','rt')as f:
        source_records={r['index']:r for r in (json.loads(line)for line in f)}
    cohorts=defaultdict(list);active=Counter();offset=8
    for j,rec in enumerate(m['record_index']):
        ck('offset:'+str(j),rec['offset']==offset)
        index,K,E,M,V=struct.unpack_from('<IHHII',b,offset);offset+=16
        ck('header:'+str(j),(index,K,E,M,V)==(rec['source_index'],old[index]['K'],old[index]['E_local'],old[index]['M'],old[index]['valid_recorded_rows']))
        rowstruct=struct.Struct('<I'+'H'*K);rows=[];A=[0]*16;D=[0]*16;hist=Counter()
        for _ in range(V):
            row=rowstruct.unpack_from(b,offset);offset+=rowstruct.size;rows.append(row)
            ck('row:'+str((index,row[0])),0<=row[0]<M and len(set(row[1:]))==K and all(0<=e<16*E for e in row[1:]))
            owners=[e//E for e in row[1:]];bits=0
            for e,d in zip(row[1:],owners):
                A[d]+=1;bits|=1<<d;hist[e]+=1
            for d in range(16):
                D[d]+=bool(bits&(1<<d))
        src=source_records[index]
        ck('original_fanout:'+str(index),A==old[index]['assignments_by_destination'] and D==old[index]['token_destinations_by_destination'])
        ck('source_histogram:'+str(index),dict(hist)=={int(k):v for k,v in src['sender_expert_histogram'].items()})
        ent=source_index[index]
        ck('source_file_SHA_and_owners:'+str(index),rec['tensor_sha256']==ent['tensor_sha256'] and rec['metadata_sha256']==ent['metadata_sha256'] and rec['owner_maps_match_contiguous'])
        ck('record_end:'+str(index),offset==rec['offset']+rec['bytes'] and len(set(r[0]for r in rows))==V)
        active[rec['role']]+=V
        cohorts[(rec['role'],rec['layer'],rec['observer_ordinal'])].append(dict(rec,rows=rows,E=E,K=K))
    ck('export_exact_domain',offset==len(b) and active=={'target':456400,'draft':260} and len(cohorts)==87 and all(len(c)==16 for c in cohorts.values()))
    maps=read(O/'route-placement-fixed-mappings-v123.json')['mappings']
    lookup={(r['role'],r['layer'],r['variant']):r for r in maps}
    ck('215_mappings',len(lookup)==len(maps)==215)
    for r in maps:
        E=r['E_local'];owner=r['owner_by_logical_expert'];physical=r['physical_ID_by_logical_expert']
        ck('bijection:'+str((r['role'],r['layer'],r['variant'])),sorted(physical)==list(range(16*E)) and Counter(owner)=={rank:E for rank in range(16)} and all(physical[e]//E==owner[e]for e in range(16*E)))
        lists=r['device_expert_logical_lists']
        ck('paired_weight_slot_labels:'+str((r['role'],r['layer'],r['variant'])),all(lists[physical[e]//E][physical[e]%E]==e for e in range(16*E)))
        ck('first_cohort_fit:'+str((r['role'],r['layer'],r['variant'])),r['training_ordinal']==min(k[2]for k in cohorts if k[:2]==(r['role'],r['layer'])) and r['trained_on_first_formal_cohort_only'])
    comparisons=read(O/'route-placement-cohort-comparisons-v123.json')['results']
    ck('435_comparisons',len(comparisons)==435)
    for r in comparisons:
        key=(r['role'],r['layer'],r['observer_ordinal']);mapping=lookup[(r['role'],r['layer'],r['variant'])]
        E=mapping['E_local'];physical=mapping['physical_ID_by_logical_expert'];loads=[0]*16;freq=[0]*(16*E)
        A_remote=D_remote=A_cross=D_cross=D_all=pair_score=V=0
        for rec in cohorts[key]:
            sender=rec['sender_EP_slot']
            for row in rec['rows']:
                dest=[physical[e]//E for e in row[1:]];bits=0;V+=1
                for e,d in zip(row[1:],dest):
                    freq[e]+=1;loads[d]+=1;A_remote+=d!=sender;A_cross+=d//8!=sender//8;bits|=1<<d
                D_all+=popcount(bits);D_remote+=popcount(bits)-bool(bits&(1<<sender))
                D_cross+=popcount(bits&(0xff00 if sender<8 else 0xff))
                pair_score+=sum(dest[i]==dest[j]for i in range(len(dest))for j in range(i))
        prefix=str((key,r['variant']))
        ck('independent_counts:'+prefix,(A_remote,D_remote,A_cross,D_cross,D_all,pair_score,loads)==(r['A_remote'],r['D_remote'],r['cross_TP8_assignment_references'],r['cross_TP8_token_destination_references'],r['token_destinations_all'],r['within_destination_expert_pair_score'],r['receiver_assignment_rows']))
        ck('work_and_C:'+prefix,V==r['active_recorded_rows'] and sum(freq)==r['assignments_all'] and max(loads)<=65536 and max(loads)>=max(max(freq),-(-sum(freq)//16))==r['count_lower_reference'])
        baseline=r['original_linear'];old_bytes=baseline['A_remote']*15392;new_bytes=D_remote*5216+15360+A_remote*10240
        ck('joint_identity:'+prefix,old_bytes-new_bytes==r['joint_vs_original_linear_old_format_logical_byte_delta']==r['placement_only_logical_byte_delta']+r['packet_after_placement_logical_byte_delta'])
        if r['variant'].endswith('within_TP8'):
            ck('half_assignment_only:'+prefix,all(physical[e]//E//8==e//E//8 for e in range(16*E)) and A_cross==baseline['cross_TP8_assignment_references'])
    aggs=read(O/'route-placement-aggregate-tradeoffs-v123.json')['aggregates']
    for a in aggs:
        rows=[r for r in comparisons if (r['role'],r['sample_partition'],r['variant'])==(a['role'],a['sample_partition'],a['variant'])]
        ck('aggregation:'+str((a['role'],a['sample_partition'],a['variant'])),a['cohorts']==len(rows) and all(a[k]==sum(r[k]for r in rows)for k in ['A_remote','D_remote','joint_vs_original_linear_old_format_logical_byte_delta','naive_independent_delta']))
    held={(a['role'],a['variant']):a for a in aggs if a['sample_partition']=='later_saved_formal'}
    ck('target_observed_tradeoff',held['target','count_balanced']['improved_receiver_max']==40 and held['target','count_balanced']['more_D_remote_cohorts']==40 and held['target','co_selection_capacity']['worse_receiver_max']==40 and held['target','co_selection_capacity']['fewer_D_remote_cohorts']==40)
    ck('not_additive_in_all_40_target_cases',all(held['target',v]['naive_addition_wrong_cohorts']==40 for v in ['count_balanced','co_selection_capacity','count_balanced_within_TP8','co_selection_within_TP8']))
    # Exhaustive six-expert,three-owner,two-slot reference. Evaluate actual sets,not affinity scores.
    experts=set(range(6));patterns=list(combinations(range(6),3));cases=nonadditive=0
    def toy(rows,owner):
        A=D=0
        for sender,es in enumerate(rows):
            ds=[owner[e]for e in es];A+=sum(d!=sender for d in ds);D+=len(set(ds)-{sender})
        return A,D
    baseline=[e//2 for e in range(6)]
    for group0 in combinations(range(6),2):
        rem=experts-set(group0)
        for group1 in combinations(sorted(rem),2):
            owner=[2]*6
            for e in group0:owner[e]=0
            for e in group1:owner[e]=1
            for rows in product(patterns,repeat=2):
                A0,D0=toy(rows,baseline);Ap,Dp=toy(rows,owner);count=72
                delta=(A0*15392)-(Dp*5216+count+Ap*10240)
                assert delta==(A0-Ap)*15392+(Ap*5152-Dp*5216-count)
                nonadditive+=delta!=((A0-Ap)*15392+(A0*5152-D0*5216-count));cases+=1
    ck('36000_exhaustive_compositions',cases==36000 and nonadditive>0)
    witness=read(O/'route-placement-nonadditivity-and-surrogate-v123.json')['pair_surrogate_counterexample']
    aa=witness['destination_occupancies_a'];bb=witness['destination_occupancies_b']
    ck('pair_surrogate_is_not_hyperedge_objective',sum(x*(x-1)//2 for x in aa)==sum(x*(x-1)//2 for x in bb)==3 and len(aa)!=len(bb) and sum(aa)==sum(bb)==6)
    graph=read(O/'route-placement-legality-and-graph-v123.json')['local_joint_graph'];nodes=set(graph['nodes'])
    ck('17_local_graph_nodes',len(nodes)==len(graph['nodes'])==17 and all(e['a']in nodes and e['b']in nodes for e in graph['edges']))
    pending=set(nodes)
    while pending:
        ready={n for n in pending if not any(e['b']==n and e['a']in pending for e in graph['edges'])}
        assert ready,'candidate graph cycle'
        pending-=ready
    ck('local_graph_DAG_not_observed',graph['state']=='CONDITIONAL_CANDIDATE_NOT_OBSERVED_RUNTIME')
    if args.preflight:
        manifest=read(O/'checkpoint-manifest-v122.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v123.json');ck('progress_and_open',cp['status']=='PROGRESS' and not cp['overall_complete'])
        cov=read(O/'candidate-relation-coverage-v123.json')
        ck('unchanged_pairs',(cov['scoped_conditional_pairs'],cov['higher_order_groups'],cov['remaining_pair_relations'])==(135,26,460))
        matrix=read(O/'module-phase-branch-matrix-v123.json');ck('unchanged_branches',len(matrix['modules'])==26 and sum(len(m['branches'])for m in matrix['modules'])==356)
        a=read(O/'operator-analysis-coverage-ledger-v122.json');z=read(O/'operator-analysis-coverage-ledger-v123.json')
        ck('original_native_uncertainty_unchanged',all(a[k]==z[k]for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']))
        for r in read(O/'candidate-record-index-v123.json')['model_candidates']:
            ck('candidate:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        ck('task_loop_active',read(T/'index.json')['status']=='ACTIVE' and read(L/'state.json')['status']=='EVALUATING')
        manifest=read(O/'checkpoint-manifest-v123.json')
    for r in manifest['artifacts']:ck('manifest:'+r['path'],pinned(r))
    result=dict(status='PASS',preflight=args.preflight,checks=len(checks),
        saved_records=1392,active_recorded_rows=456660,mappings=215,cohort_variant_comparisons=435,
        exhaustive_small_composition_cases=cases,nonadditive_small_cases=nonadditive,
        new_device_work=False,new_native_promotions=0,overall_complete=False)
    if args.preflight:write(O/'route-placement-preflight-v123.json',result)
    print(json.dumps(result))


if __name__=='__main__':
    main()
