"""Independent archive checks, fanout bounds, descriptor reexpansion and payload arithmetic."""
from pathlib import Path
from fractions import Fraction
from itertools import combinations,product
from collections import Counter
import argparse,gzip,hashlib,json
from finalize_origin_coverage_v9 import N,O,T,L,read,write
from saved_tensor_archive_v122 import load_saved_tensor_list

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');a=ap.parse_args();checks=[]
    def ck(n,v):assert v,n;checks.append(n)
    def pinned(r):
        b=Path(r['path']).read_bytes();return len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256']
    for r in read(O/'route-fanout-inputs-v122.json')['inputs']:ck('input:'+r['path'],pinned(r))
    raw=read(O/'route-fanout-saved-records-v122.json');ck('1392_saved_not_device',raw['snapshot_records']==1392 and raw['selected_formal_cohorts']==87 and not raw['new_device_work']and not raw['Torch_imported'])
    ck('boundedCPU_read',raw['read_bytes']==35632142 and raw['elapsed_CPU_process_seconds']<40)
    ib=(N/'route-snapshot-source-v122/tensor-index.json').read_bytes();index=json.loads(ib);ck('source_index_SHA',hashlib.sha256(ib).hexdigest()==raw['INDEX_SHA'])
    routes=read(N/'MC2-operator-source-v56/saved-observer/EP-routes.json')
    with gzip.open(N/'MC2-V1-observer-source-v119/decoded-records.jsonl.gz','rt')as f:
        source_records={r['index']:r for r in (json.loads(line)for line in f)}
    oldmatrix={(c['role'],c['layer'],c['ordinal']):c['sender_receiver_assignment_matrix']for c in routes['cohorts']}
    groups={}
    for i,r in enumerate(raw['records']):
        ident=(r['role'],r['layer'],r['observer_ordinal']);groups.setdefault(ident,[]).append(r)
        ck('original_matrix:'+str(i),r['assignments_by_destination']==oldmatrix[ident][r['sender_EP_slot']])
        ent=index[r['source_index']];ck('archive_SHA:'+str(i),r['tensor_sha256']==ent['tensor_sha256']and r['metadata_sha256']==ent['metadata_sha256'])
        src=source_records[r['source_index']]
        ck('placement_capacity:'+str(i),src['placement_strategy']=='linear'and src['redundant_experts']==0 and src['group_binding_complete']and src['group_ranks']==list(range(16))and src['capacity_max_output_size']==65536 and src['receiver_count_sum']<=65536 and not src['receiver_count_sum_above_capacity_argument'])
        A=r['assignments_by_destination'];D=r['token_destinations_by_destination'];V=r['valid_recorded_rows'];K=r['K']
        ck('rowbounds:'+str(i),sum(A)==V*K and all(-(-x//min(K,r['E_local']))<=d<=min(x,V)for x,d in zip(A,D)))
        h={int(k):v for k,v in r['per_packet_fanout_histogram'].items()};ck('fanout_conservation:'+str(i),sum(h.values())==sum(D)and sum(k*v for k,v in h.items())==sum(A)and r['duplicate_expert_active_rows']==0)
    ck('87_EP16_groups',len(groups)==87 and all(len(v)==16 and {r['sender_EP_slot']for r in v}==set(range(16))for v in groups.values()))
    ck('no_new_socket',not(N/'ssh166-route-v122.sock').exists())
    # Independently compute reference fanout on the locally retained real archive sample.
    local=N/'route-snapshot-source-v122';meta=read(local/'sample-metadata.json')
    ts=load_saved_tensor_list((local/'sample.pt').read_bytes());vs=dict(zip(meta['payload_names'],ts));sample=read(O/'route-snapshot-source-index-v122.json')['sample_record']
    flat=vs['topk_ids'].values();mask=vs['x_active_mask'].values();K=sample['K'];E=sample['E_local']
    assignments=[(row,k,flat[row*K+k])for row in range(sample['M'])if mask[row]for k in range(K)]
    hist=Counter(e for _,_,e in assignments);pairs={(row,e//E)for row,_,e in assignments}
    ck('sample_epoch_counts_hist',vs['observer_ordinal'].values()==[sample['ordinal']]and vs['expert_counts'].values()==sample['receiver_expert_counts']and dict(hist)=={int(k):v for k,v in sample['sender_expert_histogram'].items()})
    ck('sample_pair_bounds',len(pairs)<=len(assignments)and len(assignments)==K*sample['valid_rows'])
    expected_owner=[e%E if e//E==sample['ep_rank']else -1 for e in range(sample['E_global'])]
    ck('sample_contiguous_owner_map',vs['expert_map'].values()==vs['ascend_expert_map'].values()==expected_owner)
    # Two source rows, unique TopK3 among6 experts, independent old expert order and packet order.
    patterns=list(combinations(range(6),3));cases=0
    for r0,r1 in product(patterns,repeat=2):
        for maskbits in range(4):
            old=[(row,k,e)for row,picked in enumerate([r0,r1])if maskbits&(1<<row)for k,e in enumerate(picked)]
            old.sort(key=lambda x:(x[2],x[0]if maskbits%2 else-x[0],x[1]))
            packets={}
            for ordinal,(row,k,e)in enumerate(old):packets.setdefault((row,e//2),[]).append((ordinal,e%2,k))
            expanded={ordinal:(row,k,(dest*2+localexpert))for(row,dest),desc in reversed(list(packets.items()))for ordinal,localexpert,k in desc}
            for capacity in [0,1,4,6]:
                assert [expanded[i]for i in sorted(expanded)if i<capacity]==old[:capacity]
                cases+=1
    ck('6400_descriptor_clip_refs',cases==6400)
    models=read(O/'route-fanout-packet-and-return-models-v122.json');ck('selected5216',models['H']==5120 and models['selected_packet_stride']==5120+32+64)
    for g in models['groups']:
        A=g['A_remote'];D=g['D_remote'];count=g['observer_cohorts']*15360;base=A*5152;new=D*5216+count;ret=A*10240
        ck('role:'+g['role'],g['existing_dispatch_record_payload_bytes']==base and g['selected_dispatch_and_extra_counts_bytes']==new and g['unchanged_BF16_return_payload_bytes']==ret)
        ck('fractions:'+g['role'],Fraction(g['selected_dispatch_reduction_fraction'])==Fraction(base-new,base)and Fraction(g['selected_dispatch_plus_return_reduction_fraction'])==Fraction(base-new,base+ret))
        ck('no_speedup:'+g['role'],not g['actual_network_bytes_or_latency_measured'])
    target=next(g for g in models['groups']if g['role']=='target');draft=next(g for g in models['groups']if g['role']=='draft')
    ck('different_role_outcomes',target['logical_dispatch_byte_delta_saved']==1714345440 and draft['logical_dispatch_byte_delta_saved']==-133440)
    lm=models['layout_and_count_plan_models'];ck('12_layout_countplans',len(lm)==12)
    for r in lm:
        g=target if r['role']=='target'else draft;expected=(g['existing_dispatch_record_payload_bytes']-g['D_remote']*r['record_stride']-r['extra_D_count_content_bytes'])
        ck('layout:'+str((r['role'],r['layout'],r['extra_D_count_content_bytes'])),Fraction(r['reduction_fraction'])==Fraction(expected,g['existing_dispatch_record_payload_bytes']))
    comp=[r for r in lm if r['role']=='draft'and r['layout']=='packed_4B_descriptor_frame_epoch'];ck('not_universal_draft_rejection',any(Fraction(r['reduction_fraction'])>0 for r in comp)and any(Fraction(r['reduction_fraction'])<0 for r in comp))
    graph=read(O/'route-fanout-local-graph-map-v122.json');nodes=set(graph['new_nodes']);ck('explicit_graph_endpoints',len(nodes)==13 and all(e['a']in nodes and e['b']in nodes for e in graph['edges']))
    remaining=set(nodes)
    while remaining:
        ready={n for n in remaining if not any(e['b']==n and e['a']in remaining for e in graph['edges'])};assert ready;remaining-=ready
    ck('DAG_and_no_observed_nodes',graph['new_observed_native_or_global_catalog_nodes']==0)
    legality=read(O/'route-fanout-variants-and-legality-v122.json');ck('no_quantized_payload_attestation',not read(O/'route-fanout-source-contract-v122.json')['native_new_ABI_or_quantized_byte_equality_attested'])
    theory=read(O/'route-fanout-theory-and-measurement-v122.json')
    ck('logical_payload_not_physical_ceiling',theory['theory_state']=='SYMBOLIC_CEILING'and theory['T_lower_numeric']is None and theory['throughput_upper_numeric']is None and not theory['physical_cut_work_and_capacity_upper_bounds_bound'])
    witness=legality['packed_equality_counterexample'];ck('label_not_byte_equality',witness['old_assignment_payloads_hex'][0]!=witness['old_assignment_payloads_hex'][1])
    if a.preflight:manifest=read(O/'checkpoint-manifest-v121.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v122.json');ck('progress_open',cp['status']=='PROGRESS'and not cp['overall_complete'])
        cov=read(O/'candidate-relation-coverage-v122.json');ck('135_26_460',(cov['scoped_conditional_pairs'],cov['higher_order_groups'],cov['remaining_pair_relations'])==(135,26,460))
        matrix=read(O/'module-phase-branch-matrix-v122.json');ck('356_26',len(matrix['modules'])==26 and sum(len(m['branches'])for m in matrix['modules'])==356)
        old=read(O/'operator-analysis-coverage-ledger-v121.json');new=read(O/'operator-analysis-coverage-ledger-v122.json')
        for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']:ck('native_unchanged:'+k,old[k]==new[k])
        for r in read(O/'candidate-record-index-v122.json')['model_candidates']:ck('candidate:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        ck('Task_active_Loop_evaluating',read(T/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING')
        manifest=read(O/'checkpoint-manifest-v122.json')
    for r in manifest['artifacts']:ck('manifest:'+r['path'],pinned(r))
    result=dict(status='PASS',checks=len(checks),preflight=a.preflight,saved_snapshot_records=1392,independent_descriptor_clip_cases=cases,
        packet_count_layout_models=12,new_device_work=False,new_native_promotions=0,overall_complete=False)
    if a.preflight:write(O/'route-fanout-preflight-v122.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()
