"""Independent postcheck repair: avoid local T shadowing the task-directory import; all v124 checks retained."""
from collections import Counter
from fractions import Fraction
from itertools import combinations,product
from pathlib import Path
import argparse,hashlib,json,math,struct
from finalize_origin_coverage_v9 import N,O,R,T,L,read,write


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');a=ap.parse_args();checks=[]
    def ck(name,v):
        assert v,name
        checks.append(name)
    def pinned(r):
        b=Path(r['path']).read_bytes()
        return len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
    for r in read(O/'sparse-index-inputs-v124.json')['inputs']:ck('input:'+r['path'],pinned(r))
    contract=read(O/'sparse-index-source-contract-v124.json')
    for part in contract['parts']:
        ck('source_pin:'+part['path'],pinned(part))
        lines=Path(part['path']).read_text().splitlines()
        for s in part['segments']:ck('source_lines:'+str((part['path'],s['first_line'])),s['source']==lines[s['first_line']-1:s['last_line']])
    ck('not_actual_duplicate_or_full_scheduler_proof',not contract['historical_native_ID_values_or_duplicate_rates_attested']and not contract['whole_source_scheduler_partition_proved'])
    cases=0
    for ids in product([-1,0,1,2,5],repeat=4):
        for position in [-1,0,1,2,5,8]:
            for ratio in [1,2]:
                visible=(position+1)//ratio;sent=(1<<31)-1
                slots=[x if 0<=x<visible else sent for x in ids]
                keys=[struct.unpack('<f',struct.pack('<I',((0x81800000-x)&0xffffffff)if x<(1<<24)else x-(1<<23)))[0]for x in slots]
                decoded=[]
                for key in sorted(keys):
                    bits=struct.unpack('<I',struct.pack('<f',key))[0]
                    x=(0x81800000-bits)&0xffffffff if bits&0x80000000 else bits+(1<<23)
                    decoded.append(-1 if x==sent else x)
                expected=[x for x in sorted(ids)if 0<=x<visible]
                assert decoded==expected+[-1]*(len(ids)-len(expected))
                assert Counter(x for x in decoded if x>=0)==Counter(expected)
                cases+=1
    ck('7500_independent_float_bit_sort_refs',cases==7500)
    seed=123;domain=[0,1,(1<<24)-1,1<<24,(1<<31)-2,(1<<31)-1]
    for _ in range(10000):
        seed=(1103515245*seed+12345)&0x7fffffff;domain.append(seed)
    encoded=[]
    for x in sorted(set(domain)):
        bits=((0x81800000-x)&0xffffffff)if x<(1<<24)else x-(1<<23)
        f=struct.unpack('<f',struct.pack('<I',bits))[0]
        assert math.isfinite(f) and f!=0
        encoded.append(f)
    ck('piecewise_full_range_boundary_sample',encoded==sorted(set(encoded)))
    # Separate request intervals,zero-query requests and row-owner order.
    window_cases=0
    for B in [1,2]:
        for qlens in product(range(4),repeat=B):
            for prefixes in product(range(5),repeat=B):
                for window in range(1,5):
                    I=8;old=[];descriptor=[];owner=[];lengths=[]
                    for request,(q,prefix)in enumerate(zip(qlens,prefixes)):
                        seq=prefix+q;start=max(prefix-window,0);visible=seq-start
                        assert visible<=I
                        desc=[start,visible];descriptor.append(desc)
                        explicit=[start+j if j<visible else -1 for j in range(I)]
                        old.extend([explicit]*q);owner.extend([request]*q);lengths.extend([visible]*q)
                    restored=[[descriptor[r][0]+j if j<descriptor[r][1]else -1 for j in range(I)]for r in owner]
                    assert restored==old and [descriptor[r][1]for r in owner]==lengths
                    assert all(len([i for i in row if i>=0])==len({i for i in row if i>=0})for row in old)
                    window_cases+=1
    ck('1680_independent_request_interval_refs',window_cases==1680)
    proof=read(O/'sparse-index-multiset-and-rounding-v124.json')
    ck('multiset_duplicate_and_offset_witness',proof['prepare_duplicate_witness']['valid_length']==2 and proof['prepare_duplicate_witness']['unique_valid_count']==1 and proof['offset_sentinel_witness']['prepared']==[1,1,2])
    ck('sink_multiplicity_witness',Fraction(2,3)!=Fraction(1,2))
    r=proof['rounding_counterexample']
    ck('BF16_round_boundary_witness',r['three_rounded_exponentials']==1.5 and r['rounded_aggregate_exponential']==1.5078125 and r['BF16_final_source_style']!=r['BF16_final_fold'])
    ck('no_native_duplicate_collapse_proof',not proof['native_bitwise_duplicate_collapse_proved'])
    costs=read(O/'sparse-index-request-reuse-and-cost-v124.json')
    ck('symbolic_not_numeric_bound',costs['theory_state']=='SYMBOLIC_CEILING' and costs['numeric_T_lower_or_throughput_upper']is None)
    for r in costs['models']:
        request_count,query_rows,index_width=r['requests'],r['query_rows'],r['index_width']
        ck('metadata_model:'+str((request_count,query_rows)),r['expanded_current_index_and_length_content_bytes']==4*query_rows*(index_width+1) and r['selected_per_request_row_plus_query_owner_content_bytes']==4*request_count*(index_width+1)+4*query_rows and r['selected_interval_plus_query_owner_content_bytes']==8*request_count+4*query_rows and not r['physical_HBM_or_allocation_credit'])
    audit=read(O/'operator-graph-coverage-audit-v124.json');g=read(O/'operator-graph-G9-v124.json')
    ck('41_49_854_1346_48',(audit['new_source_node_rows'],audit['new_source_edge_rows'],audit['node_rows'],audit['edge_rows'],audit['current_structural_pair_hints'])==(41,49,854,1346,48))
    for key,path in audit['graph_catalog'].items():ck('graph_catalog_SHA:'+key,hashlib.sha256(Path(path).read_bytes()).hexdigest()==audit['graph_sha256'][key])
    for t in g['templates']:
        nodes={n['id']for n in t['nodes']};neighbors={n:set()for n in nodes}
        ck('source_graph_endpoints:'+t['id'],len(nodes)==len(t['nodes'])and all(e['producer']in nodes and e['consumer']in nodes for e in t['edges']))
        for e in t['edges']:neighbors[e['producer']].add(e['consumer']);neighbors[e['consumer']].add(e['producer'])
        for k in [2,3,4]:
            count=0
            for ss in combinations(sorted(nodes),k):
                todo={ss[0]};seen=set()
                while todo:
                    x=todo.pop();seen.add(x);todo|=(neighbors[x]&set(ss))-seen
                count+=len(seen)==k
            ck('independent_connected_sets:'+str((t['id'],k)),count==audit['G9_connected_sets'][t['id']][str(k)])
    bind=read(O/'operator-graph-candidate-bindings-v124.json')
    ck('two_existing_relation_hints_added',bind['new_pair_hints_added']==2 and all(h['existing_conditional_clause_ids']for h in bind['structural_pair_hints']if any(l.get('graph')=='G9'for l in h['locations'])))
    if a.preflight:manifest=read(O/'checkpoint-manifest-v123.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v124.json');ck('progress_open',cp['status']=='PROGRESS'and not cp['overall_complete'])
        cov=read(O/'candidate-relation-coverage-v124.json');ck('same135_26_460',(cov['scoped_conditional_pairs'],cov['higher_order_groups'],cov['remaining_pair_relations'])==(135,26,460))
        matrix=read(O/'module-phase-branch-matrix-v124.json');ck('358_26',len(matrix['modules'])==26 and sum(len(m['branches'])for m in matrix['modules'])==358)
        prev=read(O/'operator-analysis-coverage-ledger-v123.json');new=read(O/'operator-analysis-coverage-ledger-v124.json')
        ck('original_native_uncertainty_unchanged',all(prev[k]==new[k]for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']))
        fusion=read(R/'fusion-scan.json');ck('shared_graph_pointer',fusion['per_node_edge_coverage_ledger']['summary']==str(O/'operator-graph-coverage-audit-v124.json'))
        for r in read(O/'candidate-record-index-v124.json')['model_candidates']:ck('candidate:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        ck('Task_ACTIVE_Loop_EVALUATING',read(T/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING')
        manifest=read(O/'checkpoint-manifest-v124.json')
    for r in manifest['artifacts']:ck('manifest:'+r['path'],pinned(r))
    result=dict(status='PASS',preflight=a.preflight,checks=len(checks),multiset_cases=cases,interval_broadcast_cases=window_cases,
        key_boundary_sample_values=len(domain),source_graph_rows=[41,49],partial_graph_rows=[854,1346],structural_hints=48,
        new_device_work=False,new_native_promotions=0,overall_complete=False)
    if a.preflight:write(O/'sparse-index-preflight-v124b.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
