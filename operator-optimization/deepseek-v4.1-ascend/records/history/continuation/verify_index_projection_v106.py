"""Independent saved-native,typed resource,precision and source-ready graph checks."""
from pathlib import Path
from collections import Counter,defaultdict
from fractions import Fraction as F
from itertools import product
import argparse,hashlib,json,math,struct,statistics
from finalize_origin_coverage_v9 import N,O,R,read,write

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');a=parser.parse_args();checks=[]
    def check(name,value):
        assert value,name
        checks.append(name)
    def pinned(r):
        p=Path(r['path']);return p.stat().st_size==r['bytes']and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
    inp=read(O/'index-projection-inputs-v106.json')
    for r in inp['inputs']:check('input:'+r['path'],pinned(r))
    src=read(O/'index-projection-source-contract-v106.json');texts={}
    for r in src['parts']:
        check('source:'+r['path'],pinned(r));s=Path(r['path']).read_text();texts[Path(r['path']).name]=s
        check('segment:'+str(r['first_line'])+Path(r['path']).name,s.splitlines()[r['first_line']-1:r['last_line']]==r['source'])
    dsa=texts['dsa_v41.py'];indexer=texts['indexer.py'];vec=texts['quant_lightning_indexer_v2_service_vector_arch22.h']
    check('source_same_hidden_expression','attn.wq_a(query_hidden)'in dsa and '_prepare_indexer_inputs(attn, query_hidden, qr,'in dsa and 'indexer.project_weights(hidden_states)'in dsa)
    check('query_then_weight_event',dsa.index('query = indexer.project_query(qr)')<dsa.index('weights_start = main_stream.record_event()')<dsa.index('weights = indexer.project_weights(hidden_states)'))
    check('required_rounds_source','self._output(self.weights_proj, hidden_states).float() * self.weights_scale'in indexer and 'weights = weights.to(torch.float16)'in indexer and 'AscendC::Mul(inWeightsUb, inWeightsUb, inQScaleUb'in vec)
    native=read(O/'index-projection-native-frontiers-v106.json');bydb=defaultdict(list)
    for r in native['frontiers']:bydb[r['database_index']].append(r)
    keyset=set();overlap=0;roots=0;cohort_values=defaultdict(list)
    cpaths=[Path(r['path'])for r in inp['inputs']if Path(r['path']).name.startswith('db-')]
    for p in cpaths:
        db=int(p.stem[3:]);d=read(p);tasks={t['task_record_rowid']:t for t in d['TASK_records']};defs={x['globalTaskId']:x for x in d['COMPUTE_definitions']}
        for r in bydb[db]:
            s=r['stages'];assert len(s)==9
            for x in s.values():
                t=tasks[x['task_record_rowid']];de=defs[t['globalTaskId']]
                assert (x['startNs'],x['endNs'],x['streamId'])==(t['startNs'],t['endNs'],t['streamId'])
                assert x['operator']==de['opType_text']and all(de[k]==v for k,v in x['signature'].items())
                keyset.add((db,x['task_record_rowid']))
            assert s['IndexQ']['endNs']<=s['IndexWeight']['startNs']
            assert s['IndexWeight']['streamId']==s['IndexQuant']['streamId']and s['WeightScale']['endNs']<=s['IndexQuant']['startNs']
            assert s['IndexWeight']['startNs']<s['IndexRoPE']['endNs']and s['IndexRoPE']['startNs']<s['IndexWeight']['endNs'];overlap+=1;roots+=r['root_bound']
        for c in native['projection_cohorts']:
            for i,tid in c['TASK_keys']:
                if i!=db:continue
                t=tasks[tid];de=defs[t['globalTaskId']];assert de['inputDataTypes_text']==c['input_types']and de['outputDataTypes_text']==c['output_types']
                assert de['outputShapes_text']=='"'+str(c['M'])+(','+'4096'if c['role']=='IndexQ'else','+'32')+'"'
                cohort_values[(c['role'],c['M'])].append(t['endNs']-t['startNs'])
    check('640_5760_frontiers',len(native['frontiers'])==640 and len(keyset)==native['reused_unique_TASKs']==5760)
    check('already_overlap_and_shared_aux_stream',overlap==native['weight_rope_overlap_frontiers']==640 and native['weight_and_IndexQuant_same_stream_frontiers']==640)
    check('512root128unbound',roots==512 and native['phase_unbound']==128)
    check('five_shapes_ten_typed_cohorts',native['observed_M']==[3,800,816,830,834]and len(native['projection_cohorts'])==10)
    for c in native['projection_cohorts']:
        v=cohort_values[(c['role'],c['M'])];f=2*c['M']*(1280*4096 if c['role']=='IndexQ'else 5120*32)
        check('native_cohort:'+str((c['role'],c['M'])),len(v)==c['execution_count']and statistics.median(v)==c['native_TASK_ns']['median']and f==c['full_output_dense_reference_OPS']and c['dense_reference_OPS_per_native_TASK_second']==f/(statistics.median(v)*1e-9))
    cost=read(O/'index-projection-work-and-resource-models-v106.json');old=read(O/'index-context-fusion-costs-v64.json')['IndexWeight_models']
    for r in cost['models']:
        m=r['M'];fq=2*m*1280*4096;fw=2*m*5120*32
        cut=(5242880+327680-1572864+3072-1)//3072
        check('typed_model:'+str(m),r['IndexQ_classical_INT8_ops']==fq and r['IndexWeight_classical_BF16_ops']==fw and
            F(r['IndexQ_useful_compute_cycle_cut'])==F(fq,24*16384)and F(r['IndexWeight_useful_compute_cycle_cut'])==F(fw,24*8192)and
            r['combined_fixedB_weight_ingress_cycle_cut']==cut==1302 and r['combined_cycle_cut']==math.ceil(max(F(fq,24*16384),F(fw,24*8192),cut)))
        check('reused_v64_budget:'+str(m),r['CastMulCast_interface_budget_from_v64_bytes']==next(x['CastMulCast_only_interface_budget_bytes']for x in old if x['M']==m)and r['projection_epilogue_interface_budget_from_v64_bytes']==20*m*32)
    for r in cost['upstream_panel_models']:
        f=sum(2*x['M']*5120*x['N']for x in r['groups']);w=sum(2*5120*x['N']for x in r['groups'])
        naive=sum(2*24*5120*x['N']for x in r['groups'])
        check('ragged_panel:'+str(r['C1_source_branch']),r['new_useful_classical_BF16_ops']==f and r['distinct_weight_bytes']==w and r['global_rectangular_reference_ops']==naive and F(r['rectangular_to_ragged_work_ratio'])==F(naive,f)and r['fixedB_weight_ingress_cycle_cut']==math.ceil((w-1572864)/3072)and r['IndexWeight_not_in_QA_RMS'])
    check('no_physical_MFU_or_native_clock',not cost['actual_native_path_numeric_wallclock_MFU_or_speedup_proved']and not cost['all_algorithms_scope'])
    rnd=read(O/'index-projection-rounding-and-readiness-v106.json');w=rnd['intermediate_FP16_counterexample']
    check('subnormal_FP16_intermediate_counter',F(w['BF16_projection'])==F(3,2**20)and F(w['BF16_projection'])*F(1,64)==F(3,4)*F(1,2**24)and
        w['required_first_FP16_weight']==2**-24 and w['required_FP16_product']==2**-12 and w['product_if_first_FP16_round_dropped']==3*2**-14)
    check('valid_BF16_query_amax',(struct.unpack('>I',struct.pack('>f',w['valid_query_amax']))[0]&65535)==0 and w['valid_query_amax']/127==4096)
    safe=0
    half=lambda v:struct.unpack('e',struct.pack('e',v))[0]
    for e in range(-8,21):
        for sig in [1,1.25,1.5,1.9921875]:
            y=sig*2**e/64
            if 2**-14<=abs(y)<=65504 and half(y)==y:
                for q in [2**-24,2**-10,1,32]:
                    if abs(y*q)<=65504:assert half(half(y)*q)==half(y*q);safe+=1
    check('448_sufficient_domain_reference_cases',safe==rnd['finite_safe_domain_cases']==448)
    check('256_ideal_DAG_examples',len(rnd['DAG_examples'])==256)
    for r in rnd['DAG_examples']:
        q,z,w,t=r['IndexQ'],r['RoPE'],r['Weight'],r['independent_tail']
        assert r['old_finish']==q+max(z,w)+t and r['early_weight_start_finish']==max(q+z,w)+t and r['early_start_late_join_finish']==max(q+z+t,w)
        assert r['old_finish']>=r['early_weight_start_finish']>=r['early_start_late_join_finish']
    split=read(O/'index-weight-partition-and-precision-v106.json');check('30_splitK_reference_graphs',len(split['models'])==30)
    for r in split['models']:
        m,p=r['M'],r['K_partitions'];assert r['K_per_partition']==5120//p and r['same_distinct_weight_bytes']==327680
        assert r['materialized_FLOAT_partial_write_read_reference_bytes']==(8*p*m*32 if p>1 else 0)and r['chosen_FLOAT_merge_add_count']==(p-1)*m*32 and r['no_partial_BF16_round']
    fp=lambda x:struct.unpack('f',struct.pack('f',x))[0]
    check('splitK_FP32_reassociation_counter',fp(fp(2**25-2**25)+1)==1 and fp(2**25+fp(-2**25+1))==0 and not split['actual_native_split_K_or_speedup_proved'])
    comp=read(O/'index-projection-composition-v106.json');check('one_real_source_edge',len(comp['clauses'])==1 and{comp['clauses'][0]['a'],comp['clauses'][0]['b']}=={'F05','F10'}and comp['keep_old_distinct_IndexQ_Weight_input_contract'])
    if a.preflight:manifest=read(O/'checkpoint-manifest-v105.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v106.json');check('PROGRESSnotwhole',cp['status']=='PROGRESS'and not cp['overall_complete'])
        cov=read(O/'candidate-relation-coverage-v106.json');check('132_25_463',(cov['scoped_conditional_pairs'],cov['higher_order_groups'],cov['remaining_pair_relations'])==(132,25,463))
        rel=read(O/'candidate-relation-clauses-v106.json');check('old_relation_retained_and_group_extended',any(r['id']=='REL83-02'for r in rel['clauses'])and 'F10'in next(g['members']for g in rel['higher_order_groups']if g['id']=='HG75-Q-KV-JOIN'))
        bm=read(O/'module-phase-branch-matrix-v106.json');check('353rows26semantic',sum(len(m['branches'])for m in bm['modules'])==353 and all(m.get('branch_schema')=='SEMANTIC_BRANCHES_V1'for m in bm['modules']))
        old=read(O/'operator-analysis-coverage-ledger-v105.json');new=read(O/'operator-analysis-coverage-ledger-v106.json')
        for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']:check('nativeunchanged:'+k,old[k]==new[k])
        for r in read(O/'candidate-record-index-v106.json')['model_candidates']:check('master:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        manifest=read(O/'checkpoint-manifest-v106.json')
    for r in manifest['artifacts']:check('manifest:'+r['path'],pinned(r))
    result=dict(status='PASS',checks=len(checks),preflight=a.preflight,frontiers=640,reused_TASKs=5760,shape_cases=5,safe_rounding_cases=safe,
        DAG_examples=256,splitK_graphs=30,new_native_promotions=0,overall_complete=False)
    if a.preflight:write(O/'index-projection-analysis-preflight-v106.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()
