"""Independent raw scope/CID checks for all remaining eager HC/SP frontiers."""
from pathlib import Path
from collections import defaultdict,Counter
from statistics import median
import argparse,hashlib,json,sqlite3
from finalize_origin_coverage_v9 import N,O,T,L,read,write


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args();checks=[]
    def ck(label,ok):
        assert ok,label
        checks.append(label)
    def pin(p):
        b=Path(p['path']).read_bytes();return len(b)==p['bytes']and hashlib.sha256(b).hexdigest()==p['sha256']
    for p in read(O/'hc-sp-eager-inputs-v134.json')['inputs']:ck('input:'+p['path'],pin(p))
    for p in read(O/'hc-sp-eager-operator-source-contract-v134.json')['parts']:ck('source:'+str((p['path'],p['first_line'])),pin(p)and Path(p['path']).read_text().splitlines()[p['first_line']-1:p['last_line']]==p['source'])
    prof=read(O/'hc-sp-eager-native-frontiers-v134.json');bydb=defaultdict(list);rootmap={(r['database_index'],r['forward_API']['api_rowid']):r for r in prof['forward_roots']};seen_task={};seen_comm=set();native_counts=Counter();guards=Counter();geoms=Counter();qa_counts=Counter();dur=defaultdict(lambda:defaultdict(list));internal=Counter()
    for r in prof['records']:bydb[r['database_index']].append(r)
    for ref in prof['databases']:
        di=ref['database_index'];ck('rawpin:'+str(di),pin(ref['raw_database']));ck('cachepin:'+str(di),pin(ref['cached_native']))
        db=sqlite3.connect(Path(ref['raw_database']['path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;strings={r['id']:r['value']for r in db.execute('SELECT *FROM STRING_IDS')};cid=defaultdict(set)
        for x in db.execute('SELECT *FROM CONNECTION_IDS'):cid[x['id']].add(x['connectionId'])
        frame_set={(r['id'],r['stackDepth'],r['stack'])for r in db.execute('SELECT *FROM PYTORCH_CALLCHAINS')}
        def API(a,fs=None):
            rr=db.execute('SELECT *FROM PYTORCH_API WHERE rowid=?',(a['api_rowid'],)).fetchone();ck('API:'+str((di,a['api_rowid'])),all(rr[k]==a[k]for k in rr.keys())and strings[rr['name']]==a['name_text'])
            for f in fs or[]:ck('frame:'+str((di,f['id'],f['stackDepth'])),(f['id'],f['stackDepth'],f['stack'])in frame_set and strings[f['stack']]==f['value']and f['id']==a['callchainId'])
        def stage(s):
            t,d,a=s['TASK'],s['definition'],s['API'];key=di,t['task_record_rowid']
            if key in seen_task:ck('repeat_stage:'+str(key),seen_task[key]==s);return
            seen_task[key]=s;API(a,s['selected_frames']);rr=db.execute('SELECT *FROM TASK WHERE rowid=?',(key[1],)).fetchone();dd=db.execute('SELECT rowid AS compute_record_rowid,*FROM COMPUTE_TASK_INFO WHERE globalTaskId=?',(t['globalTaskId'],)).fetchone()
            ck('TASK_CID:'+str(key),all(rr[k]==t[k]for k in rr.keys())and t['connectionId']in cid[a['connectionId']]and a['api_rowid']in t['native_linked_API_rowids'])
            for k,v in d.items():ck('COMPUTE:'+str((key,k)),(strings[dd[k[:-5]]]if k.endswith('_text')else dd[k])==v)
        def comm(s):
            a,g=s['API'],s['COMM'];key=di,g['COMM_OP_rowid'];assert key not in seen_comm;seen_comm.add(key);API(a,s['selected_frames']);rr=db.execute('SELECT *FROM COMMUNICATION_OP WHERE rowid=?',(key[1],)).fetchone()
            ck('COMM_CID:'+str(key),all(rr[k]==g[k]for k in ['count','rankSize','connectionId','startNs','endNs'])and g['connectionId']in cid[a['connectionId']]and strings[rr['groupName']]==g['group_name']and g['dtype']=='BFP16'and rr['dataType']==11 and g['TASK_rowid']is None)
        def inside(a,b):return a['globalTid']==b['globalTid']and int(b['startNs'])<=int(a['startNs'])<=int(a['endNs'])<=int(b['endNs'])
        localroots={k:v for k,v in rootmap.items()if k[0]==di}
        for k,root in localroots.items():
            API(root['forward_API'],root['source_frames']);API(root['host_execute_API']);ck('forward_marker:'+str(k),inside(root['forward_API'],root['host_execute_API'])and root['forward_API']['name_text']=='forward'and root['host_execute_API']['name_text'].startswith('wio_hostdiag::execute_model/')and not root['actual_client_step_crossrank_episode_or_parameter_identity_bound'])
            rs=sorted([r for r in bydb[di]if r['forward_API_rowid']==k[1]],key=lambda r:r['source_block_ordinal']);ck('40_source_layers:'+str(k),[r['source_block_ordinal']for r in rs]==list(range(40)))
            init=root['target_HC_initialization'];ck('initial_HC_scope',len(init['stages'])==5 and init['local_rows']==rs[0]['SP_allGather']['COMM']['count']//5120)
            for st in init['stages']:
                rr=db.execute('SELECT *FROM TASK WHERE rowid=?',(st['TASK_rowid'],)).fetchone();dd=db.execute('SELECT *FROM COMPUTE_TASK_INFO WHERE globalTaskId=?',(rr['globalTaskId'],)).fetchone();ck('initial_TASK',all(rr[f]==st[f]for f in ['startNs','endNs','streamId'])and strings[dd['opType']]==st['operator'])
            for j,r in enumerate(rs):
                st=r['stages'];order=[st['HC']['API'],st['NORM']['API'],r['SP_allGather']['API'],r['output_reduceScatter']['API'],st['POST']['API'],st['FFN_PRE']['API'],st['FFN_POST']['API']]
                ck('CPU_order:'+str((k,j)),all(inside(a,root['forward_API'])for a in order)and all(int(a['endNs'])<=int(b['startNs'])for a,b in zip(order,order[1:])))
                for s in st.values():stage(s)
                for s,line in [(st['HC'],889),(st['NORM'],896),(st['POST'],902),(st['FFN_PRE'],905),(st['FFN_POST'],919)]:ck('source_core_role',any('/models/deepseek_v41/model.py('+str(line)+'): forward'in f['value']for f in s['selected_frames']))
                comm(r['SP_allGather']);comm(r['output_reduceScatter']);g=r['SP_allGather']['COMM'];n=g['count']//5120;N=int(r['SWA_KV']['definition']['inputShapes_text'].strip('"').split(',')[0]);q=int(r['QA']['definition']['inputShapes_text'].strip('"').split(',')[0]);geoms[n,N]+=1;qa_counts[n,q]+=1;internal[g['internal_records']]+=1
                ck('HC_shape',st['HC']['definition']['inputShapes_text']==f'"{n},4,5120;24,20480;3;24;{n},4"'and st['HC']['definition']['inputDataTypes_text']=='DT_BF16;FLOAT;FLOAT;FLOAT;FLOAT'and st['NORM']['definition']['inputShapes_text']==f'"{n},5120;5120"')
                for name,line,dtype,weight in [('QA',467,'DT_BF16;DT_BF16','1280,5120'),('SWA_KV',475,'DT_BF16;DT_BF16','512,5120'),('C1',71,'DT_BF16;DT_BF16','512,5120'),('C2_KV',530,'FLOAT;FLOAT','512,5120'),('C2_SCORE',531,'FLOAT;FLOAT','512,5120')]:
                    s=r[name]
                    if s:
                        stage(s);native_counts[name]+=1;d=s['definition'];prefix='/models/deepseek_v41/compressor.py'if name=='C1'else'/attention/dsa_v41.py'
                        ck('projection_source_and_type',d['opType_text']=='MatMulV2'and d['inputShapes_text'].endswith(';'+weight+'"')and d['inputDataTypes_text']==dtype and any(prefix+'('+str(line)+'):'in f['value']for f in s['selected_frames'])and int(order[2]['endNs'])<=int(s['API']['startNs'])<=int(s['API']['endNs'])<=int(order[3]['startNs']))
                ck('C1_C2_source_indices',bool(r['C1'])==(j==20)and bool(r['C2_KV'])==bool(r['C2_SCORE'])==(j in [2,8,14]))
                if j:ck('old_pre_Post_generation',r['previous_FFN_Pre']==rs[j-1]['stages']['FFN_PRE']and r['previous_FFN_Post']==rs[j-1]['stages']['FFN_POST'])
                else:ck('initial_generation',r['previous_FFN_Pre']is None and r['previous_FFN_Post']is None)
                ck('Engram_aux_guards',bool(r['Engram_positive_APIs'])==(j in [1,14])and bool(r['aux_mean'])==bool(r['aux_allGather'])==(j in [37,38,39]))
                for e in r['Engram_positive_APIs']:
                    API(e['API'],e['selected_frames']);ck('positive_Engram_source',any('/model.py(1586): forward'in f['value']for f in e['selected_frames'])and bool(e['native_TASK_rowids']))
                if r['aux_mean']:stage(r['aux_mean']);comm(r['aux_allGather']);native_counts['aux']+=1
                native_counts['Engram']+=bool(r['Engram_positive_APIs']);guards['initial'if j==0 else'Engram'if r['Engram_positive_APIs']else'aux'if r['aux_mean']else'no_interposed_guard_record']+=1
                for name in ['HC','NORM','POST','FFN_PRE','FFN_POST']:
                    t=st[name]['TASK'];dur[n,N][name].append(t['endNs']-t['startNs'])
                for name,field in [('AG','SP_allGather'),('RS','output_reduceScatter')]:dur[n,N][name].append(r[field]['COMM']['duration_ns'])
                for name in ['QA','SWA_KV']:
                    t=r[name]['TASK'];dur[n,N][name].append(t['endNs']-t['startNs'])
                ck('no_tensor_promotion',not r['exact_tensor_parameter_premix_values_or_crossrank_episode_bound'])
        db.close()
    ck('960_24',len(prof['records'])==960 and len(rootmap)==24 and dict(native_counts)==prof['source_role_counts']==dict(QA=960,SWA_KV=960,C1=24,C2_KV=72,C2_SCORE=72,Engram=48,aux=72))
    ck('reported_rows',geoms=={(800,6400):320,(816,6528):320,(834,6668):320}and qa_counts=={(800,800):320,(816,816):320,(834,834):280,(834,830):40})
    ck('nonconstant_internal_COMM_records',internal=={196:935,207:19,210:4,193:2})
    cov=read(O/'hc-sp-eager-coverage-reconciliation-v134.json');old=read(O/'hc-sp-native-frontiers-v133.json');allsp=[r for r in read(O/'communication-native-operator-population-v66.json')['operations']if r['semantic_role']=='ATTENTION_INPUT_SP_ALLGATHER'];a={(r['database_index'],r['SP_allGather']['COMM_OP_rowid'])for r in old['records']};b={(r['database_index'],r['SP_allGather']['COMM']['COMM_OP_rowid'])for r in prof['records']};c={(r['database_index'],r['COMM_OP_rowid'])for r in allsp}
    ck('complete_only_saved_SP_join',not(a&b)and a|b==c and len(c)==cov['all_saved_attention_SP']==4472 and cov['remaining_in_this_saved_population']==0 and not cov['overall_complete'])
    observed=read(O/'hc-sp-eager-observed-timing-and-geometry-v134.json');ck('source_guards',dict(guards)==observed['input_guard_counts']==dict(initial=24,Engram=48,aux=72,no_interposed_guard_record=816))
    for r in observed['geometries']:
        k=r['sender_rows'],r['KV_reported_rows']
        for name,v in dur[k].items():ck('native_interval_stats:'+str((k,name)),r['native_duration_ns'][name]==dict(n=len(v),minimum=min(v),median=median(v),maximum=max(v)))
    for r in read(O/'hc-sp-eager-typed-carrier-and-padding-models-v134.json')['models']:
        n,N=r['sender_rows'],r['KV_reported_rows'];ck('dense_reference',r['old_BF16_H_sender_content']==10240*n and r['original_global_SWA_dense_ops_per_card']==2*N*5120*512 and r['new_uniform_owner_SWA_dense_ops_per_card']==2*n*5120*512 and r['original_C2_pair_FLOAT_dense_ops_per_source_card']==4*N*5120*512)
        perrow=[1024,3584,2048,4608,5120,7680]
        for p,size in zip(r['plans'],perrow):ck('typed_content:'+str((n,p['name'])),p['total_sender_value_content']==size*n and p['BF16_value_content']+p['FLOAT_value_content']==size*n and p['ratio_to_old_H_sender']==size/10240 and not p['complete_all_reader_carrier_proved'])
        for q in r['QA_cases']:ck('QA_padding_work',q['new_uniform_owner_QA_dense_ops']==2*n*5120*1280 and q['extra_dense_ops']==2*(n-q['current_QA_reported_rows'])*5120*1280 and q['extra_dense_ops']>=0)
    if args.preflight:manifest=read(O/'checkpoint-manifest-v133.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v134.json');ck('open_progress',cp['status']=='PROGRESS'and not cp['overall_complete']and cp['whole_modules_closed']==0)
        rel=read(O/'candidate-relation-clauses-v134.json');ck('142_26_453',len(rel['clauses'])==rel['clause_count']==142 and len(rel['higher_order_groups'])==26 and read(O/'candidate-relation-coverage-v134.json')['remaining_pair_relations']==453)
        m=read(O/'module-phase-branch-matrix-v134.json');ck('358_26',len(m['modules'])==26 and sum(len(x['branches'])for x in m['modules'])==358)
        a=read(O/'operator-analysis-coverage-ledger-v133.json');b=read(O/'operator-analysis-coverage-ledger-v134.json');ck('native_gaps_retained',all(a[k]==b[k]for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']))
        for r in read(O/'candidate-record-index-v134.json')['model_candidates']:ck('master:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        ck('task_loop_active',read(T/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING');manifest=read(O/'checkpoint-manifest-v134.json')
    for p in manifest['artifacts']:ck('manifest:'+p['path'],pin(p))
    out=dict(status='PASS',checks=len(checks),preflight=args.preflight,new_eager_frontiers=960,forward_roots=24,total_saved_SP_frontiers=4472,remaining_saved_SP_frontier_joins=0,new_pair_relations=0,new_native_promotions=0,new_device_work=False,overall_complete=False)
    if args.preflight:write(O/'hc-sp-eager-preflight-v134.json',out)
    print(json.dumps(out))


if __name__=='__main__':main()
