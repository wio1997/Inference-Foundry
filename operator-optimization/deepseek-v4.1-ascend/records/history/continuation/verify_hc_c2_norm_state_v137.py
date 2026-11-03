"""Check direct native wrapper evidence and independent row/capacity proofs."""
from pathlib import Path
from collections import defaultdict,Counter
from zipfile import ZipFile
import ast,argparse,json,hashlib,sqlite3,math,struct
from finalize_origin_coverage_v9 import N,O,T,L,read,write


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args();checks=[]
    def ck(label,ok):
        assert ok,label
        checks.append(label)
    def pin(r):
        b=Path(r['path']).read_bytes();return len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256']
    for r in read(O/'hc-c2-norm-state-inputs-v137.json')['inputs']:ck('input:'+r['path'],pin(r))
    src=read(O/'hc-c2-norm-state-source-contract-v137.json')
    for r in src['parts']:ck('source:'+str((r['path'],r['first_line'])),pin(r)and Path(r['path']).read_text().splitlines()[r['first_line']-1:r['last_line']]==r['source'])
    with ZipFile(src['wrapper_archive']['path'])as z:ck('wrapper_archive',pin(src['wrapper_archive'])and z.read(src['wrapper_member'])==Path(src['parts'][-1]['path']).read_bytes())
    tree=ast.parse(Path(src['parts'][-1]['path']).read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef)and x.name=='AscendRMSNorm');f=next(x for x in cls.body if isinstance(x,ast.FunctionDef)and x.name=='forward_oot')
    j=next(i for i,x in enumerate(f.body)if x.lineno==99);tail=f.body[j+1:]
    ck('direct_rstd_unused',not any(isinstance(x,ast.Name)and x.id=='residual'and isinstance(x.ctx,ast.Load)for b in tail for x in ast.walk(b))and isinstance(tail[-1],ast.Return)and isinstance(tail[-1].value,ast.Name)and tail[-1].value.id=='x')
    ck('optional_bias_branch',any(isinstance(x,ast.Call)and isinstance(x.func,ast.Attribute)and x.func.attr=='add_'for b in tail for x in ast.walk(b)))
    prof=read(O/'hc-c2-norm-wrapper-profile-v137.json');old={(r['database_index'],r['forward_API_rowid'],r['source_block_ordinal']):r for r in read(O/'hc-c2-native-consumers-v136.json')['records']};groups=defaultdict(list);blocks=Counter();seen=set();allbias=0
    for r in prof['records']:groups[r['database_index']].append(r)
    for ref in prof['databases']:
        di=ref['database_index'];ck('rawpin:'+str(di),pin(ref['raw_database']));ck('cachepin:'+str(di),pin(ref['cached_native']))
        db=sqlite3.connect(Path(ref['raw_database']['path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;ss={r['id']:r['value']for r in db.execute('SELECT *FROM STRING_IDS')}
        wanted={r['norm_API']['callchainId']for r in groups[di]};ph=','.join('?'for _ in wanted);fs=defaultdict(list)
        for r in db.execute('SELECT *FROM PYTORCH_CALLCHAINS WHERE id IN ('+ph+')',tuple(wanted)):fs[r['id']].append(dict(r,value=ss[r['stack']]))
        biasst=[sid for sid,v in ss.items()if '/vllm_ascend/ops/layernorm.py(101):'in v];bc=set()
        if biasst:
            for r in db.execute('SELECT DISTINCT id FROM PYTORCH_CALLCHAINS WHERE stack IN ('+','.join('?'for _ in biasst)+')',biasst):bc.add(r[0])
        aps=[]
        if bc:
            aps=[dict(a,name_text=ss[a['name']])for a in db.execute('SELECT rowid AS api_rowid,*FROM PYTORCH_API WHERE callchainId IN ('+','.join('?'for _ in bc)+')',tuple(bc))]
        cid=defaultdict(set)
        for x in db.execute('SELECT *FROM CONNECTION_IDS'):cid[x['id']].add(x['connectionId'])
        for r in groups[di]:
            key=di,r['forward_API_rowid'],r['source_block_ordinal'];p=old[key];a,t,d=r['norm_API'],r['norm_TASK'],r['norm_definition'];tk=di,t['task_record_rowid'];blocks[key[2]]+=1
            ck('unique_parent:'+str(key),tk not in seen and a==p['consumers']['POOL_NORM']['API']and t==p['consumers']['POOL_NORM']['TASK']and d==p['consumers']['POOL_NORM']['definition']and not r['runtime_bias_loaded_or_all_external_rstd_observers_bound']);seen.add(tk)
            aa=db.execute('SELECT *FROM PYTORCH_API WHERE rowid=?',(a['api_rowid'],)).fetchone();tt=db.execute('SELECT *FROM TASK WHERE rowid=?',(tk[1],)).fetchone()
            ck('exact_API_TASK_CID:'+str(tk),all(a[k]==aa[k]for k in aa.keys())and all(t[k]==tt[k]for k in tt.keys())and t['connectionId']in cid[a['connectionId']])
            rawframes=sorted(fs[a['callchainId']],key=lambda x:x['stackDepth']);ck('fullframes:'+str(tk),rawframes==r['full_norm_callchain']and any('/vllm_ascend/ops/layernorm.py(99): forward_oot'in x['value']for x in rawframes)and any('/compressor.py(67): pool_projected'in x['value']for x in rawframes))
            nxt=r['next_index_WK_API'];ck('index_next_parent',nxt==p['consumers']['INDEX_WK']['API'])
            hits=[x for x in aps if x['globalTid']==a['globalTid']and int(a['endNs'])<=int(x['startNs'])<=int(x['endNs'])<=int(nxt['startNs'])];allbias+=len(hits)
            ck('exact_recorded_bias_window:'+str(tk),hits==r['recorded_bias_line101_APIs_before_index'])
        db.close()
    ck('72positive_wrapper',len(prof['records'])==prof['frontiers']==prof['source_positive_no_residual_forward99']==len(seen)==72 and blocks=={2:24,8:24,14:24}and prof['recorded_bias_line101_APIs']==allbias==0 and prof['absence_of_recorded_Add_is_not_loaded_bias_false_proof'])
    demand=read(O/'hc-c2-current-row-demand-proof-v137.json');counter=Counter()
    for i,r in enumerate(demand['cases']):
        s,n,valid=r['s'],r['n'],r['valid'];ends=[x for x in range(s,s+n)if x%2]if valid else[];pool={x for end in ends for x in [end-1,end]if x>=s};even={x for x in range(max(s,s+n-32),s+n)if x%2==0}if valid else set();full=set(range(max(s,s+n-32),s+n))if valid else set();current=set(range(s,s+n))if valid else set()
        ck('demand:'+str(i),r['E']==len(ends)and r['B']==sum(end-1<s for end in ends)and r['U']==len(full)and r['U_even']==len(even)and r['current_rows']==len(current)and r['current_pool_rows']==len(pool)and r['current_pool_or_full_tail_rows']==len(pool|full)==len(current)and r['current_pool_or_even_tail_rows']==len(pool|even)==len(current)and r['missing_from_pool']==sorted(current-pool)and r['missing_from_pool_and_even_tail']==[]and r['even_state_retention_does_not_cut_current_projection_rows'])
        ck('symbolic_counts:'+str(i),len(pool)==(n-int((s+n)%2==1)if valid and n>0 else 0)and (not(current-pool)or current-pool=={s+n-1}and (s+n-1)%2==0 and s+n-1 in even))
        counter[valid]+=1
    ck('demand_domain',demand['case_count']==len(demand['cases'])==33024 and counter=={False:16512,True:16512})
    def bfloat(x):
        raw=int.from_bytes(struct.pack('>f',x),'big');high,low=divmod(raw,65536)
        if low>32768 or low==32768 and high%2:high+=1
        return struct.unpack('>f',(high*65536).to_bytes(4,'big'))[0]
    def norm(v,eps):
        den=math.sqrt(sum(x*x for x in v)/len(v)+eps);return [bfloat(x/den)for x in v]
    wit=read(O/'hc-c2-norm-round-and-zero-witnesses-v137.json');r=wit['rounding'];ck('bf16_boundary_witness',[bfloat(x)for x in r['pooled_FLOAT']]==r['pooled_BF16']and norm(r['pooled_BF16'],r['epsilon'])==r['preserve_round']and norm(r['pooled_FLOAT'],r['epsilon'])==r['skip_round']and r['skip_round']!=r['preserve_round'])
    r=wit['noncommuting'];ck('nonlinear_domain',norm(r['pooled'],0)==r['norm_after_pool']and r['norm_after_pool']!=r['pool_after_each_norm'])
    ck('zero_output_observers',wit['zeros']['negative_zero_sign']==int(math.copysign(1,0.0*-1))==-1 and wit['zeros']['loaded_beta_example']==wit['zeros']['final_noncompletion_value_with_loaded_beta']==1)
    budget=read(O/'hc-c2-state-and-HC-live-budget-v137.json');positive=negative=0
    for i,r in enumerate(budget['cases']):
        rows,U,S,z,m=[r[k]for k in ['rows','U','S','z','m']];bank=(U+z)*3*rows*1024*4;hc=10240*m*(int(r['privateY'])+int(r['privateZ']));pooled=1024*8*m*int(r['additional_pooled_snapshot']);tail=4096*32*2*int(r['additional_tail_snapshot']);D=bank+hc+pooled+tail;lo=D//540928;hi=(D+540927)//540928
        ck('budget_contents:'+str(i),r['state_bank_bytes']==bank and r['HC_optional_private_YZ_bytes']==hc and r['pooled_snapshot_bytes']==pooled and r['tail_snapshot_bytes']==tail and r['additional_live_content_bytes']==D and r['min_global_ID_loss']==lo and r['max_global_ID_loss']==hi and r['min_net_free_ID_change']==S-hi and r['max_net_free_ID_change']==S-lo and not r['HC_X_value_content_retired_by_C2_completion'])
        rems={0,1,540927,270464,D%540928,(D-1)%540928,(D+1)%540928}
        for rem in rems:
            B=540928*1000000+rem;delta=(B-D)//540928-(B//540928-S);exact=S-((D-rem+540927)//540928)
            ck('budget_rounding:'+str((i,rem)),delta==exact and S-hi<=delta<=S-lo)
            positive+=delta>0;negative+=delta<0
    ck('joint_budget_not_automatic_gain',budget['case_count']==len(budget['cases'])==1024 and positive>0 and negative>0 and 'ceil((D_total-r)/C)'in budget['formula'])
    work=read(O/'hc-c2-pool-norm-delivery-and-work-v137.json');ck('delivery_cuts',len(work['cuts'])==4 and len(work['models'])==3)
    for r in work['models']:
        m,t=r['sender_rows'],r['T'];ck('work_references',r['full_token_norm_value_IO']==2048*t and r['full_token_norm_rstd_descriptor_value_content']==4*t and r['optional_private_Y_and_normedY_content']==20480*m and r['HC_X_value_content']==40960*m and r['C2_full_current_projection_dense_ops_per_active_row']==10485760 and r['C2_projection_row_deletion_from_even_state_or_E_only_norm']==0 and r['profile_E_or_active_rows_not_bound'])
    comp=read(O/'hc-c2-norm-state-composition-v137.json');ck('two_scoped_pairs',comp['new_pairs']==[['F01','F18'],['F01','F34']]and comp['new_pair_clauses']==2 and not comp['all_runtime_gates_proved']and not comp['whole_pair_or_all_subsets_closed'])
    if args.preflight:
        rel=read(O/'candidate-relation-clauses-v136.json')
        for pair in comp['new_pairs']:ck('pending:'+str(pair),not any({r['a'],r['b']}==set(pair)for r in rel['clauses']))
        manifest=read(O/'checkpoint-manifest-v136.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v137.json');ck('progress',cp['status']=='PROGRESS'and not cp['overall_complete']and cp['whole_modules_closed']==0)
        rel=read(O/'candidate-relation-clauses-v137.json');ck('146_26',len(rel['clauses'])==rel['clause_count']==rel['selected_pair_count']==146 and len({tuple(sorted((r['a'],r['b'])))for r in rel['clauses']})==146 and len(rel['higher_order_groups'])==26)
        for c in comp['clauses']:
            x=next(r for r in rel['clauses']if r['id']==c['id']);ck('scoped:'+c['id'],{x['a'],x['b']}=={c['a'],c['b']}and not x['all_runtime_gates_proved']and x['critical_path_gain_us']is None)
        cv=read(O/'candidate-relation-coverage-v137.json');nv=read(O/'candidate-pair-navigation-v137.json');ck('146_449',cv['scoped_conditional_pairs']==146 and cv['remaining_pair_relations']==nv['remaining_central_pair_reconciliation']==449)
        for gid,member in comp['group_extensions']:
            gg=next(r for r in rel['higher_order_groups']if r['id']==gid);ck('open_group:'+member,member in gg['members']and not gg['all_subsets_analyzed']and not gg['legal_joint_fusion_proved'])
        a=read(O/'operator-analysis-coverage-ledger-v136.json');b=read(O/'operator-analysis-coverage-ledger-v137.json');ck('native_gaps_unchanged',all(a[k]==b[k]for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']))
        m=read(O/'module-phase-branch-matrix-v137.json');ck('358_26',len(m['modules'])==26 and sum(len(r['branches'])for r in m['modules'])==358)
        for r in read(O/'candidate-record-index-v137.json')['model_candidates']:ck('master:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        ck('active',read(T/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING');manifest=read(O/'checkpoint-manifest-v137.json')
    for r in manifest['artifacts']:ck('manifest:'+r['path'],pin(r))
    result=dict(status='PASS',checks=len(checks),preflight=args.preflight,native_norm_frontiers=72,current_row_demand_cases=33024,joint_budget_cases=1024,new_scoped_pairs=2,new_native_promotions=0,new_device_work=False,overall_complete=False)
    if args.preflight:write(O/'hc-c2-norm-state-preflight-v137.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
