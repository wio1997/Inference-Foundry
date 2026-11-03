"""Bind remaining large eager HC/SP calls through CPU forward scopes and exact CIDs."""
from pathlib import Path
from collections import defaultdict,Counter
import hashlib,json,sqlite3
from finalize_origin_coverage_v9 import N,O,read,write


def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def main():
    assert not(O/'REVIEW-CHECKPOINT-v134.json').exists()
    intake=read(O/'hc-sp-eager-intake-v134.json');bydb=defaultdict(list)
    for g in intake['remaining_operators']:bydb[g['database_index']].append(g)
    rawrefs={r['database_index']:r for r in read(O/'host-launch-original-API-coverage-v126.json')['databases']}
    cache_refs={r['database_index']:r for r in read(O/'hc-sp-native-frontiers-v133.json')['databases']}
    pop=read(O/'communication-native-operator-population-v66.json')['operations'];records=[];roots=[];refs=[]
    init=read(O/'embedding-initialization-native-chains-v67.json')['target_HC_initializations']
    core=[('HC',889,'aclnnHcPre'),('NORM',896,'aclnnRmsNorm'),('AG',898,'HcclAllGather'),('RS',901,'HcclReduceScatter'),('POST',902,'aclnnHcPost'),('FFN_PRE',905,'aclnnHcPre'),('FFN_POST',919,'aclnnHcPost')]
    for di,gs in sorted(bydb.items()):
        ref=rawrefs[di];assert pin(ref['path'])['sha256']==ref['sha256'];db=sqlite3.connect(Path(ref['path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row
        strings={r['id']:r['value']for r in db.execute('SELECT *FROM STRING_IDS')};chains=defaultdict(list);roles=defaultdict(set)
        lines=[889,896,898,901,902,905,919,467,475,530,531,1561,1563,1586,1592,2553]
        expected={line:('/worker/model_runner_v1.py'if line==2553 else'/attention/dsa_v41.py'if line in [467,475,530,531]else'/models/deepseek_v41/model.py')for line in lines}
        for x in db.execute('SELECT *FROM PYTORCH_CALLCHAINS'):
            value=strings[x['stack']]
            if '/vllm_ascend/'not in value:continue
            if any(f'({i}):'in value for i in lines)or'/compressor.py(71): forward'in value:
                chains[x['id']].append(dict(x,value=value))
                for line in lines:
                    if expected[line]+f'({line}):'in value:roles[line].add(x['id'])
                if '/compressor.py(71): forward'in value:roles['C1'].add(x['id'])
        cidmap=defaultdict(set)
        for x in db.execute('SELECT *FROM CONNECTION_IDS'):cidmap[x['id']].add(x['connectionId'])
        api={r['api_rowid']:dict(r,name_text=strings[r['name']])for r in db.execute('SELECT rowid AS api_rowid,*FROM PYTORCH_API')}
        cp=Path(cache_refs[di]['path']);data=read(cp);tasks=data['TASK_records'];defs={r['globalTaskId']:r for r in data['COMPUTE_definitions']};byapi=defaultdict(list)
        for t in tasks:
            for aid in t.get('native_linked_API_rowids',[]):byapi[aid].append(t)
        def source(a,line):return a['callchainId']in roles[line]
        def stage(a,op=None):
            hits=[t for t in byapi[a['api_rowid']]if t['globalTaskId']in defs and(op is None or defs[t['globalTaskId']]['opType_text']==op)]
            assert len(hits)==1,(di,a['api_rowid'],op,len(hits));t=hits[0];d=defs[t['globalTaskId']]
            return dict(API=a,TASK=t,definition={k:d[k]for k in ['compute_record_rowid','globalTaskId','opType_text','inputShapes_text','outputShapes_text','inputDataTypes_text','outputDataTypes_text','blockNum','mixBlockNum','hf32Eligible']},selected_frames=chains[a['callchainId']])
        def comm(a,role):
            hits=[g for g in pop if g['database_index']==di and g['semantic_role']==role and g['connectionId']in cidmap[a['connectionId']]]
            assert len(hits)==1,(di,a['api_rowid'],role,len(hits));return dict(API=a,COMM=hits[0],selected_frames=chains[a['callchainId']])
        forwards=[a for a in api.values()if a['name_text']=='forward'and source(a,2553)];root_groups=defaultdict(list)
        for g in gs:
            aa=[a for a in api.values()if a['connectionId']is not None and g['connectionId']in cidmap[a['connectionId']]and a['name_text']=='HcclAllGather'and source(a,898)]
            assert len(aa)==1,(di,g['COMM_OP_rowid'],len(aa));a=aa[0]
            parents=[p for p in forwards if p['globalTid']==a['globalTid']and int(p['startNs'])<=int(a['startNs'])<=int(a['endNs'])<=int(p['endNs'])]
            assert len(parents)==1,(di,g['COMM_OP_rowid'],len(parents));root_groups[parents[0]['api_rowid']].append(g)
        for rootid,rgs in sorted(root_groups.items()):
            root=api[rootid];inside=sorted([a for a in api.values()if a['globalTid']==root['globalTid']and int(root['startNs'])<=int(a['startNs'])<=int(a['endNs'])<=int(root['endNs'])],key=lambda a:(int(a['startNs']),a['api_rowid']))
            host=[a for a in api.values()if a['name_text'].startswith('wio_hostdiag::execute_model/')and a['globalTid']==root['globalTid']and int(a['startNs'])<=int(root['startNs'])<=int(root['endNs'])<=int(a['endNs'])];assert len(host)==1
            groups={name:[a for a in inside if a['name_text']==name_text and source(a,line)]for name,line,name_text in core}
            assert len(rgs)==40 and all(len(x)==40 for x in groups.values()),(di,rootid,{k:len(v)for k,v in groups.items()})
            root_taskids={t['task_record_rowid']for a in inside for t in byapi[a['api_rowid']]}
            ini=[x for x in init if x['database_index']==di and all(s['TASK_rowid']in root_taskids for s in x['stages'])];assert len(ini)==1,(di,rootid,len(ini))
            rootinfo=dict(database_index=di,forward_API=root,host_execute_API=host[0],source_frames=chains[root['callchainId']],target_HC_initialization=ini[0],source_order_layers=40,
                actual_client_step_crossrank_episode_or_parameter_identity_bound=False)
            roots.append(rootinfo);previous=None
            for j in range(40):
                aa={name:groups[name][j]for name,_,_ in core};order=[aa[name]for name,_,_ in core]
                assert all(int(x['endNs'])<=int(y['startNs'])for x,y in zip(order,order[1:])),(di,rootid,j)
                stages={name:stage(aa[name],op)for name,op in [('HC','HcPre'),('NORM','RmsNorm'),('POST','HcPost'),('FFN_PRE','HcPre'),('FFN_POST','HcPost')]}
                ag=comm(aa['AG'],'ATTENTION_INPUT_SP_ALLGATHER');rs=comm(aa['RS'],'ATTENTION_OUTPUT_SP_REDUCESCATTER');assert ag['COMM']in rgs and ag['COMM']['TASK_rowid']is None
                attention=[a for a in inside if int(aa['AG']['endNs'])<=int(a['startNs'])<=int(a['endNs'])<=int(aa['RS']['startNs'])]
                def projection(line,dtype,weight):
                    found=[]
                    for a in attention:
                        if not source(a,line):continue
                        for t in byapi[a['api_rowid']]:
                            d=defs.get(t['globalTaskId'])
                            if d and d['opType_text']=='MatMulV2'and d['inputDataTypes_text']==dtype and d['inputShapes_text'].endswith(';'+weight+'"'):found.append(stage(a,'MatMulV2'))
                    assert len(found)<=1,(di,rootid,j,line,len(found));return found[0]if found else None
                qa=projection(467,'DT_BF16;DT_BF16','1280,5120');kv=projection(475,'DT_BF16;DT_BF16','512,5120');c1=projection('C1','DT_BF16;DT_BF16','512,5120');c2k=projection(530,'FLOAT;FLOAT','512,5120');c2s=projection(531,'FLOAT;FLOAT','512,5120');assert qa and kv
                assert bool(c1)==(j==20)and bool(c2k)==bool(c2s)==(j in [2,8,14])
                lower=int(previous['stages']['FFN_POST']['API']['endNs'])if previous else int(root['startNs']);upper=int(aa['HC']['startNs'])
                gap=[a for a in inside if lower<=int(a['startNs'])<=int(a['endNs'])<=upper]
                eg=[a for a in gap if source(a,1586)and a['connectionId']is not None and byapi[a['api_rowid']]]
                mean=[a for a in gap if source(a,1561)and a['name_text']=='aclnnMean'];aux=[a for a in gap if source(a,1563)and a['name_text']=='HcclAllGather']
                assert bool(eg)==(j in [1,14])and len(mean)==len(aux)==int(j in [37,38,39]),(di,rootid,j,len(eg),len(mean),len(aux))
                r=dict(database_index=di,forward_API_rowid=rootid,source_block_ordinal=j,stages=stages,SP_allGather=ag,output_reduceScatter=rs,
                    QA=qa,SWA_KV=kv,C1=c1,C2_KV=c2k,C2_SCORE=c2s,previous_FFN_Pre=previous['stages']['FFN_PRE']if previous else None,previous_FFN_Post=previous['stages']['FFN_POST']if previous else None,
                    Engram_positive_APIs=[dict(API=a,selected_frames=chains[a['callchainId']],native_TASK_rowids=[t['task_record_rowid']for t in byapi[a['api_rowid']]])for a in eg],
                    aux_mean=stage(mean[0],'ReduceMean')if mean else None,aux_allGather=comm(aux[0],'DSPARK_AUX_RESIDUAL_ALLGATHER')if aux else None,
                    exact_tensor_parameter_premix_values_or_crossrank_episode_bound=False,
                    grade='EXACT_CID_NATIVE_STAGES_AND_SAME_TID_FORWARD_CONTAINMENT_SOURCE_ORDER;NO_COMM_TASK_OR_TENSOR_ALIAS_INVENTED')
                records.append(r);previous=r
        refs.append(dict(database_index=di,raw_database=pin(ref['path']),cached_native=pin(cp)));db.close()
    assert len(records)==960 and len(roots)==24
    counts=dict(QA=sum(bool(r['QA'])for r in records),SWA_KV=sum(bool(r['SWA_KV'])for r in records),C1=sum(bool(r['C1'])for r in records),C2_KV=sum(bool(r['C2_KV'])for r in records),C2_SCORE=sum(bool(r['C2_SCORE'])for r in records),Engram=sum(bool(r['Engram_positive_APIs'])for r in records),aux=sum(bool(r['aux_mean'])for r in records))
    assert counts==dict(QA=960,SWA_KV=960,C1=24,C2_KV=72,C2_SCORE=72,Engram=48,aux=72)
    write(O/'hc-sp-eager-native-frontiers-v134.json',dict(records=records,forward_roots=roots,databases=refs,frontier_count=960,forward_root_count=24,source_role_counts=counts,
        all_960_COMM_without_COMPUTE_TASK_preserved=True,prior_graph80_SWA_C1_ambiguities_not_resolved_by_new_eager_records=True,new_native_operand_or_global_phase_promotion=False,new_device_work=False,overall_complete=False))
    print(json.dumps(dict(status='EAGER_HC_SP_SOURCE_FRONTIERS_RECONCILED',frontiers=960,forward_roots=24,source_role_counts=counts,new_device_work=False)))


if __name__=='__main__':main()
