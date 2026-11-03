"""Bind draft HC epochs to exact saved repeat/seed/SP anchors; no target execution."""
from pathlib import Path
from collections import defaultdict,Counter
import hashlib,json,sqlite3
from finalize_origin_coverage_v9 import N,O,read,write


def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def main():
    assert not(O/'REVIEW-CHECKPOINT-v142.json').exists()
    context=read(O/'context-sp-profile-frontiers-v130.json');rawrefs={x['database_index']:x for x in read(O/'host-launch-original-API-coverage-v126.json')['databases']};cacherefs={x['database_index']:x for x in context['databases']};groups=defaultdict(list);records=[];refs=[]
    for r in context['records']:groups[r['database_index']].append(r)
    for di,rr in sorted(groups.items()):
        raw=Path(rawrefs[di]['path']);db=sqlite3.connect(raw.as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;ss={x['id']:x['value']for x in db.execute('SELECT *FROM STRING_IDS')};fs=defaultdict(list)
        for x in db.execute('SELECT *FROM PYTORCH_CALLCHAINS'):fs[x['id']].append(dict(x,value=ss.get(x['stack'])))
        apis={x['api_rowid']:dict(x,name_text=ss[x['name']])for x in db.execute('SELECT rowid AS api_rowid,*FROM PYTORCH_API')};cids=defaultdict(set)
        for x in db.execute('SELECT *FROM CONNECTION_IDS'):cids[x['id']].add(x['connectionId'])
        cp=Path(cacherefs[di]['path']);data=read(cp);ts={x['task_record_rowid']:x for x in data['TASK_records']};defs={x['globalTaskId']:x for x in data['COMPUTE_definitions']};byapi=defaultdict(list)
        for t in ts.values():
            for aid in t.get('native_linked_API_rowids',[]):byapi[aid].append(t)
        def source(a,path,line):return any(path+f'({line}):'in (f['value']or'')for f in fs[a['callchainId']])
        def stages(aa,op):
            hits=[(a,t,defs[t['globalTaskId']])for a in aa for t in byapi[a['api_rowid']]if t['globalTaskId']in defs and defs[t['globalTaskId']]['opType_text']==op]
            return [dict(API=a,TASK=t,definition={k:d[k]for k in ['compute_record_rowid','globalTaskId','opType_text','inputShapes_text','outputShapes_text','inputDataTypes_text','outputDataTypes_text','blockNum','mixBlockNum','hf32Eligible']},selected_frames=fs[a['callchainId']])for a,t,d in hits]
        for r in rr:
            rep=ts[r['draft_repeat']['TASK_rowid']];seed=ts[r['seed_TASK']['task_record_rowid']]
            ra=[apis[x]for x in rep['native_linked_API_rowids']if source(apis[x],'/models/deepseek_v41/dspark.py',249)];sa=[apis[x]for x in seed['native_linked_API_rowids']if source(apis[x],'/spec_decode/llm_base_proposer.py',1584)]
            assert len(ra)==len(sa)==1,(di,r['source_order_ordinal'],'repeat/seed',len(ra),len(sa));ra,sa=ra[0],sa[0]
            assert ra['globalTid']==sa['globalTid']and int(ra['endNs'])<int(sa['startNs'])
            inside=sorted([a for a in apis.values()if a['globalTid']==ra['globalTid']and int(ra['endNs'])<=int(a['startNs'])<=int(a['endNs'])<=int(sa['startNs'])],key=lambda a:(int(a['startNs']),a['api_rowid']))
            aa=[a for a in inside if source(a,'/models/deepseek_v41/dspark.py',265)];layers=[]
            for j,parent in enumerate(r['layers']):
                gid=parent['draft_SP_allGather']['connectionId'];ag=[a for a in aa if a['name_text']=='HcclAllGather'and gid in cids[a['connectionId']]];assert len(ag)==1
                aag=ag[0];aa0=[a for a in aa if a['name_text']=='aclnnHcPre'and source(a,'/models/deepseek_v41/model.py',889)];assert len(aa0)==3,(di,r['source_order_ordinal'],'HCs',len(aa0));lo=int(aa0[j]['startNs']);hi=int(aa0[j+1]['startNs'])if j<2 else int(sa['startNs']);scope=[a for a in aa if lo<=int(a['startNs'])<=int(a['endNs'])<=hi];st={}
                for role,line,op in [('HC',889,'HcPre'),('NORM',896,'RmsNorm'),('POST',902,'HcPost'),('FFN_PRE',905,'HcPre'),('FFN_NORM',846,'RmsNormCast'),('FFN_POST',919,'HcPost')]:
                    h=stages([a for a in scope if source(a,'/models/deepseek_v41/model.py',line)],op);assert len(h)==1,(di,r['source_order_ordinal'],j,role,len(h));st[role]=h[0]
                rs=[a for a in scope if a['name_text']=='HcclReduceScatter'and source(a,'/models/deepseek_v41/model.py',901)];assert len(rs)==1
                layers.append(dict(source_layer_ordinal=j,context_SP_parent=parent,stages=st,SP_API=aag,RS_API=rs[0],same_CPU_source_call_window_only=True,actual_request_weight_cache_or_native_new_publication_identity=False))
            tail=stages([a for a in inside if source(a,'/models/deepseek_v41/dspark.py',273)and source(a,'/models/deepseek_v41/model.py',856)],'Cast')
            casts=tail;castin=[s for s in casts if s['definition']['inputDataTypes_text']=='DT_BF16'and s['definition']['outputDataTypes_text']=='FLOAT'];castout=[s for s in casts if s['definition']['inputDataTypes_text']=='FLOAT'and s['definition']['outputDataTypes_text']=='DT_BF16']
            def one(op):
                hh=stages([a for a in inside if source(a,'/models/deepseek_v41/dspark.py',273)and source(a,'/models/deepseek_v41/model.py',856)],op);assert len(hh)==1,(di,r['source_order_ordinal'],'collapse',op,len(hh));return hh[0]
            assert len(castin)==len(castout)==1
            records.append(dict(database_index=di,source_order_ordinal=r['source_order_ordinal'],context_parent=r,repeat=stages([ra],'Tile')[0],seed=stages([sa],'ViewCopy')[0],CPU_window=dict(globalTid=ra['globalTid'],startNs=ra['endNs'],endNs=sa['startNs']),layers=layers,collapse=dict(CAST_IN=castin[0],MUL=one('Mul'),SUM=one('ReduceSum'),CAST_OUT=castout[0]),source_ancestry='DUMMY'if any('/dspark_proposer.py(421): dummy_run'in (f['value']or'')for f in fs[ra['callchainId']])else'PROPOSE'if any('/llm_base_proposer.py(1394): _propose'in (f['value']or'')for f in fs[ra['callchainId']])else'UNKNOWN',actual_next_token_seed_or_current_proposal_metadata_values_bound=False))
        refs.append(dict(database_index=di,raw_database=pin(raw),cached_native=pin(cp),unresolved_raw_stack_ids=sorted({f['stack']for ff in fs.values()for f in ff if f['value']is None})));db.close()
    assert len(records)==104
    write(O/'hc-draft-native-frontiers-v142.json',dict(records=records,databases=refs,frontiers=104,layers=312,HC_stage_TASK_bindings=1872,collapse_TASK_bindings=416,repeat_seed_TASK_bindings=208,source_ancestry=dict(Counter(r['source_ancestry']for r in records)),new_native_promotions=0,new_device_work=False,overall_complete=False))
    print(json.dumps(dict(status='BOUND',frontiers=104,layers=312,TASK_bindings=2496,source_ancestry=Counter(r['source_ancestry']for r in records),new_device_work=False,overall_complete=False)))


if __name__=='__main__':main()
