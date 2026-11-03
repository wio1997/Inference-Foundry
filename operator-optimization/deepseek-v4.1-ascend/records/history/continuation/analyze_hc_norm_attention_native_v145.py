"""Bind eager output projection/copy and recorded collective-label cohorts."""
from pathlib import Path
from collections import defaultdict,Counter
import hashlib,json,sqlite3
from finalize_origin_coverage_v9 import N,O,read,write


def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def main():
    assert not(O/'REVIEW-CHECKPOINT-v145.json').exists()
    core=read(O/'hc-sp-eager-native-frontiers-v134.json');sp={ (r['database_index'],r['forward_API_rowid'],r['source_block_ordinal']):r for r in read(O/'hc-sparse-eager-native-frontiers-v138.json')['records']};qr={ (r['database_index'],r['forward_API_rowid'],r['source_block_ordinal']):r for r in read(O/'hc-rope-latent-native-frontiers-v140.json')['query_records']};pop=read(O/'communication-native-operator-population-v66.json')['operations'];queue=read(O/'communication-role-contract-v16.json')['queue_source'];groups=defaultdict(list);refs={r['database_index']:r for r in core['databases']};records=[];pins=[];coh=defaultdict(list);roles=Counter();holes=[]
    for r in core['records']:groups[r['database_index']].append(r)
    spec=[('WO_A','/attention/context_parallel/dsa_cp.py',1784,['TransposeBatchMatMul']),('WO_B','/attention/context_parallel/dsa_cp.py',1795,['MatMulV2','MatMulV3']),('OUTPUT_COPY','/attention/context_parallel/dsa_cp.py',1798,['TensorMove']),('HEAD_ZERO','/attention/context_parallel/dsa_v41_cp.py',213,['ZerosLike']),('HEAD_COPY','/attention/context_parallel/dsa_v41_cp.py',214,['TensorMove','ViewCopy']),('RS_ZERO','/models/common/ops/sequence_parallel.py',74,['ZerosLike']),('RS_CAT','/models/common/ops/sequence_parallel.py',74,['ConcatD'])]
    for di,rr in sorted(groups.items()):
        ref=refs[di];db=sqlite3.connect(Path(ref['raw_database']['path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;ss={x['id']:x['value']for x in db.execute('SELECT *FROM STRING_IDS')};f=defaultdict(list);roleids=defaultdict(set)
        for x in db.execute('SELECT *FROM PYTORCH_CALLCHAINS'):
            v=ss.get(x['stack'],'');names=[role for role,path,line,ops in spec if path+f'({line}):'in v]
            if '/attention/context_parallel/dsa_cp.py(90):'in v:names+=['A2A']
            if names:
                f[x['id']].append(dict(x,value=v))
                for role in names:roleids[role].add(x['id'])
        apis=[dict(x,name_text=ss[x['name']])for x in db.execute('SELECT rowid AS api_rowid,*FROM PYTORCH_API')if x['callchainId']in f];cids=defaultdict(set)
        for x in db.execute('SELECT *FROM CONNECTION_IDS'):cids[x['id']].add(x['connectionId'])
        data=read(Path(ref['cached_native']['path']));defs={x['globalTaskId']:x for x in data['COMPUTE_definitions']};byapi=defaultdict(list)
        for t in data['TASK_records']:
            for aid in t.get('native_linked_API_rowids',[]):byapi[aid].append(t)
        rankmap=[dict(x)for x in db.execute('SELECT *FROM RANK_DEVICE_MAP')];assert len(rankmap)==1
        for r in rr:
            key=di,r['forward_API_rowid'],r['source_block_ordinal'];a0,a1=r['SP_allGather']['API'],r['output_reduceScatter']['API'];inside=[a for a in apis if a['globalTid']==a0['globalTid']and int(a0['endNs'])<=int(a['startNs'])<=int(a['endNs'])<=int(a1['startNs'])];ssparse=sp[key];st=dict(HC=r['stages']['HC'],INPUT_NORM=r['stages']['NORM'],QUERY_ROPE=qr[key]['stages']['QUERY_ROPE'],POST=r['stages']['POST'])
            st.update({role:z for role,z in ssparse['stages'].items()if role in ['SMLA','INVERSE_ROPE','SEND_LAYOUT','QLI','PREPARE','QUERY_QUANT','CANDIDATE_COPY']and z is not None})
            for role,path,line,ops in spec:
                hits=[]
                for a in inside:
                    if a['callchainId']not in roleids[role]:continue
                    for t in byapi[a['api_rowid']]:
                        d=defs.get(t['globalTaskId'])
                        if d and d['opType_text']in ops:hits.append((a,t,d))
                expect=1 if role in ['WO_A','WO_B','OUTPUT_COPY']else None
                assert len(hits)==expect if expect is not None else len(hits)<=1,(key,role,len(hits))
                if hits:
                    a,t,d=hits[0];st[role]=dict(API=a,TASK=t,definition={k:d[k]for k in ['compute_record_rowid','globalTaskId','opType_text','inputShapes_text','outputShapes_text','inputDataTypes_text','outputDataTypes_text','blockNum','mixBlockNum','hf32Eligible']},source_frames=f[a['callchainId']])
            a2a=[]
            for a in inside:
                if a['callchainId']in roleids['A2A']and a['name_text']=='HcclAlltoAll':
                    gg=[g for g in pop if g['database_index']==di and g['semantic_role']=='ATTENTION_RESTORE_TP_HEADS_ALLTOALL'and g['connectionId']in cids[a['connectionId']]]
                    if gg:assert len(gg)==1;a2a.append(dict(API=a,COMM=gg[0],source_frames=f[a['callchainId']]))
            if not a2a and di==queue['database_index']:
                qa=[a for a in inside if a['api_rowid']==queue['source']['api_rowid']]
                if qa:
                    assert len(qa)==1 and qa[0]['connectionId']is None and qa[0]['callchainId']==queue['source']['callchainId']
                    gg=[g for g in pop if g['database_index']==di and g['COMM_OP_rowid']==queue['COMM_OP_rowid']]
                    assert len(gg)==1 and gg[0]['connectionId']==queue['native_connection']and gg[0]['source_grade']=='REUSED_EXPLICIT_QUEUE_FLOW_PROOF'
                    a2a.append(dict(API=qa[0],COMM=gg[0],source_frames=f[qa[0]['callchainId']],queue_flow_proof=queue,origin_evidence_grade='REUSED_EXPLICIT_QUEUE_FLOW_PROOF_NOT_DIRECT_CID'))
            assert len(a2a)==1,(key,'A2A',len(a2a));comms=dict(SP_INPUT=r['SP_allGather'],HEAD_A2A=a2a[0],OUTPUT_RS=r['output_reduceScatter']);labels={}
            for role,g in comms.items():
                raw=dict(db.execute('SELECT *FROM COMMUNICATION_OP WHERE opId=?',(g['COMM']['opId'],)).fetchone());lab=dict(raw=raw,opName_text=ss[raw['opName']],groupName_text=ss[raw['groupName']],algorithm_text=ss[raw['algType']],rank_device_map=rankmap);labels[role]=lab;coh[role,g['COMM']['group_name'],lab['opName_text']].append(dict(database_index=di,forward_API_rowid=key[1],source_block_ordinal=key[2],rank=rankmap[0]['rankId'],COMM_OP_rowid=g['COMM']['COMM_OP_rowid'],count=g['COMM']['count']))
            roles.update(st.keys());records.append(dict(database_index=di,forward_API_rowid=key[1],source_block_ordinal=key[2],stages=st,collectives=comms,recorded_labels=labels,cache_rows=int(r['SWA_KV']['definition']['inputShapes_text'].strip('"').split(',')[0]),query_rows=int(r['QA']['definition']['inputShapes_text'].strip('"').split(',')[0]),carrier_sender_rows=int(r['stages']['NORM']['definition']['inputShapes_text'].strip('"').split(',')[0]),actual_crossrank_client_value_or_tag_global_semantics_attested=False))
        pins.append(dict(database_index=di,raw_database=pin(ref['raw_database']['path']),cached_native=pin(ref['cached_native']['path'])));db.close()
    labelgroups=[]
    for (role,gn,on),members in sorted(coh.items()):
        assert len(members)==len({m['rank']for m in members})==8 and len({m['source_block_ordinal']for m in members})==1
        labelgroups.append(dict(role=role,group_name=gn,opName=on,members=members,recorded_eight_rank_label_consistency=True,request_tensor_identity_or_exporter_tag_global_semantics_proved=False,crossrank_timestamp_join_used=False))
    assert len(records)==960 and len(labelgroups)==360
    write(O/'hc-norm-attention-native-frontiers-v145.json',dict(records=records,databases=pins,frontiers=960,source_CID_TASK_bindings=sum(roles.values()),native_role_counts=dict(roles),new_output_roles=sum(roles[k]for k in ['WO_A','WO_B','OUTPUT_COPY','HEAD_ZERO','HEAD_COPY']),collective_records=2880,recorded_label_cohorts=labelgroups,label_cohort_count=360,clock_or_formal_crossrank_phase_promotions=0,new_native_promotions=0,new_device_work=False,overall_complete=False))
    print(json.dumps(dict(status='BOUND',frontiers=960,TASKs=sum(roles.values()),roles=roles,collective_records=2880,label_cohorts=360,new_device_work=False,overall_complete=False)))


if __name__=='__main__':main()
