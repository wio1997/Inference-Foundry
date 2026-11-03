"""Read only archived tensor files; exact recorded-row destination fanout, no inference."""
from pathlib import Path
from collections import Counter
import gzip,hashlib,json,os,socket,sys,time
from saved_tensor_archive_v122 import load_saved_tensor_list

ROOT=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030')
RUN=ROOT/'observer-20261030T160315Z-76b79f0f'
INDEX_SHA='REPLACE_INDEX_SHA'
DIST_SHA='REPLACE_DIST_SHA'

def main():
    assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'and os.geteuid()==0
    assert 'torch'not in sys.modules and 'torch_npu'not in sys.modules
    start=time.monotonic();deadline=start+40
    raw=(RUN/'value-review/decoded-records.jsonl.gz').read_bytes();assert hashlib.sha256(raw).hexdigest()=='d6e787e5bd8498c8e68ccd3f2d1874a8dfd20260071fae7f16ad6876ec3c3029'
    records=[json.loads(l)for l in gzip.decompress(raw).splitlines()]
    ib=(RUN/'tensor-index.json').read_bytes();assert hashlib.sha256(ib).hexdigest()==INDEX_SHA;index=json.loads(ib)
    db=(ROOT/'profile-readonly-20261031/EP-row-distribution-and-placement-counterfactual-v17.json').read_bytes();assert hashlib.sha256(db).hexdigest()==DIST_SHA
    dist=json.loads(db);formal={(c['role'],c['layer'],c['ordinal'])for c in dist['cohorts']if c['phase']=='formal_C8_measure'}
    selected=[r for r in records if(r['role'],r['layer'],r['ordinal'])in formal];assert len(formal)==87 and len(selected)==1392
    results=[];total_bytes=0;files=[];proof_digest=hashlib.sha256()
    for r in selected:
        assert time.monotonic()<deadline,'CPU read-only deadline exceeded'
        item=index[r['index']];assert item['tensor']==r['tensor']and item['metadata']==r['metadata']
        blobs={}
        for field in ['tensor','metadata']:
            p=Path(item[field]);assert str(p).startswith(str(RUN/'runtime/observer')+'/')and p.stat().st_size<=2*1024*1024
            b=p.read_bytes();total_bytes+=len(b);assert total_bytes<=128*1024*1024
            assert hashlib.sha256(b).hexdigest()==item[field+'_sha256'];blobs[field]=b
        meta=json.loads(blobs['metadata']);ts=load_saved_tensor_list(blobs['tensor']);assert len(ts)==len(meta['payload_names'])
        values=dict(zip(meta['payload_names'],ts));M,K,E=r['M'],r['K'],r['E_local'];ids=values['topk_ids']
        assert ids.size==(M,K)and ids.storage.kind in[('i',4),('q',8)]
        assert values['observer_ordinal'].values()==[r['ordinal']]and values['expert_counts'].values()==r['receiver_expert_counts']
        masks=[True]*M if meta['mask_is_none']else values['x_active_mask'].values();assert len(masks)==M and sum(masks)==r['valid_rows']
        flat=ids.values();hist=Counter();A=[0]*16;D=[0]*16;fh=Counter();examples=[];dups=0
        for row,active in enumerate(masks):
            if not active:continue
            picked=flat[row*K:(row+1)*K];assert len(picked)==K and all(0<=x<r['E_global']for x in picked)
            hist.update(picked);dups+=len(set(picked))!=K;bydest={}
            for k,expert in enumerate(picked):bydest.setdefault(expert//E,[]).append((k,expert%E))
            for dest,fanout in bydest.items():A[dest]+=len(fanout);D[dest]+=1;fh[len(fanout)]+=1
            if len(examples)<2 and any(len(v)>1 for v in bydest.values()):examples.append(dict(recorded_row=row,expert_ids=picked,destinations={str(k):v for k,v in bydest.items()}))
            proof_digest.update(json.dumps([r['index'],row,picked,sorted(bydest)],separators=(',',':')).encode())
        assert dict(hist)=={int(k):v for k,v in r['sender_expert_histogram'].items()}and sum(A)==r['sender_assignment_count']==K*r['valid_rows']
        assert sum(D)==sum(fh.values())and sum(f*n for f,n in fh.items())==sum(A)
        sender=r['ep_rank'];Ar=sum(n for d,n in enumerate(A)if d!=sender);Dr=sum(n for d,n in enumerate(D)if d!=sender)
        Ac=sum(n for d,n in enumerate(A)if d//8!=sender//8);Dc=sum(n for d,n in enumerate(D)if d//8!=sender//8)
        results.append(dict(source_index=r['index'],role=r['role'],layer=r['layer'],observer_ordinal=r['ordinal'],sender_EP_slot=sender,
            M=M,K=K,E_local=E,valid_recorded_rows=r['valid_rows'],assignments_by_destination=A,token_destinations_by_destination=D,
            per_packet_fanout_histogram=dict(fh),duplicate_expert_active_rows=dups,examples=examples,
            remote_assignments=Ar,remote_token_destination_pairs=Dr,remote_repeated_references=Ar-Dr,
            logical_crossTP8_assignments=Ac,logical_crossTP8_token_destination_pairs=Dc,
            current_remote_Hplus32_record_bytes=Ar*5152,selected_remote_Hplus64_fanout_record_bytes=Dr*5184,
            unchanged_remote_BF16_return_bytes=Ar*10240,
            tensor_sha256=item['tensor_sha256'],metadata_sha256=item['metadata_sha256']))
        files.append(dict(source_index=r['index'],tensor=item['tensor'],metadata=item['metadata'],tensor_sha256=item['tensor_sha256'],metadata_sha256=item['metadata_sha256'],tensor_bytes=len(blobs['tensor']),metadata_bytes=len(blobs['metadata'])))
    groups={}
    for role in ['target','draft']:
        rows=[r for r in results if r['role']==role]
        fields=['valid_recorded_rows','remote_assignments','remote_token_destination_pairs','remote_repeated_references',
            'logical_crossTP8_assignments','logical_crossTP8_token_destination_pairs','current_remote_Hplus32_record_bytes','selected_remote_Hplus64_fanout_record_bytes','unchanged_remote_BF16_return_bytes','duplicate_expert_active_rows']
        sums={f:sum(r[f]for r in rows)for f in fields};groups[role]=dict(records=len(rows),**sums)
    assert 'torch'not in sys.modules and 'torch_npu'not in sys.modules
    print(json.dumps(dict(hostname=socket.gethostname(),status='READ_ONLY_ARCHIVE_FANOUT_COMPLETE',run_id=RUN.name,
        selected_formal_cohorts=len(formal),snapshot_records=len(results),read_bytes=total_bytes,elapsed_CPU_process_seconds=time.monotonic()-start,
        INDEX_SHA=INDEX_SHA,DIST_SHA=DIST_SHA,groups=groups,records=results,files=files,recorded_active_rows_digest=proof_digest.hexdigest(),
        scope='Same archivedrecordpostmaskM×Krows,notrequestIDs/newdeviceevents/laterprofiletiming. Packetcomparisonisaselectedlayoutreferenceconditionalonquantizedbyteequality,metadataandnormalnativeinput-rowassociation.',
        new_device_work=False,Torch_imported=False,new_server_files_written=False,no_model_or_replay=True)))

if __name__=='__main__':main()
