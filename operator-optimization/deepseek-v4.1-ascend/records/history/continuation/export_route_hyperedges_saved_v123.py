"""Export bounded saved routing IDs/masks only; CPU-only, no server data writes."""
from pathlib import Path
from collections import Counter
import base64,gzip,hashlib,json,os,socket,struct,sys,time
from saved_tensor_archive_v122 import load_saved_tensor_list

ROOT=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030')
RUN=ROOT/'observer-20261030T160315Z-76b79f0f'
INDEX_SHA='REPLACE_INDEX_SHA'
DIST_SHA='REPLACE_DIST_SHA'


def main():
    assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105' and os.geteuid()==0
    assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
    start=time.monotonic()
    raw=(RUN/'value-review/decoded-records.jsonl.gz').read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='d6e787e5bd8498c8e68ccd3f2d1874a8dfd20260071fae7f16ad6876ec3c3029'
    records=[json.loads(line)for line in gzip.decompress(raw).splitlines()]
    ib=(RUN/'tensor-index.json').read_bytes()
    assert hashlib.sha256(ib).hexdigest()==INDEX_SHA
    index=json.loads(ib)
    db=(ROOT/'profile-readonly-20261031/EP-row-distribution-and-placement-counterfactual-v17.json').read_bytes()
    assert hashlib.sha256(db).hexdigest()==DIST_SHA
    dist=json.loads(db)
    formal={(c['role'],c['layer'],c['ordinal'])for c in dist['cohorts']if c['phase']=='formal_C8_measure'}
    selected=[r for r in records if (r['role'],r['layer'],r['ordinal'])in formal]
    assert len(formal)==87 and len(selected)==1392
    packed=bytearray(b'WIOEP123');metadata=[];read_bytes=0;row_counts=Counter()
    for r in selected:
        assert time.monotonic()-start<45,'saved archive CPU-only reader deadline'
        ent=index[r['index']]
        assert ent['tensor']==r['tensor'] and ent['metadata']==r['metadata']
        blobs={}
        for field in ['tensor','metadata']:
            p=Path(ent[field])
            assert str(p).startswith(str(RUN/'runtime/observer')+'/') and p.stat().st_size<=2*1024*1024
            b=p.read_bytes();read_bytes+=len(b)
            assert read_bytes<=128*1024*1024 and hashlib.sha256(b).hexdigest()==ent[field+'_sha256']
            blobs[field]=b
        meta=json.loads(blobs['metadata']);ts=load_saved_tensor_list(blobs['tensor'])
        assert len(ts)==len(meta['payload_names'])
        values=dict(zip(meta['payload_names'],ts));M,K,E=r['M'],r['K'],r['E_local']
        ids=values['topk_ids'];assert ids.size==(M,K)
        assert values['observer_ordinal'].values()==[r['ordinal']] and values['expert_counts'].values()==r['receiver_expert_counts']
        masks=[True]*M if meta['mask_is_none']else values['x_active_mask'].values()
        assert len(masks)==M and sum(masks)==r['valid_rows']
        # Validate both saved owner maps for this recorded linear placement domain.
        expected=[e%E if e//E==r['ep_rank']else -1 for e in range(r['E_global'])]
        assert values['expert_map'].values()==values['ascend_expert_map'].values()==expected
        flat=ids.values();hist=Counter();offset=len(packed)
        packed.extend(struct.pack('<IHHII',r['index'],K,E,M,r['valid_rows']))
        row_struct=struct.Struct('<I'+'H'*K)
        for row,active in enumerate(masks):
            if not active:
                continue
            picked=flat[row*K:(row+1)*K]
            assert len(set(picked))==K and all(0<=e<r['E_global']<65536 for e in picked)
            packed.extend(row_struct.pack(row,*picked));hist.update(picked);row_counts[r['role']]+=1
        assert dict(hist)=={int(k):v for k,v in r['sender_expert_histogram'].items()}
        metadata.append(dict(source_index=r['index'],offset=offset,bytes=len(packed)-offset,
            role=r['role'],layer=r['layer'],observer_ordinal=r['ordinal'],sender_EP_slot=r['ep_rank'],
            tensor_sha256=ent['tensor_sha256'],metadata_sha256=ent['metadata_sha256'],owner_maps_match_contiguous=True))
    assert dict(row_counts)=={'target':456400,'draft':260} and read_bytes==35632142
    assert len(packed)<=10*1024*1024
    compressed=gzip.compress(bytes(packed),compresslevel=1,mtime=0)
    assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
    print(json.dumps(dict(hostname=socket.gethostname(),status='SAVED_ROUTE_HYPEREDGES_EXPORTED',
        run_id=RUN.name,records=len(metadata),formal_cohorts=len(formal),active_rows=dict(row_counts),
        read_bytes=read_bytes,archive_reader_elapsed_wall_seconds=time.monotonic()-start,
        packed_bytes=len(packed),packed_sha256=hashlib.sha256(packed).hexdigest(),
        compressed_bytes=len(compressed),compressed_sha256=hashlib.sha256(compressed).hexdigest(),
        compressed_base64=base64.b64encode(compressed).decode(),record_index=metadata,
        index_sha256=INDEX_SHA,distribution_sha256=DIST_SHA,
        exported_data='Active original source-row indices and original TopK expert IDs only;no activation/scale/weights/model data.',
        no_model_or_replay=True,Torch_imported=False,new_device_work=False,new_server_data_or_source_files_written=False)))


if __name__=='__main__':
    main()
