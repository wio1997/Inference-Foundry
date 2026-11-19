"""Opt-in bounded observation of real eager attention operands and outputs.
No tensor fed to the model is modified. Unarmed calls only run the original.
Snapshots are diagnostic synchronization, never performance evidence.
"""
from pathlib import Path
import functools,hashlib,json,os,time
import torch

_EXPECTED='42fc8b4d29efd8cd204a689bbcc9554887eb7ae385f3eeab8e06647756a2bad3'

def install(root):
    import npu_backend
    root=Path(root)
    source=Path(npu_backend.__file__)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==_EXPECTED
    cls=npu_backend.DeepseekV41NPUBackend
    assert not getattr(cls,'_forward_probe_g1096',False)
    prior=cls._forward_attention
    chosen={};counts={};arm_file=root/'forward_probe_ARM.json'
    @functools.wraps(prior)
    def observed(self,q,k,v,layer,forward_batch,compress_ratio=0,save_kv_cache=True,attn_sink=None,**kwargs):
        result=prior(self,q,k,v,layer,forward_batch,compress_ratio=compress_ratio,
                     save_kv_cache=save_kv_cache,attn_sink=attn_sink,**kwargs)
        if not arm_file.exists() or forward_batch.forward_mode.is_idle() or not q.shape[0]:return result
        assert q.device.type=='npu'
        from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode
        assert not get_is_capture_mode() and not getattr(torch.npu,'is_current_stream_capturing',lambda:False)(),'Observation requires eager, not graph capture'
        arm=json.loads(arm_file.read_text());assert arm['probe']=='G1096_REAL_ATTENTION' and 1<=arm['limit_per_ratio']<=12
        ratio=int(compress_ratio);lid=int(layer.layer_id)
        chosen.setdefault(ratio,lid)
        if chosen[ratio]!=lid:return result
        key=(arm['nonce'],ratio);count=counts.get(key,0)
        if count>=arm['limit_per_ratio']:return result
        counts[key]=count+1
        rows=sorted(set([0,int(q.shape[0])-1]));ri=torch.tensor(rows,dtype=torch.long,device=q.device)
        core=self.forward_metadata.core_metadata;pool=self.token_to_kv_pool
        original=self._view(pool.get_swa_raw_buffer(lid))
        oi=core.swa_page_indices.reshape(q.shape[0],-1).index_select(0,ri).to('cpu',torch.int64)
        if ratio:
            item=pool.layer_mapping[lid];compressed=self._view(item.compress_kv_pool.kv_buffer[item.compress_layer_id])
            ci=core.sparse_page_indices(ratio).reshape(q.shape[0],-1).index_select(0,ri).to('cpu',torch.int64)
        else:
            compressed=self._empty_kv;ci=torch.full((len(rows),512),-1,dtype=torch.int64)
        def selected_values(buf,idx):
            ids=sorted(set(i for i in idx.flatten().tolist() if i>=0))
            assert not ids or max(ids)<buf.shape[0]
            slots=torch.tensor(ids,dtype=torch.long)
            values=buf.index_select(0,slots.to(buf.device)).detach().cpu() if ids else torch.empty((0,512),dtype=buf.dtype)
            return slots,values
        oslots,ovals=selected_values(original,oi);cslots,cvals=selected_values(compressed,ci)
        data={'probe':'G1096_REAL_ATTENTION','nonce':arm['nonce'],'pid':os.getpid(),'layer_id':lid,'ratio':ratio,
              'mode':str(forward_batch.forward_mode),'call_in_ratio':count,'q_rows':rows,'full_q_shape':list(q.shape),
              'scaling':float(layer.scaling),'q':q.index_select(0,ri).detach().cpu(),
              'result':result.index_select(0,ri).detach().cpu(),'original_indices':oi,'compressed_indices':ci,
              'original_slots':oslots,'original_values':ovals,'compressed_slots':cslots,'compressed_values':cvals,
              'sink':(self._no_sink[:q.shape[1]] if attn_sink is None else attn_sink).detach().float().cpu(),
              'positions':forward_batch.positions.detach().cpu(),'input_ids':forward_batch.input_ids.detach().cpu(),
              'req_pool_indices':forward_batch.req_pool_indices.detach().cpu(),
              'seq_lens':forward_batch.seq_lens.detach().cpu(),'out_cache_loc':forward_batch.out_cache_loc.detach().cpu(),
              'scope':'Actual operands after original attention call; CPU copies synchronize diagnostic only. Not model correctness/performance proof.'}
        folder=root/'forward_probe'/arm['nonce'];folder.mkdir(parents=True,exist_ok=True)
        path=folder/f'rankpid{os.getpid()}_ratio{ratio}_call{count:02d}.pt';tmp=path.with_suffix('.tmp')
        assert not path.exists();torch.save(data,tmp);assert tmp.stat().st_size<=arm['maximum_snapshot_bytes']<=2097152;tmp.replace(path)
        with (folder/f'manifest.pid{os.getpid()}.jsonl').open('a') as f:
            f.write(json.dumps({'file':path.name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                'layer_id':lid,'ratio':ratio,'call':count,'q_rows':rows,'shape':list(q.shape),
                                'scope':data['scope']})+'\n')
        return result
    cls._forward_attention=observed;cls._forward_probe_g1096=True
    (root/f'forward_probe_install.pid{os.getpid()}.json').write_text(json.dumps({'original_backend_SHA256':_EXPECTED,'opt_in_arm_file':arm_file.name,'unarmed_math_unchanged':True,'eager_required_for_armed_capture':True},indent=2)+'\n')
