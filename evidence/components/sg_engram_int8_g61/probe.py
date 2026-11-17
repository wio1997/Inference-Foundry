"""Real TP8 integration fixture; not model serving or performance acceptance."""
from pathlib import Path
from types import SimpleNamespace
import datetime
import hashlib
import inspect
import json
import os
import traceback

import torch
import torch_npu
import torch.distributed as dist
from safetensors.torch import save_file, load_file

import adapter
import source_uva

rank=int(os.environ['RANK']);assert rank==int(os.environ['LOCAL_RANK'])
os.sched_setaffinity(0,{168+rank});torch.set_num_threads(1);torch.npu.set_device(rank)
R=Path(__file__).resolve().parent;E=R/'evidence';E.mkdir(exist_ok=True)
record={'rank':rank,'pid':os.getpid(),'scope':'Synthetic indexed INT8 tables through native SGLang embedding/Engram methods and real EP8-device TP allreduce; no complete model/PD/performance acceptance','phase':'imports'}


def save():
    p=E/f'rank{rank}.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(record,indent=2));q.replace(p)


def digest(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def oracle_gate(x,kv,q,k,rotation,ids):
    # Independent full-matrix FP64 inverse, not the adapter's block32 expression.
    h=x.double();hc=h.shape[-2];dim=h.shape[-1]
    original=h@rotation.double().T
    key,value=kv.double().split([hc*dim,dim],dim=-1);key=key.reshape(-1,hc,dim)
    norm_h=(original.square().sum(-1)/dim+1e-20).sqrt()
    norm_k=(key.square().sum(-1)/dim+1e-20).sqrt()
    dot=(original*key*q.double()*k.double()).sum(-1)/(norm_h*norm_k*(dim**.5))
    signed=torch.where(dot<0,-torch.ones_like(dot),torch.ones_like(dot))*dot.abs().clamp_min(1e-6).sqrt()
    out=(h+torch.sigmoid(signed).unsqueeze(-1)*value.unsqueeze(-2)).bfloat16()
    return torch.where((ids==129264)[:,None,None],x,out)


try:
    save()
    dist.init_process_group('hccl',init_method='env://',timeout=datetime.timedelta(seconds=180))
    cpu_group=dist.new_group(backend='gloo')
    root=R/'fixture';n=3072;dim=256;hidden=5120
    if rank==0:
        root.mkdir(exist_ok=False);(root/'optional').mkdir()
        config={'model_type':'deepseek_v41','text_config':{'hidden_size':hidden,'engram_layer_ids':[1,14]},
                'engram_rotation_config':{'value_projection_rotated':True,'value_basis':'quarot_global','key_and_gate_basis':'original','runtime_delta_rotation':False}}
        (root/'config.json').write_text(json.dumps(config));weights={};index={};quant={}
        g=torch.Generator().manual_seed(103)
        for layer in [1,14]:
            prefix=f'layers.{layer}.engram.embed';codes=torch.randint(-127,128,(n,dim),generator=g,dtype=torch.int8)
            raw=torch.randint(120,135,(n,8),generator=g,dtype=torch.uint8)
            weights[prefix+'.weight']=codes;weights[prefix+'.scale']=raw.view(torch.float8_e8m0fnu)
            index[prefix+'.weight']=index[prefix+'.scale']='tables.safetensors';quant[prefix+'.quant_type']='INT8_SYM'
        save_file(weights,root/'tables.safetensors')
        (root/'quant_model_weights.safetensors.index.json').write_text(json.dumps({'weight_map':index}))
        (root/'quant_model_description.json').write_text(json.dumps(quant))
        block=torch.ones((1,1),dtype=torch.float32)
        for _ in range(5):block=torch.cat([torch.cat([block,block],1),torch.cat([block,-block],1)],0)
        block=(block/(32**.5)).bfloat16();rotation=torch.block_diag(*[block]*(hidden//32))
        save_file({'global_rotation':rotation},root/'optional/quarot.safetensors')
    dist.barrier(group=cpu_group)
    import sglang.srt.layers.engram as native
    before_hash=hashlib.sha256(inspect.getsource(native.EngramHasher).encode()).hexdigest()
    before_lookup=native.EngramEmbedding.forward
    before_native_lookup=native.EngramEmbedding._lookup
    # Harness supplies the actual all-eight-device process group to the native
    # TP seam; serving uses its normal RuntimeContext/group publication.
    native.get_parallel=lambda:SimpleNamespace(tp_size=8,tp_rank=rank,attn_dp_size=1,attn_cp_size=1)
    def allreduce(t):dist.all_reduce(t);return t
    native.tensor_model_parallel_all_reduce=allreduce
    source_uva.CHUNK_ROWS=128
    installed=adapter.install(root)
    assert before_hash==hashlib.sha256(inspect.getsource(native.EngramHasher).encode()).hexdigest()
    assert native.EngramEmbedding.forward is before_lookup
    assert native.EngramEmbedding._lookup is before_native_lookup
    record.update(phase='installed',device_name=torch.npu.get_device_name(rank),torch=torch.__version__,
                  torch_npu=torch_npu.__version__,adapter=installed,hasher_source_sha256=before_hash,
                  native_lookup_method_identity_preserved=True,native_forward_method_identity_preserved=True)
    save()
    tables=load_file(root/'tables.safetensors',device='cpu');boundaries=tuple(range(0,n+1,128))
    lookups=[]
    for layer in [1,14]:
        embed=native.EngramEmbedding(n,dim,layer,boundaries)
        prefix=f'layers.{layer}.engram.embed'
        callbacks=[(embed.scale,tables[prefix+'.scale']),(embed.weight,tables[prefix+'.weight'])]
        if layer==14:callbacks.reverse()
        for param,tensor in callbacks:param.weight_loader(param,tensor)
        embed.finish_load(str(layer));assert embed.weight.device.type==embed.scale.device.type=='cpu'
        assert embed.weight.dtype==torch.int8 and embed.scale.dtype==torch.float32
        expected_scales=tables[prefix+'.scale'].float()
        assert torch.equal(embed.scale,expected_scales[embed.row_start:embed.row_start+embed.rows])
        checks=[]
        for tokens in [0,1,16,80]:
            ids_cpu=(torch.arange(tokens*24,dtype=torch.int64).reshape(tokens,24)*17+37)%n
            if tokens:ids_cpu[0,0]=-1;ids_cpu[-1,-1]=n+13
            ids=ids_cpu.to(f'npu:{rank}');values=embed(ids);torch.npu.synchronize()
            valid=(ids_cpu>=0)&(ids_cpu<n);safe=ids_cpu.clamp(0,n-1)
            cpu=(tables[prefix+'.weight'][safe].double().reshape(tokens,24,8,32)*expected_scales[safe].double().unsqueeze(-1)).reshape(tokens,24,dim).bfloat16()
            cpu.masked_fill_(~valid.unsqueeze(-1),0);assert torch.equal(values.cpu(),cpu)
            checks.append({'tokens':tokens,'bitwise_bf16_match':True,'hash':digest(values)})
        ids=torch.zeros((16,24),dtype=torch.int64,device=f'npu:{rank}')
        for _ in range(2):embed(ids)
        graph=torch.npu.NPUGraph()
        with torch.npu.graph(graph,pool=torch.npu.graph_pool_handle(),auto_dispatch_capture=True):out=embed(ids)
        hashes=[]
        for replay in range(3):
            newids=(torch.arange(16*24).reshape(16,24)*31+replay*107)%n
            ids.copy_(newids);dist.barrier(group=cpu_group);graph.replay();torch.npu.synchronize()
            expected=(tables[prefix+'.weight'][newids].double().reshape(16,24,8,32)*expected_scales[newids].double().unsqueeze(-1)).reshape(16,24,dim).bfloat16()
            assert torch.equal(out.cpu(),expected);hashes.append(digest(out))
        assert len(set(hashes))==3
        lookups.append({'layer':layer,'load':embed.load_receipt,'callbacks':len(embed._callbacks),
                        'lookup_cases':checks,'graph_changed_inputs':hashes,'all_bitwise_bf16':True})
        del graph;torch.npu.synchronize();embed.close_host_offload();embed.close_host_offload()
        assert embed.weight.numel()==embed.scale.numel()==0
    record.update(phase='TP8_lookup_passed',lookups=lookups);save()
    # Exercise the original native Engram constructor/forward too, at real hidden5120.
    layout=SimpleNamespace(layer_ids=[1,14],num_embeddings=[n,n],head_dim=256,max_ngram_size=4,n_heads=8,
                           column_bounds=lambda _:boundaries)
    config=SimpleNamespace(hidden_size=hidden,hc_mult=4,rms_norm_eps=1e-20)
    with torch.device(f'npu:{rank}'):
        torch.set_default_dtype(torch.bfloat16)
        engram=native.Engram(config,1,layout,image_token_id=129264)
    for param,tensor in [(engram.embed.weight,tables['layers.1.engram.embed.weight']),
                         (engram.embed.scale,tables['layers.1.engram.embed.scale'])]:param.weight_loader(param,tensor)
    engram.embed.finish_load('native_forward')
    engram.wkv.weight.data.fill_(0.001)
    g=torch.Generator().manual_seed(21);x_cpu=torch.randn((3,4,hidden),generator=g,dtype=torch.float32).bfloat16()
    ids_cpu=torch.arange(72,dtype=torch.int64).reshape(3,24)*29%n
    inputs_cpu=torch.tensor([10,129264,12],dtype=torch.int64)
    x=x_cpu.to(f'npu:{rank}');ids=ids_cpu.to(f'npu:{rank}');inputs=inputs_cpu.to(f'npu:{rank}')
    out=engram(x,ids,input_ids=inputs);torch.npu.synchronize()
    emb=engram.embed(ids);kv,_=engram.wkv(emb.flatten(-2));torch.npu.synchronize()
    rotation=load_file(root/'optional/quarot.safetensors')['global_rotation']
    expected=oracle_gate(x_cpu,kv.cpu(),engram.q_weight.cpu(),engram.k_weight.cpu(),rotation,inputs_cpu)
    torch.testing.assert_close(out.cpu().float(),expected.float(),rtol=.008,atol=.008)
    assert torch.equal(out.cpu()[1],x_cpu[1])
    wrong=adapter.rotated_gate(x,kv,engram.q_weight,engram.k_weight,1e-20,1e-6,torch.eye(32,device=x.device),inputs,129264)
    assert not torch.equal(out,wrong),'Rotation-sensitive fixture must reject identity-basis gate'
    graph=torch.npu.NPUGraph()
    for _ in range(2):engram(x,ids,input_ids=inputs)
    with torch.npu.graph(graph,pool=torch.npu.graph_pool_handle(),auto_dispatch_capture=True):out_graph=engram(x,ids,input_ids=inputs)
    replay_hashes=[]
    for replay in range(3):
        x_cpu.add_(.125);x.copy_(x_cpu);ids_cpu.add_(31).remainder_(n);ids.copy_(ids_cpu)
        dist.barrier(group=cpu_group);graph.replay();torch.npu.synchronize()
        emb=engram.embed(ids);kv,_=engram.wkv(emb.flatten(-2));torch.npu.synchronize()
        expected=oracle_gate(x_cpu,kv.cpu(),engram.q_weight.cpu(),engram.k_weight.cpu(),rotation,inputs_cpu)
        torch.testing.assert_close(out_graph.cpu().float(),expected.float(),rtol=.008,atol=.008)
        assert torch.equal(out_graph.cpu()[1],x_cpu[1]);replay_hashes.append(digest(out_graph))
    assert len(set(replay_hashes))==3
    del graph;engram.embed.close_host_offload()
    record.update(phase='passed',native_Engram_forward_hidden5120=True,rotation_identity_negative_control_rejected=True,
                  native_forward_changed_input_graph_hashes=replay_hashes,gate_tolerance={'rtol':.008,'atol':.008},
                  codes_and_scale_load_callbacks_both_orders=True,host_buffer_aliases_released=True)
    save();dist.barrier(group=cpu_group);dist.destroy_process_group()
except BaseException as exc:
    record.update(phase='failed',error=repr(exc),traceback=traceback.format_exc());save();raise
