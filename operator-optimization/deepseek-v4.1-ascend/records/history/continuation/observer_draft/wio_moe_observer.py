"""Save same-call MC2 counts/mask/top-k/ordinal; never modify model tensors."""
import functools
import hashlib
import json
import os
from pathlib import Path
import re
import time

_counter={}
_python_calls={}
_eager_calls={}
_eager_total={}
_capture_total=0
_installed=False

def _write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(path)

def _observe(comm,inp,result,module):
    import torch
    import torch.distributed as dist
    from vllm.distributed import get_ep_group,get_dp_group,get_tp_group
    from vllm_ascend.ascend_config import get_ascend_config,is_mega_moe_supported
    global _capture_total
    assert inp is not None and inp.layer is not None,'Observer cannot bind routed layer'
    counts=result.expert_tokens
    assert isinstance(counts,torch.Tensor) and counts.dtype==torch.int32 and counts.ndim==1
    x=inp.hidden_states;topk=inp.topk_ids;mask=inp.routing.mc2_mask
    assert topk.ndim==2 and topk.shape[0]==x.shape[0]
    assert counts.device==topk.device==x.device
    assert mask is None or (mask.device==x.device and mask.dtype in (torch.bool,torch.int8) and mask.numel()<=8192)
    layer=inp.layer;layer_name=getattr(layer,'layer_name',None);assert isinstance(layer_name,str) and layer_name
    ctx=module._EXTRA_CTX;role='draft' if bool(ctx.is_draft_model) else 'target'
    ep=get_ep_group();ep_rank=int(ep.rank_in_group);world=int(ep.world_size)
    assert world==int(comm.token_dispatcher.ep_world_size)
    group=str(comm.token_dispatcher.moe_all_to_all_group_name)
    key=(role,layer_name,group,str(x.device));M=int(x.shape[0]);K=int(topk.shape[1]);E=int(counts.numel())
    assert M<=512 and E in (8,24),'Observed custom fused shape outside frozen intended domain'
    # Per-layer observer ordinal counts eager calls and actual captured replays.
    # It is NOT a scheduler/request ID. Cross-rank alignment must be validated.
    if key not in _counter:_counter[key]=torch.zeros((1,),dtype=torch.int64,device=x.device)
    ordinal=_counter[key];ordinal.add_(1)
    capturing=bool(torch.npu.is_current_stream_capturing())
    root=Path(os.environ['WIO_OBSERVER_ROOT']);arm=root/'ARMED.json'
    if capturing:
        _capture_total+=1;assert _capture_total<=2048,'Capture-path budget exceeded'
    else:
        if not arm.exists():return
        phase=arm.stat().st_mtime_ns
        ekey=(key,M,K,phase);rkey=(ep_rank,phase)
        if _eager_calls.get(ekey,0)>=3 or _eager_total.get(rkey,0)>=256:return
        _eager_calls[ekey]=_eager_calls.get(ekey,0)+1;_eager_total[rkey]=_eager_total.get(rkey,0)+1
    _python_calls[key]=_python_calls.get(key,0)+1;entry=_python_calls[key]
    safe=re.sub(r'[^a-zA-Z0-9_.-]','_',layer_name)
    rankroot=root/f'ep{ep_rank:02d}-device{int(x.device.index)}';rankroot.mkdir(parents=True,exist_ok=True)
    stem=f'{role}-{safe}-M{M}-K{K}-entry{entry:06d}'
    tensors=[counts,topk,ordinal];names=['expert_counts','topk_ids','observer_ordinal']
    if mask is not None:tensors.append(mask);names.append('x_active_mask')
    # CPU maps are serialized as static binding metadata; NPU maps travel in the
    # same snapshot. Never insert a CPU tensor into a device TensorList.
    maps={}
    for name in ['global_expert_map','ascend_expert_map','expert_map','expert_global_to_physical','expert_physical_to_global','expert_local_to_global']:
        try:value=getattr(layer,name,None)
        except AttributeError:continue
        if not isinstance(value,torch.Tensor):continue
        if value.numel()>8192:
            maps[name]={'shape':list(value.shape),'dtype':str(value.dtype),'device':str(value.device),'values_omitted_by_metadata_budget':True};continue
        if value.device.type=='cpu':maps[name]={'device':'cpu','shape':list(value.shape),'dtype':str(value.dtype),'values':value.tolist()}
        elif value.device==x.device:
            tensors.append(value);names.append(name);maps[name]={'device':str(value.device),'shape':list(value.shape),'dtype':str(value.dtype),'payload_index':len(tensors)-1}
    cfg=get_ascend_config()
    meta={'schema':1,'role':role,'layer_name':layer_name,'ep_rank':ep_rank,'ep_world_size':world,'dp_rank':int(get_dp_group().rank_in_group),'tp_rank':int(get_tp_group().rank_in_group),'global_rank':int(dist.get_rank()),'pid':os.getpid(),'device':str(x.device),'python_entry':entry,'capturing':capturing,'python_entry_monotonic_ns_not_replay_timestamp':time.monotonic_ns(),'M':M,'K':K,'E_local':E,'E_global':int(layer.global_num_experts),'mask_is_none':mask is None,'mask_shape':None if mask is None else list(mask.shape),'input_dtype':str(x.dtype),'topk_dtype':str(topk.dtype),'counts_dtype':str(counts.dtype),'enable_fused_mc2':int(comm.enable_fused_mc2),'use_mega_moe':bool(ctx.use_mega_moe),'is_mega_moe_supported':bool(is_mega_moe_supported()),'capacity_max_output_size':int(cfg.mega_moe_max_tokens),'group':group,'ep_group_ranks':list(getattr(ep,'ranks',[])),'comm_object_id':id(comm),'counts_data_ptr':counts.data_ptr(),'topk_data_ptr':topk.data_ptr(),'mask_data_ptr':None if mask is None else mask.data_ptr(),'placement_strategy':str(getattr(layer,'expert_placement_strategy',None)),'global_redundant_experts':int(getattr(layer,'global_redundant_expert_num',0)),'payload_names':names,'maps':maps,'snapshot_semantics':'immediately after returned FusedExpertsResult, before next consumer/shared buffer overwrite; registered TensorList graph helper','ordinal_semantics':'per rank/layer observer invocation ordinal, not scheduler step; captured path replays reuse filenames','inference_outputs_modified':False}
    _write(rankroot/(stem+'.json'),meta)
    torch.ops.npu.save_npugraph_tensor.tensor_list(tensors,save_path=str(rankroot/(stem+'.pt')))

def install(module):
    global _installed
    if _installed:return
    cls=module.FusedMC2CommImpl;original=cls.fused_experts
    @functools.wraps(original)
    def observed(self,*args,**kwargs):
        inp=kwargs.get('fused_experts_input',args[0] if args else None)
        result=original(self,*args,**kwargs)
        _observe(self,inp,result,module)
        return result
    cls.fused_experts=observed;_installed=True
    root=Path(os.environ['WIO_OBSERVER_ROOT']);root.mkdir(parents=True,exist_ok=True)
    _write(root/('instrumentation-installed-pid'+str(os.getpid())+'.json'),{'module':module.__name__,'module_file':module.__file__,'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'class':'FusedMC2CommImpl','method':'fused_experts','pid':os.getpid(),'outputs_unchanged':True,'real_tensor_probe_required':True})
