"""Opt-in attention-boundary operands for one exact teacher-forced prefix.
The original method is called once; no operand or result is replaced.
All real rows are copied to CPU, so armed calls are diagnostic only.
"""
from pathlib import Path
import datetime,functools,hashlib,inspect,json,os


def install(cls, framework, x_name, expected_source_sha, layer_name):
    import torch
    from safetensors.torch import save_file
    root=Path(os.environ['DS41_LAYER_PROBE_ROOT'])
    source=Path(inspect.getfile(cls));assert hashlib.sha256(source.read_bytes()).hexdigest()==expected_source_sha
    prior=cls.forward;assert not getattr(prior,'_ds41_layer_probe',False)
    signature=inspect.signature(prior);seen=set()
    @functools.wraps(prior)
    def observed(self,*args,**kwargs):
        armfile=root/'layer_probe_ARM.json'
        if not armfile.exists():return prior(self,*args,**kwargs)
        arm=json.loads(armfile.read_text());assert arm['probe']=='G1145_SAME_PREFIX_ATTENTION' and arm['max_layers']==40
        bound=signature.bind(self,*args,**kwargs);bound.apply_defaults();x=bound.arguments[x_name];pos=bound.arguments['positions'];n=len(arm['input_ids'])
        # Decode/idle/padding-only calls are excluded, never treated as prefill.
        if pos.numel()<n or x.shape[0]<n:return prior(self,*args,**kwargs)
        positions=pos[:n].detach().cpu().contiguous()
        if positions.tolist()!=list(range(n)):return prior(self,*args,**kwargs)
        if framework=='sglang':
            fb=bound.arguments['forward_batch'];assert fb.forward_mode.is_extend_without_speculative()
            actualids=fb.input_ids[:n].detach().cpu().tolist();assert actualids==arm['input_ids']
        assert x.device.type=='npu' and x.dtype==torch.bfloat16 and x.shape[-1]==5120 and x.ndim==2
        layer=int(self.role.layer_idx if framework=='vllm' else getattr(self,layer_name));assert 0<=layer<40
        key=(arm['nonce'],layer);assert key not in seen,'Only one prefill per nonce; no retry'
        seen.add(key);rank=torch.distributed.get_rank();assert 0<=rank<8
        before=x[:n].detach().cpu().contiguous().clone()
        result=prior(self,*args,**kwargs)
        partial=hasattr(result,'partial');value=result.partial if partial else result
        assert isinstance(value,torch.Tensor) and value.ndim==2 and value.shape[0]>=n and value.shape[-1]==5120
        after=value[:n].detach().cpu().contiguous().clone()
        folder=root/'layer_probe'/arm['nonce'];folder.mkdir(parents=True,exist_ok=True)
        stem=f'rank{rank}_layer{layer:02d}';path=folder/(stem+'.safetensors');meta=folder/(stem+'.json');assert not path.exists() and not meta.exists()
        save_file({'input':before,'output':after,'positions':positions},str(path))
        assert path.stat().st_size<=4*1024**2
        record={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'framework':framework,'rank':rank,'pid':os.getpid(),'layer':layer,'nonce':arm['nonce'],'input_ids':arm['input_ids'],'positions':positions.tolist(),'real_rows':n,'full_input_shape':list(x.shape),'full_output_shape':list(value.shape),'output_is_TP_partial':partial,'original_source_SHA256':expected_source_sha,'probe_source_SHA256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'file':path.name,'file_SHA256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,'scope':'Actual attention normalized-input and projected-output, real prompt rows only. Observer CPU copies synchronize; not performance or solecause proof.'}
        meta.write_text(json.dumps(record,indent=2)+'\n')
        return result
    observed._ds41_layer_probe=True;cls.forward=observed
    (root/f'layer_probe_install.pid{os.getpid()}.json').write_text(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'framework':framework,'class':cls.__name__,'source_SHA256':expected_source_sha,'prior_signature':str(signature),'wrapper_signature_preserved':str(inspect.signature(observed))==str(signature),'ARM_present':(root/'layer_probe_ARM.json').exists(),'math_unchanged_when_unarmed':True},indent=2)+'\n')


def install_native_prefix_guard(cls):
    import torch
    root=Path(os.environ['DS41_LAYER_PROBE_ROOT']);prior=cls.forward;signature=inspect.signature(prior)
    @functools.wraps(prior)
    def observed(self,*args,**kwargs):
        armfile=root/'layer_probe_ARM.json'
        if armfile.exists():
            arm=json.loads(armfile.read_text());n=len(arm['input_ids']);bound=signature.bind(self,*args,**kwargs);bound.apply_defaults();ids=bound.arguments['input_ids'];pos=bound.arguments['positions']
            if pos.numel()>=n and pos[:n].detach().cpu().tolist()==list(range(n)):
                actual=ids[:n].detach().cpu().tolist();assert actual==arm['input_ids']
                rank=torch.distributed.get_rank();folder=root/'layer_probe'/arm['nonce'];folder.mkdir(parents=True,exist_ok=True);p=folder/f'rank{rank}_actual_prefix.json';assert not p.exists();p.write_text(json.dumps({'rank':rank,'pid':os.getpid(),'input_ids':actual,'positions':list(range(n)),'scope':'Actual native model input prefix before any layer'},indent=2)+'\n')
        return prior(self,*args,**kwargs)
    cls.forward=observed
