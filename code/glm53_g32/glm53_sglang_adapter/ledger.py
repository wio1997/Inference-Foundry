# SPDX-License-Identifier: Apache-2.0
"""Complete named load-coverage ledger, preserving native parameter loaders.

A returned loader call is coverage evidence, not a tensor-value correctness
claim. None-returning expert loaders may skip nonlocal EP experts internally;
those outcomes remain explicit rather than being invented as physical writes.
"""
import functools
import hashlib
import inspect
import sys
import json
import os
from pathlib import Path
import re
import time


def _scalar(x):
    return x if isinstance(x,(str,int,float,bool,type(None))) else {'type':type(x).__name__}


def _atomic(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)

_RMS_SOURCE_SHA = {
    'sglang.srt.models.glm4v.Glm4vRMSNorm':'d9f922c251eb24ca62bf7fe3724a01dbe1a8a4d428ca40db7b4b4c66e97b202f',
    'sglang.srt.layers.layernorm.RMSNorm':'b383408a7518f4136680f67647c07043f6df26d94efe92182e2071e3cee6b460',
    'sglang.srt.models.glm_ocr.GlmOcrRMSNorm':'9acab22342f46d398989fb7dfff2b242659ac87eef13194202ee5c1bad3f0495',
}
_MODELSLIM_SHA = '03fc670afe3cb769e05b4b459b6f08efa5c7001d4014b8812bdde87a15b6f7be'


def _checkpoint_name(name):
    # Exact target GLM loader prefix conversions, not a broad bias exemption.
    return name.replace('language_model.','').replace('model.visual.','visual.').replace('attn.qkv.','attn.qkv_proj.')


def _checkpoint_names(names, role, selected_layer):
    normalized={_checkpoint_name(name) for name in names}
    if role=='draft':
        prefix=f'model.layers.{selected_layer}'
        for name in tuple(normalized):
            if not name.startswith(prefix+'.'):continue
            # Exact is_nextn branch: these weights sit beside decoder under model.
            special=any(part in name for part in ('shared_head.norm','eh_proj','enorm','hnorm'))
            mapped=name.replace(prefix,'model' if special else 'model.decoder',1)
            normalized.add(mapped)
    return normalized


def _derived_rms_bias(model, name, param, index, blocked_names, role):
    proof={'parameter':name,'classification':'missing','reason':'not_a_proven_derived_constant'}
    if role not in ('target','draft') or not name.endswith('.bias') or not index:return None,proof
    import torch
    module_name=name.rsplit('.',1)[0]
    try:module=model.get_submodule(module_name)
    except (AttributeError,KeyError):return None,proof
    kind=type(module);fq=kind.__module__+'.'+kind.__name__
    if fq not in _RMS_SOURCE_SHA or getattr(sys.modules.get(kind.__module__),kind.__name__,None) is not kind:return None,proof
    chain=[]
    for cls in kind.__mro__:
        key=cls.__module__+'.'+cls.__name__
        if key.startswith('sglang.'):
            if key not in _RMS_SOURCE_SHA:return None,dict(proof,reason='unbound_norm_inheritance_source',module_class=key)
            path=Path(inspect.getfile(cls));actual=hashlib.sha256(path.read_bytes()).hexdigest()
            if actual!=_RMS_SOURCE_SHA[key]:return None,dict(proof,reason='norm_source_SHA_mismatch')
            chain.append({'class':key,'source_sha256':actual})
        if key=='sglang.srt.layers.layernorm.RMSNorm':break
    else:return None,proof
    base=next(c for c in kind.__mro__ if c.__module__+'.'+c.__name__=='sglang.srt.layers.layernorm.RMSNorm')
    ctor=base.__init__;ctorfile=Path(ctor.__code__.co_filename)
    if ctor.__name__!='init' or ctor.__qualname__!='npu_wrapper_rmsnorm_init.<locals>.init' or hashlib.sha256(ctorfile.read_bytes()).hexdigest()!=_MODELSLIM_SHA:
        return None,dict(proof,reason='actual_ModelSlim_init_wrapper_not_proven')
    if module._parameters.get('bias') is not param or getattr(module,'bias',None) is not param or not isinstance(param,torch.nn.Parameter):return None,dict(proof,reason='bias_identity_mismatch')
    weight=getattr(module,'weight',None)
    if not isinstance(weight,torch.nn.Parameter) or tuple(param.shape)!=tuple(weight.shape) or param.ndim!=1 or param.numel()==0 or tuple(param.shape)!=(getattr(module,'hidden_size',None),):return None,dict(proof,reason='bias_shape_mismatch')
    if param.dtype!=weight.dtype or param.dtype not in (torch.float32,torch.float16,torch.bfloat16) or param.device!=weight.device or param.requires_grad or getattr(module,'ignore_anti',None) is not True:
        return None,dict(proof,reason='bias_dtype_device_or_initialization_contract_mismatch')
    if name in blocked_names:
        return None,dict(proof,reason='checkpoint_or_iterator_contains_bias_key')
    if not bool(torch.all(param.detach()==0).item()):return None,dict(proof,reason='derived_bias_is_not_all_zero')
    result={'parameter':name,'classification':'derived_initialized','role':role,'module_class':fq,'module_source_chain':chain,
            'constructor_source_sha256':_MODELSLIM_SHA,'own_parameter_identity':True,'shape':list(param.shape),'dtype':str(param.dtype),
            'device':str(param.device),'requires_grad':False,'all_zero':True,'checkpoint_bias_key_absent':True,'name_mapping':'target_prefix_plus_selected_nextn_layer' if role=='draft' else 'target_prefix',
            'reason':'ModelSlim RMSNorm wrapper creates zero antiquant bias; checkpoint has no corresponding bias'}
    return result,None


class WeightLedger:
    def __init__(self,model,out,model_path,default_loader,role=None,rank=0):
        self.model=model;self.root=Path(out);self.root.mkdir(parents=True,exist_ok=True)
        self.role=role or ('draft' if 'NextN' in type(model).__name__ else 'target')
        self.prefix=f'{self.role}.rank{rank}.pid{os.getpid()}.{time.time_ns()}'
        self.path=self.root/(self.prefix+'.coverage.json')
        self.model_path=Path(model_path);self.default_loader=default_loader
        try:
            self.params=dict(model.named_parameters(remove_duplicate=False))
        except TypeError:
            self.params=dict(model.named_parameters())
        self.original={};self._mutations=[];self.instrumentation=[]
        self.calls={n:[] for n in self.params}
        self.sources={};self.current=None;self.error=None
        self.expected_index={};self.index_sha=None
        for name in ['quant_model_weights.safetensors.index.json','model.safetensors.index.json']:
            p=self.model_path/name
            if p.is_file():
                self.expected_index=json.loads(p.read_text()).get('weight_map',{})
                self.index_sha=hashlib.sha256(p.read_bytes()).hexdigest();break
        self.hidden_layers=int(getattr(model.config,'num_hidden_layers',0))

    @staticmethod
    def _loader_slot(param, loader):
        descriptor=inspect.getattr_static(type(param),'weight_loader',None)
        if isinstance(descriptor,property) and descriptor.fset is None:
            getter=descriptor.fget
            # Exact installed BasevLLMParameter contract: return self._weight_loader.
            if (getter is None or getter.__code__.co_names != ('_weight_loader',)
                    or not hasattr(param,'_weight_loader')
                    or getattr(param,'_weight_loader') is not loader):
                raise TypeError('Unsupported readonly weight_loader descriptor; refusing incomplete coverage')
            return '_weight_loader'
        return 'weight_loader'

    def _restore(self):
        errors=[]
        for param,attr,had_own,old in reversed(self._mutations):
            try:
                if had_own:setattr(param,attr,old)
                else:delattr(param,attr)
            except Exception as exc:errors.append(repr(exc))
        self._mutations=[]
        if errors:raise RuntimeError('Failed to restore native loader state: '+repr(errors))

    def __enter__(self):
        groups={}
        for name,param in self.params.items():
            groups.setdefault(id(param),{'param':param,'names':[]})['names'].append(name)
        plans=[]
        try:
            # Preflight every descriptor before touching the first parameter.
            for group in groups.values():
                param=group['param'];names=tuple(group['names'])
                loader=getattr(param,'weight_loader',self.default_loader)
                if not callable(loader):
                    self.instrumentation.append({'aliases':names,'mode':'noncallable_missing_coverage'})
                    continue
                attr=self._loader_slot(param,loader)
                plans.append((param,names,loader,attr))
            for param,names,loader,attr in plans:
                def traced(*args,_names=names,_loader=loader,**kwargs):
                    record={'source':self.current,'parameter_aliases':list(_names),
                            'positional_components':[_scalar(v) for v in args[2:]],
                            'keyword_components':{k:_scalar(v) for k,v in kwargs.items()},'outcome':'pending'}
                    try:
                        result=_loader(*args,**kwargs)
                        record['outcome']='reported_skipped' if result is False else 'reported_loaded' if result is True else 'returned_no_write_assertion'
                    except Exception as exc:
                        record['outcome']='exception';record['exception']=f'{type(exc).__name__}: {exc}';raise
                    finally:
                        for name in _names:self.calls[name].append(record)
                        if self.current in self.sources:
                            self.sources[self.current]['dispatches'].append({'parameter':_names[0],**record})
                            for name in _names:
                                for before,after in [('.q_a_proj.','.kv_a_proj_with_mqa.'),('.kv_a_proj_with_mqa.','.q_a_proj.')]:
                                    if 'fused_qkv_a_proj_with_mqa' in name and before in self.current:
                                        sibling=self.current.replace(before,after)
                                        if sibling in self.sources:
                                            self.sources[sibling]['dispatches'].append({'parameter':name,'source':sibling,'outcome':record['outcome'],'fused_with':self.current})
                    return result
                had_own=attr in vars(param);old=vars(param).get(attr)
                self._mutations.append((param,attr,had_own,old))
                setattr(param,attr,traced)
                if getattr(param,'weight_loader') is not traced:
                    raise RuntimeError('Loader interception did not preserve descriptor semantics')
                self.instrumentation.append({'aliases':list(names),'mode':'backing_attribute' if attr=='_weight_loader' else 'public_attribute','attribute':attr})
            return self
        except Exception as exc:
            self._restore()
            self.finish(f'instrumentation_setup_failed: {type(exc).__name__}: {exc}')
            raise

    def stream(self,weights):
        for name,tensor in weights:
            self.current=name
            self.sources.setdefault(name,{'shape':list(tensor.shape),'dtype':str(tensor.dtype),'dispatches':[]})
            yield name,tensor
        self.current=None

    def _reason(self,name):
        layer=re.search(r'(?:^|\.)layers\.(\d+)\.',name)
        if name=='rot.weight' or name.startswith('rot.'):
            return 'auxiliary_rotation_loaded_from_file' if self.role=='draft' and getattr(self.model,'_glm53_rotation_evidence',{}).get('loaded') else 'target_export_rotation_folded'
        if 'rotary_emb.inv_freq' in name:return 'derived_rotary_frequencies'
        if self.role=='target' and layer and int(layer.group(1))>=self.hidden_layers:return 'draft_layer_outside_target'
        if self.role=='draft' and (not layer or int(layer.group(1))!=self.hidden_layers):return 'outside_selected_nextn_layer'
        if self.role=='target' and getattr(self.model,'language_only',False) and 'visual' in name:return 'vision_excluded_by_language_only'
        return None

    def finish(self,exception=None):
        self.error=exception
        loaded=[];missing=[];deferred=[];derived=[];derived_rejections=[]
        blocked_names=_checkpoint_names(set(self.expected_index)|set(self.sources),self.role,self.hidden_layers)
        for name in self.params:
            successful=[c for c in self.calls[name] if c['outcome'] in ['reported_loaded','returned_no_write_assertion']]
            if successful:loaded.append(name)
            elif self.role=='draft' and name in ['model.embed_tokens.weight','lm_head.weight']:
                deferred.append(name)
            else:
                constant,rejection=_derived_rms_bias(self.model,name,self.params[name],self.expected_index,blocked_names,self.role)
                if constant is not None:derived.append(constant)
                else:
                    missing.append(name)
                    if name.endswith('.bias'):derived_rejections.append(rejection)
        unexpected=[];excluded=[]
        for name,row in self.sources.items():
            if row['dispatches']:row['classification']='dispatched'
            else:
                reason=self._reason(name)
                row['classification']='intentional_exclusion' if reason else 'unexpected_undispatched'
                row['reason']=reason
                (excluded if reason else unexpected).append(name)
        unseen=sorted(set(self.expected_index)-set(self.sources))
        post_params=dict(self.model.named_parameters())
        post_new={name:{'shape':list(param.shape),'dtype':str(param.dtype)} for name,param in post_params.items() if name not in self.params or param is not self.params[name]}
        result={'schema':'glm53_weight_ledger_g11','derived_initialized':derived,'derived_rejections':derived_rejections,'derived_initialized_count':len(derived),'instrumentation':self.instrumentation,'role':self.role,'meaning':'Native loader dispatch coverage plus separately proven initialized constants; not tensor-value correctness or physical EP writes',
                'checkpoint_index_sha256':self.index_sha,'checkpoint_index_key_count':len(self.expected_index),
                'post_load_new_or_replaced_parameters':post_new,'provided_iterator_scope':True,'source_records':self.sources,'parameter_calls':self.calls,'loaded_parameters':loaded,'missing_parameters':missing,
                'unexpected_undispatched':unexpected,'intentional_exclusions':excluded,'unseen_index_keys':unseen,
                'deferred_target_shared_parameters':deferred,'auxiliary_rotation':getattr(self.model,'_glm53_rotation_evidence',None),
                'exception':exception,'component_coverage':'per-call shard_id/expert_id retained; None-returning native loaders do not establish per-expert writes',
                'status':'failed' if exception or missing or unexpected else 'awaiting_target_shares' if deferred else 'dispatch_coverage_complete'}
        _atomic(self.path,result)
        summary={k:v for k,v in result.items() if k not in ['source_records','parameter_calls']}
        summary.update(coverage_file=self.path.name,loaded_parameter_count=len(loaded),missing_parameter_count=len(missing),unexpected_count=len(unexpected),source_key_count=len(self.sources))
        _atomic(self.root/(self.prefix+'.summary.json'),summary)
        self.model._glm53_weight_ledger_path=str(self.path)
        return result

    def __exit__(self,kind,value,traceback):
        self._restore()
        self.finish(None if value is None else f'{kind.__name__}: {value}')
        return False


def audited_load(fn):
    @functools.wraps(fn)
    def wrapper(model,weights,*args,**kwargs):
        out=os.environ.get('GLM53_LOAD_LEDGER_DIR')
        if not out or getattr(model,'_glm53_active_ledger',None) is not None:
            return fn(model,weights,*args,**kwargs)
        from sglang.srt.runtime_context import get_model
        from sglang.srt.model_loader.weight_utils import default_weight_loader
        try:
            import torch.distributed as dist
            rank=dist.get_rank() if dist.is_initialized() else 0
        except Exception:rank=0
        ledger=WeightLedger(model,out,get_model().model_path,default_weight_loader,rank=rank)
        model._glm53_active_ledger=ledger
        try:
            with ledger:result=fn(model,ledger.stream(weights),*args,**kwargs)
            if os.environ.get('GLM53_LOAD_LEDGER_STRICT','1')=='1':
                report=json.loads(ledger.path.read_text())
                if report['missing_parameters'] or report['unexpected_undispatched']:
                    raise RuntimeError(f'GLM53 weight coverage failed; inspect {ledger.path}')
            return result
        finally:model._glm53_active_ledger=None
    return wrapper


def record_target_shares(model,embed,head):
    path=getattr(model,'_glm53_weight_ledger_path',None)
    if not path:return
    p=Path(path);result=json.loads(p.read_text())
    result['target_shares']={'model.embed_tokens.weight':{'shape':list(embed.shape),'identity_bound':model.model.embed_tokens.weight is embed},
                             'lm_head.weight':{'shape':list(head.shape),'identity_bound':model.lm_head.weight is head}}
    if not all(v['identity_bound'] for v in result['target_shares'].values()):raise RuntimeError('NextN target share binding failed')
    result['deferred_target_shared_parameters']=[]
    if not result['exception'] and not result['missing_parameters'] and not result['unexpected_undispatched']:
        result['status']='dispatch_coverage_complete'
    _atomic(p,result)
    summary_path=p.with_name(p.name.replace('.coverage.json','.summary.json'))
    if summary_path.exists():
        summary=json.loads(summary_path.read_text());summary['target_shares']=result['target_shares'];summary['deferred_target_shared_parameters']=[];summary['status']=result['status'];_atomic(summary_path,summary)
