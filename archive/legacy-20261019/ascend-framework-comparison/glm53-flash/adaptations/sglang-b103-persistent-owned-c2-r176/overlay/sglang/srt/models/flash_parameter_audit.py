# SPDX-License-Identifier: Apache-2.0
"""Diagnostic native GLM weight-loader coverage. No tensor values are logged.

Wrap the existing loaders, preserve their arguments/results, and record actual
source and target metadata. The optional postprocess stop is a deliberate gate
before model warmup/requests, not successful model/service qualification.
"""
import contextvars,functools,gzip,hashlib,json,os,pathlib,re,collections,sys
_CURRENT=contextvars.ContextVar('flash_parameter_source',default=None)
_AUDITS={}


def _meta(t):
    return {'shape':list(t.shape),'dtype':str(t.dtype),'device':str(t.device)}


class _Audit:
    def __init__(self,model):
        import torch
        assert torch.distributed.is_initialized()
        self.rank=torch.distributed.get_rank();self.world=torch.distributed.get_world_size()
        assert self.world==16 and 0<=self.rank<16
        self.root=pathlib.Path(os.environ['FLASH_PARAMETER_AUDIT_DIR']);self.root.mkdir(parents=True,exist_ok=True)
        self.path=self.root/('weight-audit-rank'+str(self.rank)+'.jsonl.gz')
        fd=os.open(self.path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        self.raw=os.fdopen(fd,'wb');self.log=gzip.GzipFile(fileobj=self.raw,mode='wb',compresslevel=1,mtime=0)
        self.count=0;self.bytes=0;self.sources={};self.covered=set();self.targets=collections.Counter();self.pending={};self.error=None;self.linear_scale_expected={};self.linear_scale_parts=collections.Counter();self.linear_scale_checks=0
        self.params=dict(model.named_parameters());self.names={id(v):k for k,v in self.params.items()}
        self.initial={k:_meta(v) for k,v in self.params.items()}
        typename=lambda v:type(v).__module__+'.'+type(v).__name__
        self.modules={k:{'module':typename(v),'quant_method':typename(getattr(v,'quant_method',None)),'scheme':typename(getattr(v,'scheme',None)),'scheme_dynamic':getattr(getattr(v,'scheme',None),'is_dynamic',None),'scheme_kernel':typename(getattr(getattr(v,'scheme',None),'kernel',None))} for k,v in model.named_modules()}
        manifest=json.loads(pathlib.Path('/run-evidence/combined-manifest.json').read_text());self.bindings={}
        for row in manifest['files']:
            p=pathlib.Path('/sgl-workspace/sglang/python')/row['target'];sha=hashlib.sha256(p.read_bytes()).hexdigest();assert sha==row['sha256'];name=row['target'].removesuffix('.py').replace('/','.')
            module=sys.modules.get(name);self.bindings[name]={'sha256':sha,'path':str(p),'imported':module is not None,'actual_module_path':getattr(module,'__file__',None)}
        self.event('runtime_bindings',bindings=self.bindings,quant_metadata_entries=len(model.quant_config.quant_description))
        self.zero_bias_defaults={k+'.bias' if k else 'bias' for k,v in model.named_modules() if getattr(v,'ignore_anti',False) and getattr(v,'bias',None) is not None}
        self.mapper=model.hf_to_sglang_mapper
        with gzip.open(os.environ['FLASH_PARAMETER_SCHEMA'],'rt') as f:self.expected=json.load(f)
        self.event('start',world=self.world,params=self.initial,modules=self.modules,zero_bias_defaults=sorted(self.zero_bias_defaults))
        self.log.flush();self.raw.flush()

    def invoke(self,original,*args,**kwargs):
        source=_CURRENT.get();target=args[0];value=args[1]
        name=self.names[id(target)]
        assert target is self.params[name],name
        bound=[] if source is None else [source]
        if name in self.pending:bound=sorted(self.pending.pop(name))
        self.event('loader_call',sources=bound,target=name,input=_meta(value),target_before=_meta(target),args=[str(x) for x in args[2:]],kwargs={k:str(v) for k,v in kwargs.items()})
        # Independent CPU expected shard for only the 90 Flash dynamic FFN
        # scale/offset pairs. Native loader and its arguments remain unchanged.
        check = bool(re.fullmatch(r'model\.layers\.\d+\.mlp\.(?:shared_experts\.)?(?:gate_up_proj|down_proj)\.weight_(?:scale|offset)', name))
        if check:
            import torch
            assert value.dtype == target.dtype == torch.float32
            assert value.ndim == target.ndim == 2 and value.shape[1] == target.shape[1] == 1
            assert len(bound) == 1 and not kwargs
            if '.gate_up_proj.' in name:
                assert len(args) == 3 and args[2] in (0, 1) and value.shape[0] % self.world == 0
                per = value.shape[0] // self.world
                assert target.shape[0] == 2 * per
                expected = value.narrow(0, self.rank * per, per).detach().cpu().contiguous().clone()
                offset = args[2] * per
            else:
                assert len(args) == 2 and value.shape == target.shape
                expected = value.detach().cpu().contiguous().clone();offset = 0
            digest = lambda t: hashlib.sha256(t.numpy().tobytes()).hexdigest()
            expected_sha = digest(expected)
        result=original(*args,**kwargs)
        if check:
            assert result is not False
            actual = target.detach().narrow(0, offset, expected.shape[0]).cpu().contiguous()
            assert torch.equal(actual, expected), 'FP32_CHECKPOINT_LINEAR_SCALE_VALUE_MISMATCH: ' + name
            if name not in self.linear_scale_expected:
                self.linear_scale_expected[name] = torch.empty(tuple(target.shape), dtype=torch.float32, device='cpu')
            self.linear_scale_expected[name].narrow(0, offset, expected.shape[0]).copy_(expected)
            key = name + ':' + str(offset);assert not self.linear_scale_parts[key];self.linear_scale_parts[key] += 1
            self.linear_scale_checks += 1
            self.event('linear_scale_value_check', target=name, sources=bound, offset=offset,
                       elements=expected.numel(), source_dtype=str(value.dtype), target_dtype=str(target.dtype),
                       expected_sha256=expected_sha, actual_sha256=digest(actual), exact=True)
        if result is not False:
            self.targets[name]+=1;self.covered.update(bound)
        self.event('loader_return',sources=bound,target=name,target_after=_meta(target),loaded=result is not False)
        if value.dtype==__import__('torch').int8 and target.dtype!=__import__('torch').int8:
            raise RuntimeError('INT8 checkpoint weight lost native INT8 target storage: '+name)
        if value.dtype.is_floating_point and target.dtype==__import__('torch').int8:
            raise RuntimeError('FLOAT checkpoint tensor was loaded into an INT8 parameter: '+name)
        return result

    def event(self,kind,**fields):
        b=(json.dumps({'kind':kind,'rank':self.rank,**fields},sort_keys=True,separators=(',',':'))+'\n').encode();self.bytes+=len(b)
        if self.bytes>128*1024*1024:raise RuntimeError('Weight metadata log exceeds128MiB/rank')
        self.log.write(b);self.count+=1
        if self.count%256==0:self.log.flush();self.raw.flush()

    def stream(self,weights):
        for name,value in weights:
            if name in self.sources:raise RuntimeError('Duplicate checkpoint tensor: '+name)
            if name not in self.expected:raise RuntimeError('Unexpected checkpoint tensor: '+name)
            e=self.expected[name];meta=_meta(value)
            dtype={'I8':'torch.int8','BF16':'torch.bfloat16','F32':'torch.float32'}[e['dtype']]
            assert meta['shape']==e['shape'] and meta['dtype']==dtype,(name,meta,e)
            mapped=self.mapper.apply_list([name]);assert len(mapped)==1;canonical=mapped[0]
            self.sources[name]={'mapped':canonical,**meta};self.event('source',name=name,mapped=canonical,metadata=meta)
            for key in ['.q_a_proj.','.kv_a_proj_with_mqa.']:
                if key in canonical:
                    target=canonical.replace(key,'.fused_qkv_a_proj_with_mqa.')
                    if target in self.params:self.pending.setdefault(target,set()).add(name)
            token=_CURRENT.set(name)
            try:yield name,value
            finally:_CURRENT.reset(token)

    def load_done(self,model):
        # Full wrapper excludes its one MTP layer and the already-exported vision
        # rotation auxiliary. Everything else must reach an actual native loader.
        skipped={}
        for name,e in self.sources.items():
            if name in self.covered:continue
            m=re.match(r'model\.layers\.(\d+)\.',e['mapped'])
            if m and int(m.group(1))>=model.config.num_hidden_layers:skipped[name]='MTP_layer_not_instantiated_in_main_model'
            elif name=='rot.weight':skipped[name]='Standalone_vision_rotation_already_exported_in_merger_down_proj_as_fixed_vLLM_wrapper'
            else:skipped[name]='UNEXPLAINED_NO_NATIVE_LOADER_CALL'
        missing=set(self.params)-set(self.targets)
        defaults=missing&self.zero_bias_defaults
        unexplained_targets=sorted(missing-defaults)
        absent_sources=sorted(set(self.expected)-set(self.sources))
        # Iterator policies may exclude only the same explicit auxiliaries.
        allowed_absent=[];bad_absent=[]
        for name in absent_sources:
            mapped=self.mapper.apply_list([name])[0];m=re.match(r'model\.layers\.(\d+)\.',mapped)
            if name=='rot.weight' or (m and int(m.group(1))>=model.config.num_hidden_layers):allowed_absent.append(name)
            else:bad_absent.append(name)
        out={'project_id':'glm53-flash','rank':self.rank,'world_size':self.world,'source_count':len(self.sources),'covered_source_count':len(self.covered),'target_count':len(self.params),'loaded_target_count':len(self.targets),'explicit_zero_bias_defaults':sorted(defaults),'skipped_sources':skipped,'allowed_iterator_absent':allowed_absent,'unexplained_iterator_absent':bad_absent,'unexplained_targets':unexplained_targets,'unresolved_fused_sources':{k:sorted(v) for k,v in self.pending.items() if any(x not in self.covered for x in v)},'integer_weight_storage_preserved':True,'actual_postprocess_complete':False,'generation_requests':0,'runtime_bindings':self.bindings}
        out['load_coverage_passed']=not (unexplained_targets or bad_absent or out['unresolved_fused_sources'] or any(v.startswith('UNEXPLAINED') for v in skipped.values()))
        assert self.linear_scale_checks == 270 and len(self.linear_scale_expected) == 180
        for name, expected in self.linear_scale_expected.items():
            parts = 2 if '.gate_up_proj.' in name else 1
            assert sum(v for k,v in self.linear_scale_parts.items() if k.startswith(name + ':')) == parts
            assert __import__('torch').equal(self.params[name].detach().cpu(), expected)
        out['exact_FP32_linear_scale_load_checks'] = self.linear_scale_checks
        out['exact_FP32_linear_scale_targets'] = len(self.linear_scale_expected)
        self.summary=out;self.event('load_complete',summary=out);self.save()
        if not out['load_coverage_passed']:raise RuntimeError('Flash parameter coverage failed; see weight-summary-rank'+str(self.rank)+'.json')

    def save(self):
        p=self.root/('weight-summary-rank'+str(self.rank)+'.json');temp=p.with_suffix('.tmp');temp.write_text(json.dumps(self.summary,indent=2)+'\n');temp.replace(p);self.log.flush();self.raw.flush()

    def postprocessed(self,model):
        import torch
        final = {}
        for name, expected in self.linear_scale_expected.items():
            actual = self.params[name].detach().cpu().flatten().contiguous();expected = expected.flatten().contiguous()
            assert actual.dtype == torch.float32 and torch.equal(actual, expected), 'POSTPROCESS_FP32_SCALE_VALUE_MISMATCH: ' + name
            sha = hashlib.sha256(actual.numpy().tobytes()).hexdigest()
            final[name] = {'dtype': str(actual.dtype), 'elements':actual.numel(), 'expected_sha256':hashlib.sha256(expected.numpy().tobytes()).hexdigest(), 'actual_sha256':sha, 'exact':True}
        self.summary['exact_FP32_linear_scale_postprocess'] = final
        self.summary['actual_postprocess_complete']=True
        self.summary['postprocess_parameter_dtype_counts']=dict(collections.Counter(str(t.dtype) for t in model.parameters()))
        self.event('postprocess_complete',summary=self.summary);self.save()
        if os.environ.get('FLASH_PARAMETER_STOP_AFTER_POSTPROCESS')=='1':
            torch.distributed.barrier()
            self.summary['all_ranks_postprocess_barrier_complete']=True;self.event('barrier_complete');self.save();self.log.close();self.raw.close()
            torch.distributed.barrier()  # Every rank persists its gate before any worker exits.
            raise RuntimeError('FLASH_INTENTIONAL_PARAMETER_AUDIT_STOP_AFTER_NATIVE_POSTPROCESS')


def call_native_loader(model,loader,*args,**kwargs):
    audit=_AUDITS.get(id(model))
    return loader(*args,**kwargs) if audit is None else audit.invoke(loader,*args,**kwargs)


def install(model_class):
    if os.environ.get('FLASH_PARAMETER_AUDIT_DIR') is None:return
    if getattr(model_class,'_flash_parameter_audit_installed',False):return
    model_class._flash_parameter_audit_installed=True
    original=model_class.load_weights
    @functools.wraps(original)
    def load_weights(self,weights,*args,**kwargs):
        if kwargs.get('is_nextn') or (args and args[0]):raise RuntimeError('Main-model audit cannot qualify MTP')
        audit=_Audit(self);_AUDITS[id(self)]=audit
        try:
            result=original(self,audit.stream(weights),*args,**kwargs)
            audit.load_done(self);return result
        except BaseException as e:
            audit.event('load_error',error=repr(e));audit.log.flush();audit.raw.flush();raise
    model_class.load_weights=load_weights
    from sglang.srt.model_loader.loader import DefaultModelLoader
    previous=DefaultModelLoader.postprocess_weights
    @functools.wraps(previous)
    def postprocess(model,target_device):
        result=previous(model,target_device)
        audit=_AUDITS.get(id(model))
        if audit is not None:audit.postprocessed(model)
        return result
    DefaultModelLoader.postprocess_weights=staticmethod(postprocess)
