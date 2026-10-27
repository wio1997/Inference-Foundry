"""Replay saved completion conversion and greedy normalization from AST, CPU stdlib only.

No framework import. Capturing constructors stand in for framework classes; results
are source-derived contracts, never historical sampler telemetry or logits.
"""
from pathlib import Path
import ast, copy, hashlib, json, re, types, time

D = Path(__file__).resolve().parent
R = D.parent
S = D / 'installed-source'
E = D / 'saved-inputs'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
facts = []

def cls(path, name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == name)

def method(path, owner, name):
    n = next(n for n in cls(path, owner).body if isinstance(n, ast.FunctionDef) and n.name == name)
    facts.append({'path': str(path.relative_to(D)), 'sha256': sha(path), 'symbol': owner+'.'+name, 'line': n.lineno, 'end_line': n.end_lineno})
    return copy.deepcopy(n)

def execute_method(n, namespace=None):
    n.decorator_list = []
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), n], type_ignores=[])
    env = dict(namespace or {})
    exec(compile(ast.fix_missing_locations(tree), '<saved-source-method>', 'exec'), env)
    return env[n.name]

def fields(path, name):
    out = {}
    for n in cls(path, name).body:
        if not isinstance(n, ast.AnnAssign) or not isinstance(n.target, ast.Name) or n.value is None:
            continue
        value = n.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == 'Field':
            defaults = [k.value for k in value.keywords if k.arg == 'default']
            if defaults: value = defaults[0]
            elif value.args: value = value.args[0]
            elif any(k.arg == 'default_factory' and isinstance(k.value, ast.Name) and k.value.id == 'list' for k in value.keywords):
                out[n.target.id] = []; continue
            else: continue
        try: out[n.target.id] = ast.literal_eval(value)
        except (ValueError, TypeError): pass
    return out

VP = S/'vllm/entrypoints/openai/completion/protocol.py'
SP = S/'sglang/srt/entrypoints/openai/protocol.py'
VSP = S/'vllm/sampling_params.py'
SSP = S/'sglang/srt/sampling/sampling_params.py'

def payloads():
    out = []
    prompt = read(E/'dataset/in512-c1.json')[0]
    for kind, label in [('vllm','full-lineage-fx-model-g81-exec-g3-vllm-in512-c1-o32'),('sglang','full-lineage-replay-fences-g67-sglang-in512-c1-o32')]:
        m = read(E/'waves'/label/'manifest.json'); q = m['requests'][0]
        body = {'model':m['model'],'prompt':prompt,'max_tokens':m.get('max_tokens',q['max_tokens']),'stream':True}
        body.update(m['request_params']);body['seed']=q['seed'];body.update(m['extra_body'])
        b = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(',',':')).encode()
        assert len(b)==q['payload_bytes'] and hashlib.sha256(b).hexdigest()==q['payload_sha256']
        out.append((kind,body,q))
    assert out[0][1]==out[1][1]
    return out

class Capture:
    @staticmethod
    def from_optional(**kw): return kw

def convert_vllm(body, model_defaults):
    kw = fields(VP,'CompletionRequest');kw.update(body)
    obj = types.SimpleNamespace(**kw)
    obj.extract_structured_outputs = lambda: None  # request has no structured-output field
    obj._DEFAULT_SAMPLING_PARAMS = fields(VP,'CompletionRequest')['_DEFAULT_SAMPLING_PARAMS']
    n = method(VP,'CompletionRequest','to_sampling_params')
    f = execute_method(n, {'SamplingParams':Capture,'RequestOutputKind':types.SimpleNamespace(DELTA='DELTA',FINAL_ONLY='FINAL_ONLY')})
    return f(obj,body['max_tokens'],model_defaults)

def convert_sg(body, preferred):
    kw=fields(SP,'CompletionRequest');kw.update(body)
    n=method(S/'sglang/srt/entrypoints/openai/serving_completions.py','OpenAIServingCompletion','_build_sampling_params')
    out=execute_method(n)(None,types.SimpleNamespace(**kw))
    return {**preferred,**out}  # exact precedence in tokenizer_manager.py:1461

def normalize_vllm(params):
    obj=types.SimpleNamespace(**params)
    obj._verify_greedy_sampling=lambda: None  # n=1 checked independently
    assert obj.n==1
    n=method(VSP,'SamplingParams','__post_init__')
    greedy=[z for z in n.body if isinstance(z,ast.If) and ast.unparse(z.test)=='self.temperature < _SAMPLING_EPS']
    assert len(greedy)==1
    # Execute the original normalization branch, without framework validation/allocation.
    fn=ast.FunctionDef(name='normalization',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='self')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=greedy,decorator_list=[])
    eps=next(ast.literal_eval(z.value) for z in ast.parse(VSP.read_text()).body if isinstance(z,ast.Assign) and any(isinstance(t,ast.Name)and t.id=='_SAMPLING_EPS'for t in z.targets))
    execute_method(fn,{'_SAMPLING_EPS':eps})(obj)
    typ=execute_method(method(VSP,'SamplingParams','sampling_type'),{'_SAMPLING_EPS':eps,'SamplingType':types.SimpleNamespace(GREEDY='GREEDY',RANDOM_SEED='RANDOM_SEED',RANDOM='RANDOM')})(obj)
    return vars(obj),typ

def normalize_sg(params):
    kw=fields(SSP,'SamplingParams');kw.update(params)
    obj=types.SimpleNamespace(**kw)
    tree=ast.parse(SSP.read_text())
    constants={}
    for n in tree.body:
        if isinstance(n,ast.Assign):
            for t in n.targets:
                if isinstance(t,ast.Name)and t.id in ['_SAMPLING_EPS','TOP_K_ALL']: constants[t.id]=eval(compile(ast.Expression(n.value), '<source-constant>', 'eval'), {'__builtins__':{}})
    execute_method(method(SSP,'SamplingParams','__post_init__'),constants)(obj)
    return vars(obj),all(x.top_k<=1 for x in [obj])

def replay_argmax_routing():
    """Run original branch control flow with shape/argmax sentinels, no tensor data."""
    token=types.SimpleNamespace(shape=(1,));token.view=lambda *args:token
    calls={'vllm_ascend_argmax':0,'sglang_argmax':0,'sglang_TP_sync_stub':0}
    class Logits:
        shape=(1,248320)
        def argmax(self,**kwargs):
            assert kwargs=={'dim':-1};calls['vllm_ascend_argmax']+=1;return token
    meta=types.SimpleNamespace(all_greedy=True,all_random=False,max_num_logprobs=None,logprob_token_ids=None)
    ascend=S/'vllm_ascend/sample/sampler.py'
    assert not any(isinstance(z,ast.FunctionDef)and z.name=='sample'for z in cls(ascend,'AscendSampler').body)
    greedy=execute_method(method(ascend,'AscendSampler','greedy_sample'))
    sampler=types.SimpleNamespace(logprobs_mode='raw_logprobs',greedy_sample=greedy)
    out=execute_method(method(S/'vllm/v1/sample/sampler.py','Sampler','sample'))(sampler,Logits(),meta)
    assert out==(token,None)
    def argmax(logits,dim):
        assert dim==-1;calls['sglang_argmax']+=1;return token
    def sync(out,info):
        assert out is token;calls['sglang_TP_sync_stub']+=1
    sampler=types.SimpleNamespace(_preprocess_logits=lambda logits,info:logits,_sync_token_ids_across_tp=sync)
    info=types.SimpleNamespace(is_all_greedy=True,sampling_mask_batch_indices=None)
    env={'_is_hip':False,'_use_aiter':False,'_trace_e2e_sampler':lambda *a,**k:None,'torch':types.SimpleNamespace(argmax=argmax)}
    out=execute_method(method(S/'sglang/srt/layers/sampler.py','Sampler','forward'),env)(sampler,types.SimpleNamespace(next_token_logits=Logits()),info,False,[],[],None)
    assert out is token and calls=={'vllm_ascend_argmax':1,'sglang_argmax':1,'sglang_TP_sync_stub':1}
    return {'calls':calls,'scope':'Control flow with sentinels; preprocess and TP sync are stubs, no logits, tensors, numerical equality or dynamic binding proof.'}

def run():
    # Verify freshly downloaded source bytes and all saved replay inputs.
    download=read(S/'download.json')
    rows=download['files']+read(S/'ascend-download.json')['files']
    for row in rows:
        p=S/row['relative_path'];assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes']
        if 'installed_equals_git_HEAD'in row: assert row['installed_equals_git_HEAD']
    for row in read(D/'saved-input-manifest.json')['files']:
        p=D/row['path'];assert sha(p)==row['sha256']
    pairs=payloads();body=pairs[0][1]
    vlog=(E/'vllm-serve.log').read_text().split('\n')
    line=next((i+1,z)for i,z in enumerate(vlog)if 'Default vLLM sampling parameters have been overridden' in z)
    defaults=ast.literal_eval(re.search(r': `(\{.*\})`',line[1]).group(1))
    sglog=(E/'sglang-serve.log').read_text().split('\n')
    sgline=next((i+1,z)for i,z in enumerate(sglog)if 'server_args=' in z)
    sgargs=ast.literal_eval(sgline[1].split('server_args=',1)[1]);assert sgargs['sampling_defaults']=='model'
    config=read(S/'sglang/mounted-model/generation_config.json')
    assert sha(S/'sglang/mounted-model/generation_config.json')==sha(S/'vllm/mounted-model/generation_config.json')
    preferred={k:config[k]for k in ['temperature','top_k','top_p']}
    v0=convert_vllm(body,defaults);s0=convert_sg(body,preferred)
    assert s0==convert_sg(body,{})  # source result invariant to these model preferences
    v,vt=normalize_vllm(v0);s,st=normalize_sg(s0)
    assert v0['top_k']==20 and s0['top_k']==-1
    assert v['temperature']==0 and v['top_k']==0 and v['top_p']==1 and vt=='GREEDY'
    assert s['temperature']==1 and s['top_k']==1 and s['top_p']==1 and st
    assert v['repetition_penalty']==s['repetition_penalty']==1
    assert v['frequency_penalty']==s['frequency_penalty']==0
    assert v['presence_penalty']==s['presence_penalty']==0
    # Capture source ranges for every link without claiming dynamic observation.
    for rel,owner,name in [
        ('vllm/config/model.py','ModelConfig','get_diff_sampling_param'),
        ('vllm/entrypoints/openai/completion/serving.py','OpenAIServingCompletion','__init__'),
        ('vllm/v1/worker/gpu_input_batch.py','InputBatch','add_request'),
        ('vllm/v1/worker/gpu_input_batch.py','InputBatch','all_greedy'),
        ('vllm/v1/sample/sampler.py','Sampler','sample'),
        ('vllm/v1/sample/sampler.py','Sampler','greedy_sample'),
        ('sglang/srt/configs/model_config.py','ModelConfig','get_default_sampling_params'),
        ('sglang/srt/managers/tokenizer_manager.py','TokenizerManager','_create_tokenized_object'),
        ('sglang/srt/sampling/sampling_batch_info.py','SamplingBatchInfo','from_schedule_batch'),
        ('sglang/srt/layers/sampler.py','Sampler','forward')]: method(S/rel,owner,name)
    clean=lambda x:{k:v for k,v in x.items()if not callable(v)}
    routing=replay_argmax_routing()
    result={'task_id':'Q27-FULL-MAP-20261025','generation':38,'verified_ns':time.time_ns(),'status':'saved_request_and_current_frozen_source_contract_replayed_not_runtime_sampler_observation',
        'collection_Host_Architecture':'aarch64','CPU_replay_Host_Architecture':__import__('platform').machine(),'NPU':'910B1','source_receipt':download['source'],
        'same_canonical_payload_SHA256':pairs[0][2]['payload_sha256'],'payload_bytes':pairs[0][2]['payload_bytes'],
        'runtime_defaults_log':{'vllm':{'line':line[0],'text':line[1],'default_sampling_params':defaults},'sglang':{'line':sgline[0],'sampling_defaults':sgargs['sampling_defaults'],'preferred_sampling_params':sgargs['preferred_sampling_params']}},
        'vllm':{'converted':v0,'normalized':clean(v),'sampling_type_source_replay':vt},
        'sglang':{'converted_after_preferred_merge':s0,'normalized':clean(s),'is_all_greedy_source_replay':st},
        'facts':facts,'argmax_control_flow_replay':routing,'download_source_files_SHA_verified':len(rows),'Python_files_equal_git_HEAD':sum('installed_equals_git_HEAD'in row for row in rows),
        'conclusion':'Both saved requests select argmax under inspected source semantics. vLLM model top_k20 is reset to0 at temperature0; SG completion defaults top_k-1 override model preference then normalize to1. Model do_sample=true and default top_p0.95 do not override explicit temperature0/top_p1 in this source route. This does not explain unequal decoded output.',
        'replay_scope':'Original API conversion methods; exact vLLM zero-temperature normalization branch and sampling_type property; full SG SamplingParams.__post_init__. Capturing constructors and schema defaults replace imports/Pydantic/msgspec. No claim of full framework execution or below-epsilon positive-temperature normalization.',
        'preferred_model_values_or_empty_same_SG_completion_result':True,
        'runtime_sampler_arguments_observed':False,'runtime_sampler_class_binding_observed':False,'historical_loaded_generation_config_SHA_available':False,'vllm_sampled_IDs_available':False,'equal_logits_proven':False,'first_different_sampled_token_known':False,'greedy_equivalence_passed':False,'optimization_adopted':False,'overall_complete':False}
    (D/'sampling-contract-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result

if __name__=='__main__':
    x=run();print(json.dumps({'status':x['status'],'vllm_sampling_type':x['vllm']['sampling_type_source_replay'],'SG_all_greedy':x['sglang']['is_all_greedy_source_replay'],'source_facts':len(x['facts']),'overall_complete':False}))
