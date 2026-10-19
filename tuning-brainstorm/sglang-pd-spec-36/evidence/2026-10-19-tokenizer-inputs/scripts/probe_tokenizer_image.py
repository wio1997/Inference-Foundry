#!/usr/bin/env python3
"""Actual SGLang tokenizer loader and async batching in a device-free image."""
import argparse,ast,asyncio,copy,datetime,functools,hashlib,importlib,importlib.metadata,json,os,pathlib,subprocess,sys,threading,traceback,types
ap=argparse.ArgumentParser();ap.add_argument('--source-root',required=True);ap.add_argument('--expected-hashes',required=True);ap.add_argument('--model-hashes',required=True);ap.add_argument('--stdout',action='store_true');args=ap.parse_args()
expected=json.loads(args.expected_hashes);model_hashes=json.loads(args.model_hashes);snapshot=pathlib.Path(args.source_root);root=pathlib.Path('/sgl-workspace/sglang');model=pathlib.Path('/model')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
nodes=sorted({str(p) for pat in ['davinci*','devmm_svm','hisi_hdc','ascend*'] for p in pathlib.Path('/dev').glob(pat)});assert not nodes
assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0' and os.environ.get('SGLANG_USE_CPU_ENGINE')=='1'
head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip();assert head=='ce0a8fea5f4245839d174b2b8c55692be1810a7f'
source_hashes={}
for f,h in expected.items():
 source_hashes[f]=sha(root/f);assert source_hashes[f]==h and sha(snapshot/f)==h
assert set(model_hashes)=={'config.json','tokenizer_config.json','tokenizer.json','chat_template.jinja'}
for f,h in model_hashes.items():assert sha(model/f)==h
result={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'probe_status':'PENDING','source_commit':head,'source_hashes':source_hashes,'model_hashes':model_hashes,'script_sha256':sha(pathlib.Path(__file__)),'accelerator_device_nodes':nodes,'environment':{'python':sys.version,'cpu_engine':True,'backend_autoload':False,'network':'none by host controller'},'scope':'Actual image SGLang tokenizer loader/post-load helpers and AsyncDynamicbatchTokenizer in explicit CPU mode. No inference engine, HTTP request, weights or NPU devices. Fixtures are candidate inputs, not a frozen PD serving contract. No performance comparison.'}
def ids_hash(ids):return hashlib.sha256(json.dumps(list(ids),separators=(',',':')).encode()).hexdigest()
stage='import'
try:
 sys.path.insert(0,str(root/'python'))
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 mod=importlib.import_module('sglang.srt.utils.hf_transformers.tokenizer')
 batch_mod=importlib.import_module('sglang.srt.managers.async_dynamic_batch_tokenizer')
 assert pathlib.Path(mod.__file__).resolve()==root/'python/sglang/srt/utils/hf_transformers/tokenizer.py'
 assert 'torch_npu' not in sys.modules
 result['environment'].update({x:importlib.metadata.version(x) for x in ['transformers','tokenizers','jinja2']});result['environment']['torch']=torch.__version__
 stage='load';trace=[];originals={}
 for name in ['_auto_tokenizer_from_pretrained','_resolve_tokenizers_backend','_load_tokenizer_by_declared_class','_apply_post_load_fixes','_fix_v5_tokenizer_components','_fix_v5_add_bos_eos_token','_fix_special_tokens_pattern']:
  original=getattr(mod,name);originals[name]=original
  def wrap(original,name):
   @functools.wraps(original)
   def call(*a,**kw):
    entry={'function':name,'event':'enter'};trace.append(entry)
    value=original(*a,**kw);trace.append({'function':name,'event':'return','type':type(value).__module__+'.'+type(value).__name__});return value
   return call
  setattr(mod,name,wrap(original,name))
 tok=mod.get_tokenizer(str(model),tokenizer_mode='auto',tokenizer_backend='huggingface',trust_remote_code=False,local_files_only=True)
 for name,original in originals.items():setattr(mod,name,original)
 result['loader_trace']=trace;result['tokenizer']={'class':type(tok).__module__+'.'+type(tok).__name__,'is_fast':getattr(tok,'is_fast',None),'vocab_size':len(tok),'eos_token_id':tok.eos_token_id,'pad_token_id':tok.pad_token_id,'additional_stop_token_ids':sorted(getattr(tok,'additional_stop_token_ids',None) or []),'pre_tokenizer':type(tok._tokenizer.pre_tokenizer).__name__,'decoder':type(tok._tokenizer.decoder).__name__,'template_sha256':hashlib.sha256(tok.chat_template.encode()).hexdigest()};assert result['tokenizer']['template_sha256']==model_hashes['chat_template.jinja']
 stage='token-differential'
 from tokenizers import Tokenizer as RawTokenizer
 raw=RawTokenizer.from_file(str(model/'tokenizer.json'));texts=['','hello world','  leading and trailing  ','line1\nline2\tend','\u4e2d\u6587\u5206\u8bcd\u68c0\u67e5','\U0001f600 caf\u00e9 e\u0301','[gMASK]<sop><|user|>x<|assistant|><think>','json: {"x": 1, "y": "\\n"}'];rows=[]
 for text in texts:
  plain=tok.encode(text,add_special_tokens=False);reference=raw.encode(text,add_special_tokens=False).ids;assert plain==reference
  auto=tok.encode(text,add_special_tokens=True);raw_auto=raw.encode(text,add_special_tokens=True).ids
  rows.append({'text':text,'plain_ids':plain,'plain_matches_raw':True,'auto_ids':auto,'auto_matches_raw_post_processor':auto==raw_auto})
 extra=json.loads((model/'tokenizer_config.json').read_text())['extra_special_tokens'];special=[]
 for text in extra:
  got=tok.encode(text,add_special_tokens=False);ref=raw.encode(text,add_special_tokens=False).ids;assert got==ref and len(got)==1;special.append({'text':text,'id':got[0]})
 result['raw_differential']=rows;result['special_tokens']=special
 stage='template'
 serving=root/'python/sglang/srt/entrypoints/openai/serving_chat.py';tree=ast.parse(serving.read_bytes());owner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='OpenAIServingChat');node=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='_probe_prompt_text_round_trip');env={};m=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),copy.deepcopy(node)],type_ignores=[]);exec(compile(ast.fix_missing_locations(m),'<actual-serving-roundtrip-probe>','exec'),env)
 auto_adds=bool(tok.encode(''));receiver=types.SimpleNamespace(tokenizer_manager=types.SimpleNamespace(tokenizer=tok),_tokenizer_auto_adds_specials=auto_adds);lossy=env['_probe_prompt_text_round_trip'](receiver);result['serving_roundtrip_probe']={'result':lossy,'auto_adds_specials':auto_adds,'method_line':node.lineno,'ast_sha256':hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest(),'http_route_executed':False}
 messages=[{'role':'user','content':'Protocol fixture.'}];template_rows=[];defaults={}
 for effort in ['__omitted__','low','high','max',None,'invalid']:
  for thinking in ['__omitted__',False,True]:
   kw={'add_generation_prompt':True,'return_dict':False}
   if effort!='__omitted__':kw['reasoning_effort']=effort
   if thinking!='__omitted__':kw['enable_thinking']=thinking
   rendered=tok.apply_chat_template(messages,tokenize=False,**kw);ids=tok.apply_chat_template(messages,tokenize=True,**kw);via_text=tok.encode(rendered,**({'add_special_tokens':False} if auto_adds else {}));template_rows.append({'reasoning_effort':effort,'enable_thinking':thinking,'tokens':len(ids),'ids_sha256':ids_hash(ids),'rendered_sha256':hashlib.sha256(rendered.encode()).hexdigest(),'roundtrip_equal':list(ids)==list(via_text),'prefix':rendered[:100],'suffix':rendered[-60:]})
   if thinking=='__omitted__':defaults[str(effort)]=ids_hash(ids)
 assert defaults['__omitted__']==defaults['max']==defaults['None']==defaults['invalid'];assert defaults['low']!=defaults['max'] and defaults['high']!=defaults['max'];result['template_rows']=template_rows
 stage='fixtures';fixtures=[]
 def render_case(label,count):
  for unit in [' x',' a',' z',' 1']:
   content='Synthetic protocol fixture '+label+'. Markers follow.'
   def build(n):
    msg=[{'role':'user','content':content+unit*n}];kw={'add_generation_prompt':True,'return_dict':False,'reasoning_effort':'max'};return msg,tok.apply_chat_template(msg,tokenize=False,**kw),list(tok.apply_chat_template(msg,tokenize=True,**kw))
   base_n=100;_,_,sample=build(base_n);n=base_n+count-len(sample)
   if n<0:continue
   msg,text,ids=build(n)
   if len(ids)==count:
    encoded=tok.encode(text,add_special_tokens=False);return {'label':label,'input_tokens':count,'messages':msg,'rendered_text':text,'input_ids':ids,'input_ids_sha256':ids_hash(ids),'roundtrip_plain_equal':encoded==ids,'template_kwargs':{'reasoning_effort':'max','add_generation_prompt':True},'note':'Synthetic candidate fixture; not assigned to task serving resources or output policy.'}
  raise AssertionError('Could not construct exact token count for '+label)
 for label,count in [('c2-short-0',512),('c2-short-1',512),('c3-long-0',32768),('c3-long-1',32768),('c3-long-2',32768)]:fixtures.append(render_case(label,count))
 result['input_fixtures']=fixtures
 stage='async-correctness'
 class SpyTokenizer:
  def __init__(self,hold=False):self.calls=[];self.hold=hold;self.started=threading.Event();self.release=threading.Event()
  def __call__(self,prompts,**kw):
   self.calls.append({'batch_size':len(prompts) if isinstance(prompts,list) else 1,'kwargs':kw})
   if self.hold:self.started.set();assert self.release.wait(10),'Cancellation gate was not released';self.hold=False
   return tok(prompts,**kw)
 async def close(actor):
  if actor._batcher_task:
   actor._batcher_task.cancel()
   try:await actor._batcher_task
   except asyncio.CancelledError:pass
   actor._batcher_task=None
  actor._executor.shutdown(wait=True)
 async def check(prompts,kwargs_list,max_batch_size=32,delays=None):
  spy=SpyTokenizer();actor=batch_mod.AsyncDynamicbatchTokenizer(spy,max_batch_size=max_batch_size,batch_wait_timeout_s=0.002)
  try:
   async def run(i):
    if delays:await asyncio.sleep(delays[i])
    return await actor.encode(prompts[i],**kwargs_list[i])
   got=await asyncio.wait_for(asyncio.gather(*(run(i) for i in range(len(prompts)))),20);want=[dict(tok(p,**kw)) for p,kw in zip(prompts,kwargs_list)];assert [dict(x) for x in got]==want
   return {'requests':len(prompts),'max_batch_size':max_batch_size,'delays_requested':delays,'calls':spy.calls,'ids_sha256':[ids_hash(x['input_ids']) for x in got],'matches_individual':True}
  finally:await close(actor)
 async def checks():
  texts=[f['rendered_text'] for f in fixtures];kw={'add_special_tokens':False};out=[]
  out.append(await check(texts[:2],[kw]*2));out.append(await check(texts[2:],[kw]*3));out.append(await check(texts[:2],[kw,{'add_special_tokens':True}]));out.append(await check(['one','two','three'],[kw]*3,delays=[0,0.01,0.02]));out.append(await check(texts[:2],[kw]*2,max_batch_size=1))
  spy=SpyTokenizer(hold=True);actor=batch_mod.AsyncDynamicbatchTokenizer(spy,batch_wait_timeout_s=0.002);tasks=[]
  try:
   tasks=[asyncio.create_task(actor.encode(x,**kw)) for x in ['cancel-me','survivor']];assert await asyncio.to_thread(spy.started.wait,5);tasks[0].cancel();spy.release.set();got=await asyncio.wait_for(asyncio.gather(*tasks,return_exceptions=True),15);assert isinstance(got[0],asyncio.CancelledError) and dict(got[1])==dict(tok('survivor',**kw));out.append({'case':'cancellation','cancelled':True,'survivor_matches':True,'calls':spy.calls})
  finally:
   spy.release.set()
   for task in tasks:
    if not task.done():task.cancel()
   await close(actor)
  spy=SpyTokenizer();actor=batch_mod.AsyncDynamicbatchTokenizer(spy,batch_wait_timeout_s=0.002)
  try:
   bad=await asyncio.wait_for(asyncio.gather(*(actor.encode(x,padding='__invalid_padding_fixture__') for x in ['bad-one','bad-two']),return_exceptions=True),10);assert all(isinstance(x,ValueError) for x in bad)
   good=await asyncio.wait_for(actor.encode('recovered',**kw),10);assert dict(good)==dict(tok('recovered',**kw));out.append({'case':'actual_tokenizer_error_recovery','errors':[type(x).__name__ for x in bad],'recovery_matches':True,'calls':spy.calls})
  finally:await close(actor)
  return out
 result['async_cases']=asyncio.run(checks());result['probe_status']='PASS';result['correctness']='PASS';assert 'torch_npu' not in sys.modules
 result['loaded_sglang_files']={str(pathlib.Path(m.__file__).resolve().relative_to(root)):sha(pathlib.Path(m.__file__)) for name,m in list(sys.modules.items()) if name.startswith('sglang') and getattr(m,'__file__',None) and pathlib.Path(m.__file__).resolve().is_relative_to(root) and pathlib.Path(m.__file__).suffix=='.py'}
except ImportError as exc:result.update({'probe_status':'UNAVAILABLE','correctness':'NOT_EVALUATED','stage':stage,'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
except Exception as exc:result.update({'probe_status':'ERROR','correctness':'FAIL','stage':stage,'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
for f,h in model_hashes.items():assert sha(model/f)==h
print(json.dumps(result));sys.exit(1 if result['probe_status']=='ERROR' else 0)
