#!/usr/bin/env python3
"""Exact selected SGLang bodies + exact historical HF tokenizer, CPU component only."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
import ast,array,dataclasses,datetime,hashlib,json,logging,pathlib,sys,traceback,typing
from collections import OrderedDict,defaultdict
from types import SimpleNamespace as NS
from transformers import AutoTokenizer
import transformers,tokenizers
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L030-R004-detokenizer-cpu';O.mkdir(parents=True,exist_ok=False)
checks=[];cases=[];extracted=[];current='initialization'
def ck(v,label):
 if not v:raise AssertionError(current+': '+label)
 checks.append({'case':current,'check':label})
def digest(x):return hashlib.sha256(x.encode() if isinstance(x,str) else bytes(x)).hexdigest()
ns={'dataclasses':dataclasses,'OrderedDict':OrderedDict,'defaultdict':defaultdict,'logger':logging.getLogger('M033'),'DETOKENIZER_MAX_STATES':65536,'INIT_INCREMENTAL_DETOKENIZATION_OFFSET':5,**vars(typing)}
def load(path,classes=(),functions=(),methods=None):
 p=R/'sources/installed'/path;t=ast.parse(p.read_text());nodes=[]
 for n in t.body:
  if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in functions:nodes.append(n)
  if isinstance(n,ast.ClassDef) and n.name in classes:nodes.append(n)
  if isinstance(n,ast.ClassDef) and methods and n.name in methods:
   wanted=methods[n.name];body=[m for m in n.body if isinstance(m,(ast.FunctionDef,ast.AsyncFunctionDef)) and m.name in wanted];assert {m.name for m in body}==set(wanted)
   nodes.append(ast.ClassDef(name=n.name,bases=[],keywords=[],body=body,decorator_list=[]))
 for n in nodes:
  extracted.append({'path':path,'name':n.name,'lines':[getattr(n,'lineno',None),getattr(n,'end_lineno',None)],'body_ast_sha256':digest(ast.dump(n,include_attributes=False)),'extraction':'complete class' if n.name in classes else ('complete selected method bodies; module imports/constructors bypassed' if isinstance(n,ast.ClassDef) else 'complete function')})
 mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[]);ast.fix_missing_locations(mod);exec(compile(mod,str(p),'exec'),ns)
try:
 load('python/sglang/utils.py',functions=['_is_chinese_char','find_printable_text'])
 load('python/sglang/srt/utils/patch_tokenizer.py',functions=['decode_without_hf_kwargs'])
 load('python/sglang/srt/managers/detokenizer_manager.py',classes=['DecodeStatus','LimitedCapacityDict'],methods={'DetokenizerManager':['trim_matched_stop','_clamp_decode_ids','_grouped_batch_decode','_decode_batch_token_id_output']})
 load('python/sglang/srt/managers/schedule_batch.py',methods={'Req':['output_ids_through_stop','init_incremental_detokenize','_stop_match_tail_len','tail_str','check_match_stop_str_prefix']})
 tok=AutoTokenizer.from_pretrained(str(R/'fixtures/L030-exact-tokenizer'),local_files_only=True,trust_remote_code=False)
 ck(tok.is_fast and type(tok).__name__=='TokenizersBackend','same historical tokenizer class/fast path')
 hashes=json.loads((R/'specs/L010-cpu-model-hashes.json').read_text())
 for n in ['tokenizer.json','tokenizer_config.json']:ck(digest((R/'fixtures/L030-exact-tokenizer'/n).read_bytes())==hashes[n],n+' historical SHA256')
 class Counter:
  def __init__(self,fast=True):self.is_fast=fast;self.calls=[]
  def __getattr__(self,n):return getattr(tok,n)
  def decode(self,ids,**kw):self.calls.append(('decode',len(ids)));return tok.decode(ids,**kw)
  def batch_decode(self,ids,**kw):self.calls.append(('batch',len(ids)));return tok.batch_decode(ids,**kw)
 def manager(disable=False,capacity=65536,fast=True):
  m=ns['DetokenizerManager']();m.tokenizer=Counter(fast);m.vocab_size=len(tok);m.disable_tokenizer_batch_decode=disable;m.is_tool_call_parser_gpt_oss=False;m.decode_status=ns['LimitedCapacityDict'](capacity);return m
 def encode(text):return tok.encode(text,add_special_tokens=False)
 def request(rid,ids,stop=None,no_trim=False,skip=True,space=False,finish_at=None,reason=None,prompt=None):
  q=ns['Req']();q.rid=rid;q.origin_input_ids_unpadded=array.array('q',prompt if prompt is not None else encode('历史上下文 hello world.\n'));q.output_ids=array.array('q');q.finished_len=None;q.surr_offset=q.read_offset=None;q.surr_and_decode_ids=None;q.cur_decode_ids_len=0;q.send_decode_id_offset=0;q.decoded_text='';q.tokenizer=tok
  q.sampling_params=NS(stop_strs=[stop] if stop else [],stop_regex_strs=[],stop_str_max_len=len(stop) if stop else 0,stop_regex_max_len=0)
  return {'q':q,'all':ids,'pos':0,'finish_at':finish_at or len(ids),'reason':reason or {'type':'length','length':len(ids)},'skip':skip,'space':space,'no_trim':no_trim,'text':'','chunks':[],'stop':stop,'done':False,'suppressed':0}
 def stream(items,pattern,disable=False,guard=True,capacity=65536,m=None):
  m=m or manager(disable,capacity);rounds=0;partial=0;pending=0;sent=0;events=[]
  while any(not z['done'] for z in items):
   batch=[]
   for z in items:
    if z['done']:continue
    step=pattern[rounds%len(pattern)];end=min(z['pos']+step,len(z['all']));z['q'].output_ids.extend(z['all'][z['pos']:end]);z['pos']=end;fin=end>=z['finish_at'];q=z['q']
    if fin:q.finished_len=z['finish_at'];z['done']=True
    if not fin and guard and q.check_match_stop_str_prefix():z['suppressed']+=1;continue
    ids,off=q.init_incremental_detokenize();delta=list(ids[q.send_decode_id_offset:]);q.send_decode_id_offset=len(ids)
    batch.append((z,delta,off,z['reason'] if fin else None))
   if rounds%2:batch.reverse()
   recv=NS(rids=[z['q'].rid for z,_,_,_ in batch],decoded_texts=['']*len(batch),decode_ids=[ids for _,ids,_,_ in batch],read_offsets=[off for _,_,off,_ in batch],finished_reasons=[f for _,_,_,f in batch],skip_special_tokens=[z['skip'] for z,_,_,_ in batch],spaces_between_special_tokens=[z['space'] for z,_,_,_ in batch],no_stop_trim=[z['no_trim'] for z,_,_,_ in batch])
   outs=m._decode_batch_token_id_output(recv);ck(len(outs)==len(batch),'batch scatter size')
   for (z,_,_,fin),text in zip(batch,outs):
    z['text']+=text;z['chunks'].append(text);sent+=1
    if fin:ck(z['q'].rid not in m.decode_status,'finish removes RID state')
    else:
     s=m.decode_status[z['q'].rid];ck(s.sent_offset>=s.decoded_text_len,'sent offset does not trail committed length');pending+=s.sent_offset>s.decoded_text_len
     full=tok.decode(list(z['q'].output_ids),skip_special_tokens=z['skip'],spaces_between_special_tokens=z['space']);partial+=full.endswith('�')
     ck(full.startswith(z['text']),'visible text is a prefix of full current decode')
    events.append((z['q'].rid,len(text),bool(fin)))
   rounds+=1;ck(rounds<=600,'finite component schedule')
  ck(not m.decode_status,'all finished states removed')
  return {'rounds':rounds,'sent_request_chunks':sent,'partial_utf8_prefix_observations':partial,'pending_uncommitted_observations':pending,'suppressed_stop_prefix_steps':sum(z['suppressed'] for z in items),'event_digest':digest(json.dumps(events))},m
 def expected(z):
  ids=z['all'][:z['finish_at']];matched=z['reason'].get('matched')
  if isinstance(matched,int) and not z['no_trim']:ids=ids[:-1]
  text=tok.decode(ids,skip_special_tokens=z['skip'],spaces_between_special_tokens=z['space'])
  if isinstance(matched,str) and matched in text:text=text[:text.index(matched)+(len(matched) if z['no_trim'] else 0)]
  return text
 base=['中文 hello world\n𠮷 🧑🏽‍🚀 🦄 你好 e\u0301\tEND. ','English punctuation: a, b; c!\n汉字😀 mixed text. ','第一行\nSecond line — café 𐍈\n']
 for bs,budget in [(2,128),(3,256)]:
  for pattern in [(1,),(2,),(4,),(1,4,2,3)]:
   current=f'interleaved-c{bs}-output{budget}-blocks'+','.join(map(str,pattern));pairs=[];info=[]
   for disable in [False,True]:
    items=[request(str(i),encode(base[i]*40)[:budget],skip=i%2==0,space=i%2==1) for i in range(bs)]
    stats,m=stream(items,pattern,disable)
    for z in items:ck(z['text']==expected(z),'incremental equals full decode per RID')
    pairs.append([z['chunks'] for z in items]);info.append(stats)
   ck(pairs[0]==pairs[1],'grouped/scalar identical per-request deltas')
   cases.append({'id':current,'scope':'COMPONENT synthetic accepted token blocks; not serving request shape','batch_size':bs,'output_tokens':budget,'pattern':pattern,'grouped':info[0],'scalar':info[1]})
 current='stop-string-prefix-finish'
 for pattern in [(1,),(4,)]:
  for trim in [False,True]:
   ids=encode('before 你好<STOP>after 𠮷');at=next(i for i in range(1,len(ids)+1) if '<STOP>' in tok.decode(ids[:i]));z=request('stop',ids,stop='<STOP>',no_trim=trim,finish_at=at,reason={'type':'stop','matched':'<STOP>'});stats,_=stream([z],pattern);ck(z['text']==expected(z),'stop string final trim/no trim');cases.append({'id':current,'pattern':pattern,'no_stop_trim':trim,**stats})
 current='stop-token-trailing-accepted-IDs'
 for skip in [False,True]:
  for trim in [False,True]:
   ids=encode('token stop 中文')+[tok.eos_token_id]+encode('ignored tail');at=len(encode('token stop 中文'))+1;z=request('eos',ids,skip=skip,no_trim=trim,finish_at=at,reason={'type':'stop','matched':tok.eos_token_id});stats,_=stream([z],(4,));ck(z['text']==expected(z),'stop token keeps/excludes marker and excludes IDs after finished_len');cases.append({'id':current,'skip':skip,'no_stop_trim':trim,**stats})
 current='abort-shaped-finish-and-RID-reuse'
 m=manager()
 for text,reason in [('aborted 中文𠮷',{'type':'abort','message':'synthetic received finish'}),('reuse hello 新请求',{'type':'length','length':9})]:
  z=request('same-rid',encode(text),reason=reason);stats,m=stream([z],(2,),m=m);ck(z['text']==expected(z),'received finish and same RID reuse reconstruct independently');cases.append({'id':current,'finish_type':reason['type'],**stats})
 current='grouping-empty-and-option-scatter'
 rows=[[],encode('hello'),[tok.eos_token_id],[],encode('中文'),[tok.eos_token_id],encode('end'),[]];skip=[False,False,True,True,True,False,False,False];space=[False,False,False,True,True,True,False,False]
 for fast in [True,False]:
  m=manager(fast=fast);out=m._grouped_batch_decode(rows,skip,space);gold=[tok.decode(ids,skip_special_tokens=sk,spaces_between_special_tokens=sp) if fast else ns['decode_without_hf_kwargs'](tok,ids,sk) for ids,sk,sp in zip(rows,skip,space)];ck(out==gold,'option grouping preserves row values');nonempty=sum(bool(ids) for ids in rows);ngroups=len({(sk,sp) for ids,sk,sp in zip(rows,skip,space) if ids});ck(len(m.tokenizer.calls)==(ngroups if fast else nonempty),'existing grouping/slow-protocol call count');m.tokenizer.calls.clear();ck(m._grouped_batch_decode([[],[]],[True,False],[False,True])==['',''] and not m.tokenizer.calls,'all empties bypass backend');cases.append({'id':current,'backend':'actual HF fast' if fast else 'slow protocol facade around same HF backend','nonempty_rows':nonempty,'backend_calls':ngroups if fast else nonempty})
 current='placeholder-context-and-empty-batch'
 m=manager();z=request('placeholder',encode('real output 中文'),prompt=[-101,-102,len(tok)+1,3,4,5]);stats,m=stream([z],(1,4));ck(z['text']==expected(z),'context-only placeholders clamped symmetrically');ck(m._clamp_decode_ids([-101,0,len(tok)-1,len(tok)],len(tok))==[0,0,len(tok)-1,0],'explicit clamp boundaries');recv=NS(rids=[],decoded_texts=[],decode_ids=[],read_offsets=[],finished_reasons=[],skip_special_tokens=[],spaces_between_special_tokens=[],no_stop_trim=[]);ck(m._decode_batch_token_id_output(recv)==[],'empty batch');cases.append({'id':current,**stats})
 current='capacity-eviction-controlled-counterexample'
 m=manager(capacity=2);r=NS(rids=['a','b','c'],decoded_texts=['']*3,decode_ids=[encode('hello')]*3,read_offsets=[0]*3,finished_reasons=[None]*3,skip_special_tokens=[True]*3,spaces_between_special_tokens=[False]*3,no_stop_trim=[False]*3)
 try:m._decode_batch_token_id_output(r);raise AssertionError('capacity error missing')
 except RuntimeError as e:ck('Decode status not found for request a' in str(e) and 'SGLANG_DETOKENIZER_MAX_STATES' in str(e),'existing actionable capacity error');cases.append({'id':current,'expected_exception':type(e).__name__,'capacity':2,'simultaneous_RIDs':3,'current_runtime_incident':False})
 current='chunk-store-materialize'
 s=ns['DecodeStatus']('',[],0,0)
 for t in ['a','中文','','𠮷']*32:s.append_decoded_text(t)
 ck(s.decoded_text_len==len('a中文𠮷'*32),'chunk length accounting');ck(s.get_decoded_text()=='a中文𠮷'*32 and not s.decoded_text_chunks,'materialize once clears chunk list');ck(s.get_decoded_text()=='a中文𠮷'*32,'repeated materialize no duplication');cases.append({'id':current,'appends':128})
 current='stream-interval-source-predicate'
 p=R/'sources/installed/python/sglang/srt/managers/scheduler_components/output_streamer.py';t=ast.parse(p.read_text());cl=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='_GenerationStreamAccumulator');fn=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name=='accept');expr=next(n.value for n in ast.walk(fn) if isinstance(n,ast.Assign) and any(isinstance(z,ast.Name) and z.id=='should_output' for z in n.targets) and isinstance(n.value,ast.IfExp));code=compile(ast.Expression(expr),str(p),'eval');policy=[]
 for budget in [128,256]:
  for block in [1,2,4]:
   for interval in [1,2,3,4]:
    hits=[n for n in range(block,budget,block) if eval(code,{'req':NS(output_ids=range(n)),'stream_interval':interval})];policy.append({'output_budget':budget,'accepted_block':block,'stream_interval':interval,'nonfinal_send_hits':len(hits),'first_nonfinal_hit_tokens':hits[0] if hits else None})
    if interval==1:ck(len(hits)==(budget-1)//block,'default interval1 emits each nonfinal accepted block')
    if block in [2,4] and interval in [2,4] and block%interval==0:ck(not hits,'modulo congruence can skip all nonfinal send points')
 cases.append({'id':current,'scope':'exact AST expression only; finish/stop guard traced in source, full accumulator NOT executed','expression':ast.unparse(expr),'matrix':policy})
 ck(sum(z.get('grouped',{}).get('partial_utf8_prefix_observations',0) for z in cases)>0,'real tokenizer partial UTF8 prefixes exercised')
 result={'status':'PASS','semantic_checks':len(checks),'case_count':len(cases),'environment':{'python':sys.version,'transformers':transformers.__version__,'tokenizers':tokenizers.__version__,'torch_used':False},'tokenizer':{'class':type(tok).__module__+'.'+type(tok).__name__,'is_fast':tok.is_fast,'vocab_size':len(tok),'sha256':{n:hashes[n] for n in ['tokenizer.json','tokenizer_config.json']},'loader':'AutoTokenizer local_files_only=True; exact HF backend; full SGLang get_tokenizer NOT imported'},'extracted':extracted,'cases':cases,'checks':checks,'native_executed':False,'full_modules_imported':False,'full_accumulator_executed':False,'stop_position_selection':'fixture boundary, not full scheduler stop matcher','SSE_HTTP_slow_client_cancel_allocator_tested':False,'performance':'NOT_MEASURED','finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(O/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':'PASS','semantic_checks':len(checks),'cases':len(cases)}))
except Exception as e:
 (O/'result.json').write_text(json.dumps({'status':'FAIL','failed_case':current,'error_type':type(e).__name__,'error':str(e),'completed_checks':len(checks),'completed_cases':cases,'native_executed':False,'performance':'NOT_MEASURED'},ensure_ascii=False,indent=2)+'\n');traceback.print_exc();sys.exit(1)
