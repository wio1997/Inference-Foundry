#!/usr/bin/env python3
"""Whole exact accumulator -> explicit in-process carrier -> exact detok CPU bodies."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
import ast,array,dataclasses,datetime,hashlib,json,logging,pathlib,sys,traceback,types,typing
from collections import OrderedDict,defaultdict
from types import SimpleNamespace as NS
from transformers import AutoTokenizer
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L030-R005-accumulator-cpu';O.mkdir(parents=True,exist_ok=False);checks=[];cases=[];extracted=[];current='init';boundary_calls=[]
def ck(v,label):
 if not v:raise AssertionError(current+': '+label)
 checks.append({'case':current,'check':label})
def digest(x):return hashlib.sha256(x.encode() if isinstance(x,str) else bytes(x)).hexdigest()
mod=types.ModuleType('m033_frozen_cpu');sys.modules[mod.__name__]=mod
ns=vars(mod);ns.update({'dataclasses':dataclasses,'dataclass':dataclasses.dataclass,'field':dataclasses.field,'OrderedDict':OrderedDict,'defaultdict':defaultdict,'logger':logging.getLogger('M033'),'DETOKENIZER_MAX_STATES':65536,'INIT_INCREMENTAL_DETOKENIZATION_OFFSET':5,**vars(typing)})
# Reuse only the controller's AST extraction function; do not re-execute R004.
old=ast.parse((R/'scripts/probe_L030_detokenize.py').read_text());helper=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='load');exec(compile(ast.Module(body=[helper],type_ignores=[]),'frozen-controller-loader','exec'),globals())
def weight_boundary(events,**kw):boundary_calls.append({'boundary':'weight_version_spans','events_empty':not events});return None
def pickle_boundary(obj):boundary_calls.append({'boundary':'IPC pickle not executed'});return obj
try:
 load('python/sglang/utils.py',functions=['_is_chinese_char','find_printable_text']);load('python/sglang/srt/utils/patch_tokenizer.py',functions=['decode_without_hf_kwargs'])
 load('python/sglang/srt/managers/detokenizer_manager.py',classes=['DecodeStatus','LimitedCapacityDict'],methods={'DetokenizerManager':['trim_matched_stop','_clamp_decode_ids','_grouped_batch_decode','_decode_batch_token_id_output']})
 load('python/sglang/srt/managers/schedule_batch.py',methods={'Req':['finished','output_ids_through_stop','init_incremental_detokenize','_stop_match_tail_len','tail_str','check_match_stop_str_prefix']})
 ns.update({'BatchTokenIDOutput':NS,'compute_weight_version_spans':weight_boundary,'wrap_as_pickle':pickle_boundary})
 load('python/sglang/srt/managers/scheduler_components/output_streamer.py',classes=['_GenerationStreamAccumulator'])
 tok=AutoTokenizer.from_pretrained(str(R/'fixtures/L030-exact-tokenizer'),local_files_only=True,trust_remote_code=False);spec=NS(is_none=lambda:False)
 def encode(x):return tok.encode(x,add_special_tokens=False)
 class Finish:
  def __init__(self,x):self.x=x
  def to_json(self):return self.x
 def req(rid,ids,interval=0,stop=None,finish_at=None,reason=None):
  q=ns['Req']();q.rid=rid;q.output_ids=array.array('q');q.origin_input_ids=q.origin_input_ids_unpadded=array.array('q',encode('历史 input context hello.\n'));q.surr_offset=q.read_offset=None;q.surr_and_decode_ids=None;q.cur_decode_ids_len=0;q.send_decode_id_offset=q.send_token_offset=q.send_output_token_logprobs_offset=0;q.finished_len=None;q.finished_reason=None;q.finished_output=False;q.beam_group=None;q.stream=True;q.http_worker_ipc=None;q.decoded_text='';q.reasoning_tokens=q.cached_tokens=0;q.mm_image_tokens=q.mm_audio_tokens=q.mm_video_tokens=0;q.multimodal_inputs=None;q.retraction_count=0;q.weight_version_events=[];q.time_stats=None;q.customized_info=None;q.spec_verify_ct=q.spec_num_correct_drafts=q.spec_num_block_accept_tokens=q.spec_num_cap_tokens=0;q.spec_correct_drafts_histogram={};q.spec_cap_lens_histogram={};q.tokenizer=tok
  q.sampling_params=NS(stream_interval=interval,skip_special_tokens=True,spaces_between_special_tokens=False,no_stop_trim=False,stop_strs=[stop] if stop else [],stop_regex_strs=[],stop_str_max_len=len(stop) if stop else 0,stop_regex_max_len=0)
  return {'q':q,'ids':ids,'pos':0,'text':'','outids':[],'sent':[],'finish_at':finish_at or len(ids),'reason':reason or {'type':'length','length':len(ids)},'done':False}
 def accum(default):return ns['_GenerationStreamAccumulator'](return_logprob=False,return_hidden_states=False,return_routed_experts=False,return_indexer_topk=False,spec_algorithm=spec,disaggregation_mode='DECODE_FIXTURE',default_stream_interval=default,default_force_stream_interval=50,get_cached_tokens_details=lambda q:None,current_weight_version=None,rust_server_mode=False)
 def run(items,block,interval):
  m=ns['DetokenizerManager']();m.tokenizer=tok;m.vocab_size=len(tok);m.disable_tokenizer_batch_decode=False;m.is_tool_call_parser_gpt_oss=False;m.decode_status=ns['LimitedCapacityDict'](65536);turn=0;finish_rows=0;metadata=[]
  while any(not z['done'] for z in items):
   a=accum(interval);ordered=items[::-1] if turn%2 else items
   for z in ordered:
    if z['done']:continue
    q=z['q'];end=min(z['pos']+block,len(z['ids']));new=end-z['pos'];q.output_ids.extend(z['ids'][z['pos']:end]);z['pos']=end;q.spec_verify_ct+=1;q.spec_num_block_accept_tokens+=new;q.spec_num_correct_drafts+=max(new-1,0);q.spec_num_cap_tokens+=4
    if end>=z['finish_at']:q.finished_reason=Finish(z['reason']);q.finished_len=z['finish_at'];z['done']=True
    a.accept(req=q)
   payload=a.to_payload(dp_rank=0,is_idle_batch=False)
   if payload is not None:
    out=m._decode_batch_token_id_output(payload);ck(len(out)==len(payload.rids),'one text row per emitted request')
    for i,rid in enumerate(payload.rids):
     z=next(z for z in items if z['q'].rid==rid);z['text']+=out[i];z['outids'].extend(payload.output_ids[i]);z['sent'].append({'n':z['pos'],'finished':payload.finished_reasons[i] is not None,'ids':len(payload.output_ids[i]),'text_chars':len(out[i])})
     ck(payload.completion_tokens[i]==len(z['q'].output_ids_through_stop),'completion count is cumulative through stop');ck(payload.spec_verify_ct[i]==z['q'].spec_verify_ct,'spec metadata survives accept/payload');ck(payload.spec_num_correct_drafts[i]==z['q'].spec_num_correct_drafts,'draft-count metadata survives');ck(payload.dp_ranks[i]==0,'dp rank row alignment')
     if payload.finished_reasons[i] is not None:finish_rows+=1;ck(rid not in m.decode_status,'finish state removed after exact accept/payload')
    metadata.append({'rids':payload.rids,'decode_ID_lengths':[len(x) for x in payload.decode_ids],'text_char_lengths':[len(x) for x in out]})
   turn+=1;ck(turn<=300,'finite accepted-block fixture')
  for z in items:
   full=z['ids'][:z['finish_at']];expected=tok.decode(full,skip_special_tokens=True);matched=z['reason'].get('matched')
   if isinstance(matched,str):expected=expected.split(matched)[0]
   ck(z['text']==expected,'full accumulator incremental output equals expected final text');ck(z['outids']==full,'no duplicate/lost output IDs across skipped sends');ck(sum(e['finished'] for e in z['sent'])==1,'one final row per request')
  ck(not m.decode_status,'all component states reclaimed');ck(finish_rows==len(items),'all requests final-emitted');return {'rounds':turn,'request_events':[{'rid':z['q'].rid,'nonfinal_events':sum(not e['finished'] for e in z['sent']),'first_send_after_tokens':z['sent'][0]['n'],'final_send_tokens':z['sent'][-1]['n'],'output_ID_sha256':digest(json.dumps(z['outids']))} for z in items],'payloads':len(metadata),'payload_trace_sha256':digest(json.dumps(metadata))}
 for bs,budget in [(2,128),(3,256)]:
  for block in [1,2,4]:
   for interval in [1,2,3,4]:
    current=f'whole-accumulator-c{bs}-out{budget}-block{block}-interval{interval}';items=[req(str(i),encode(('中文 hello 𠮷 🧑🏽‍🚀\n' if i%2==0 else 'English mixed 文本 😀\n')*80)[:budget]) for i in range(bs)];stats=run(items,block,interval)
    if interval==1:ck(all(e['nonfinal_events']==(budget-1)//block for e in stats['request_events']),'interval1 already groups each accepted block once')
    if block%interval==0 and interval>1:ck(all(e['nonfinal_events']==0 and e['first_send_after_tokens']==budget for e in stats['request_events']),'even congruence witness: only final event')
    cases.append({'id':current,'concurrency_fixture':bs,'output_budget':budget,'accepted_block':block,'stream_interval':interval,**stats})
 current='request-override-default'
 stats=run([req('override',encode('hello 中文 𠮷\n'*50)[:128],interval=1)],4,4);ck(stats['request_events'][0]['nonfinal_events']==31,'request interval1 overrides default4');cases.append({'id':current,**stats})
 current='full-stop-prefix-and-finish-bypass'
 ids=encode('before 中文<STOP>never visible tail');at=next(i for i in range(1,len(ids)+1) if '<STOP>' in tok.decode(ids[:i]));stats=run([req('stop',ids,interval=1,stop='<STOP>',finish_at=at,reason={'type':'stop','matched':'<STOP>'})],1,1);cases.append({'id':current,'fixture_stop_index':at,**stats})
 current='received-abort-finish-bypass'
 stats=run([req('abort',encode('first abort 你好'),interval=4,reason={'type':'abort','message':'synthetic received abort'})],4,4);cases.append({'id':current,**stats})
 current='empty-payload'
 a=accum(1);ck(a.to_payload(dp_rank=0,is_idle_batch=False) is None,'empty normal accumulator produces no carrier');p=a.to_payload(dp_rank=0,is_idle_batch=True);ck(p.rids==[],'idle carrier is empty');cases.append({'id':current})
 result={'status':'PASS','semantic_checks':len(checks),'case_count':len(cases),'cases':cases,'checks':checks,'extracted':extracted,'full_accumulator_class_executed':True,'full_modules_imported':False,'native_executed':False,'tokenizer_class':type(tok).__module__+'.'+type(tok).__name__,'explicit_boundaries':['Req fixture attributes/counters and externally chosen finish position; not scheduler/model sampling','BatchTokenIDOutput replaced with SimpleNamespace in-process carrier; msgspec/ZMQ/IPC serialization not executed','wrap_as_pickle identity boundary; weight_version_spans returns None for empty fixture events, not verified','spec_algorithm.is_none=False fixture; DECODE mode marker, no physical P/D','beam/logprob/hidden/routed/MM/custom-info optional branches excluded','TokenizerManager/HTTP/SSE/usage formatting/client disconnect/allocator reclaim not executed'],'boundary_calls':dict((k,sum(z['boundary']==k for z in boundary_calls)) for k in {z['boundary'] for z in boundary_calls}),'performance':'NOT_MEASURED','finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(O/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':'PASS','semantic_checks':len(checks),'cases':len(cases)}))
except Exception as e:
 (O/'result.json').write_text(json.dumps({'status':'FAIL','failed_case':current,'error_type':type(e).__name__,'error':str(e),'completed_checks':len(checks),'completed_cases':cases,'native_executed':False},ensure_ascii=False,indent=2)+'\n');traceback.print_exc();sys.exit(1)
