"""Load frozen frontend bodies with explicit input/time/engine boundaries, no serving module import."""
import ast,asyncio,array,dataclasses,hashlib,importlib.util,json,logging,pathlib,pickle,struct,sys,time,types,typing,uuid
from http import HTTPStatus
from types import SimpleNamespace as NS
import fastapi,msgspec,numpy as np,zmq
from fastapi import BackgroundTasks
from fastapi.responses import ORJSONResponse,StreamingResponse
from pydantic import BaseModel
R=pathlib.Path(__file__).resolve().parents[1];S=R/'sources/installed';records=[]
mod=types.ModuleType('lf31_component_namespace');sys.modules[mod.__name__]=mod;ns=vars(mod)
ns.update({k:v for k,v in vars(typing).items() if not k.startswith('_')});ns.update({'asyncio':asyncio,'array':array.array,'dataclasses':dataclasses,'dataclass':dataclasses.dataclass,'field':dataclasses.field,'msgspec':msgspec,'np':np,'pickle':pickle,'struct':struct,'fastapi':fastapi,'BackgroundTasks':BackgroundTasks,'ORJSONResponse':ORJSONResponse,'StreamingResponse':StreamingResponse,'json':json,'time':time,'uuid':uuid,'HTTPStatus':HTTPStatus,'zmq':zmq,'BaseModel':BaseModel,'logger':logging.getLogger('L031'),'OrderedDict':__import__('collections').OrderedDict,'defaultdict':__import__('collections').defaultdict,'HEALTH_CHECK_RID_PREFIX':'HEALTH_CHECK_FIXTURE_','_REQUEST_STATE_WAIT_TIMEOUT':1.0})
def load(path,classes=(),functions=(),methods=None,aliases=(),all_functions=False,constant_prefix=None):
 p=S/path;t=ast.parse(p.read_text());nodes=[]
 for n in t.body:
  if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and (n.name in functions or all_functions):nodes.append(n)
  elif isinstance(n,ast.ClassDef):
   if n.name in classes:nodes.append(n)
   elif methods and n.name in methods:
    wanted=methods[n.name];body=[z for z in n.body if isinstance(z,(ast.FunctionDef,ast.AsyncFunctionDef)) and z.name in wanted];assert {z.name for z in body}==set(wanted)
    nodes.append(ast.ClassDef(name=n.name,bases=[],keywords=[],body=body,decorator_list=[]))
  elif isinstance(n,ast.Assign):
   names=[z.id for z in n.targets if isinstance(z,ast.Name)]
   if any(z in aliases or (constant_prefix and z.startswith(constant_prefix)) for z in names):nodes.append(n)
 for n in nodes:records.append({'path':path,'symbol':getattr(n,'name',ast.unparse(n.targets[0]) if isinstance(n,ast.Assign) else '?'),'AST_sha256':hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest(),'complete_body':True})
 module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[]);ast.fix_missing_locations(module);exec(compile(module,str(p),'exec'),ns)
def import_source(name,path):
 spec=importlib.util.spec_from_file_location(name,S/path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);records.append({'path':path,'symbol':'WHOLE_MODULE','SHA256':hashlib.sha256((S/path).read_bytes()).hexdigest(),'complete_module_imported':True});return m
# Package shells bypass heavy sglang.__init__, all endpoint constructors and runtime registry.
for name in ['sglang','sglang.srt','sglang.srt.entrypoints','sglang.srt.entrypoints.openai']:
 if name not in sys.modules:m=types.ModuleType(name);m.__path__=[];sys.modules[name]=m
load('python/sglang/utils.py',functions=['convert_json_schema_to_str']);u=types.ModuleType('sglang.utils');u.convert_json_schema_to_str=ns['convert_json_schema_to_str'];sys.modules[u.__name__]=u
protocol=import_source('sglang.srt.entrypoints.openai.protocol','python/sglang/srt/entrypoints/openai/protocol.py');sse=import_source('sglang.srt.entrypoints.openai.sse_utils','python/sglang/srt/entrypoints/openai/sse_utils.py');usage=import_source('sglang.srt.entrypoints.openai.usage_processor','python/sglang/srt/entrypoints/openai/usage_processor.py')
ns.update({n:getattr(protocol,n) for n in ['ChatCompletionRequest','ChatCompletionMessageGenericParam','ChatCompletionStreamResponse','ChatCompletionResponseStreamChoice','DeltaMessage','ErrorResponse','PromptTokensDetails','SglExt','StreamOptions']});ns['UsageProcessor']=usage.UsageProcessor;ns['build_sse_content']=sse.build_sse_content
load('python/sglang/srt/beam_search/types.py',classes=['BeamSearchSequence']);load('python/sglang/srt/utils/weight_versions.py',classes=['WeightVersionSpan'],aliases=['WeightVersionSpans'],functions=['add_weight_versions_to_meta_info'])
aliases=['TokenLogprobValues','TokenLogprobIndices','TopLogprobValues','TopLogprobIndices','TokenIdsLogprobValues','TokenIdsLogprobIndices','HiddenStateChunk','OutputHiddenStates','CachedTokensDetails','FinishReasonDict']
load('python/sglang/srt/managers/io_struct.py',classes=['BaseReq','BaseBatchReq','BeamSearchOutput','PickleWrapper','BatchStrOutput','AbortReq'],aliases=aliases,functions=['_maybe_wrap_pickle','_maybe_unwrap_pickle','msgpack_encode','msgpack_decode','wrap_as_pickle','unwrap_from_pickle','sock_send','sock_recv'])
class UnusedNativeTensor:pass
ns.update(torch=NS(Tensor=UnusedNativeTensor),CudaIpcTensorTransportProxy=type('UnusedNativeProxy',(),{}))
load('python/sglang/srt/utils/msgpack_utils.py',all_functions=True,constant_prefix='_MSGPACK')
ns['_primitive_types']=(int,float,bool,bytes);ns['_msgpack_encoder']=msgspec.msgpack.Encoder(enc_hook=ns['enc_hook']);ns['_msgpack_decoder']=msgspec.msgpack.Decoder(typing.Union[ns['BatchStrOutput'],ns['AbortReq'],int,float,bool,bytes],dec_hook=ns['dec_hook'],ext_hook=ns['ext_hook']);ns['_USE_PICKLE_IPC']=True
class GenerateFixture(NS):pass
class EmbeddingFixture(NS):pass
class TokenIDFixture(NS):pass
class EmbeddingOutputFixture(NS):pass
ns.update(GenerateReqInput=GenerateFixture,EmbeddingReqInput=EmbeddingFixture,BatchTokenIDOutput=TokenIDFixture,BatchEmbeddingOutput=EmbeddingOutputFixture)
ns['_ragged_verify_cap_accept']=lambda:False
load('python/sglang/srt/beam_search/output.py',functions=['try_build_beam_search_out_dict'])
load('python/sglang/srt/managers/tokenizer_manager.py',classes=['ReqState'],aliases=['_INCREMENTAL_STREAMING_META_INFO_KEYS'],functions=['_slice_streaming_output_meta_info'],methods={'TokenizerManager':['_handle_batch_output','_calculate_spec_decoding_metrics','_coalesce_streaming_chunks','_wait_one_response','_stream_one_response','_handle_abort_finish_reason','abort_request','create_abort_task']})
load('python/sglang/srt/entrypoints/openai/utils.py',functions=['should_include_usage','spec_tokens_details_from_meta_info'])
load('python/sglang/srt/entrypoints/openai/serving_base.py',methods={'OpenAIServingBase':['create_streaming_error_response','create_error_response']})
load('python/sglang/srt/entrypoints/openai/serving_chat.py',methods={'OpenAIServingChat':['_should_return_input_ids','_should_return_output_ids','_continuous_usage_cached_details','_reported_prompt_tokens','_effective_tools','_tool_call_parsing_active','_generate_stream_content','_generate_chat_stream','_handle_streaming_request']})
Manager=ns['TokenizerManager'];Chat=ns['OpenAIServingChat'];State=ns['ReqState'];Batch=ns['BatchStrOutput']
class TimeFixture:
 def __init__(self):self.first_token_time=0.;self.response_sent_to_client_time=0.;self.trace_ctx=NS(tracing_enable=False)
 def set_first_token_time(self):self.first_token_time=1.
 def set_finished_time(self,**kw):pass
 def get_e2e_latency(self):return 0.
 def set_response_sent_to_client_time(self):self.response_sent_to_client_time=1.
 def get_response_sent_to_client_realtime(self):return 0.
class Wire:
 def __init__(self,pickled):
  self.pickled=pickled;self.ctx=zmq.Context();self.a=self.ctx.socket(zmq.PAIR);self.b=self.ctx.socket(zmq.PAIR)
  for s in [self.a,self.b]:s.setsockopt(zmq.RCVTIMEO,2000);s.setsockopt(zmq.SNDTIMEO,2000);s.setsockopt(zmq.LINGER,0)
  url='inproc://lf31-'+uuid.uuid4().hex;self.a.bind(url);self.b.connect(url);self.records=[]
 def exchange(self,obj):
  ns['_USE_PICKLE_IPC']=self.pickled;ns['sock_send'](self.a,obj);out=ns['sock_recv'](self.b);self.records.append({'type':type(out).__name__,'finish_code_types':[type(x.get('status_code')).__name__ if x else None for x in getattr(out,'finished_reasons',[])]});return out
 def close(self):self.a.close();self.b.close();self.ctx.term()
def manager(incremental=False,notify=16):
 cfg=NS(batch_notify_size=notify,tokenizer_worker_num=1,stream_response_default_include_usage=False,incremental_streaming_output=incremental,return_input_ids=False,return_output_ids=False,enable_cache_report=False,weight_version=None)
 ns['get_serving']=lambda:cfg;ns['get_spec']=lambda:NS(speculative_algorithm='EAGLE_FIXTURE',speculative_num_draft_tokens=4)
 m=Manager();m.rid_to_state={};m.incremental_streaming_output=incremental;m.enable_metrics=m.enable_lora=False;m.skip_tokenizer_init=False;m.dump_requests_folder=m.crash_dump_folder=None;m.config_value=lambda key:getattr(cfg,key,None);m.request_logger=NS(log_finished_request=lambda *a,**k:None);m.request_metrics_exporter_manager=NS(exporter_enabled=lambda:False);m.dispatched=[];m._dispatch_to_scheduler=m.dispatched.append;return m,cfg
load('python/sglang/utils.py',functions=['_is_chinese_char','find_printable_text'])
load('python/sglang/srt/utils/patch_tokenizer.py',functions=['decode_without_hf_kwargs'])
load('python/sglang/srt/managers/detokenizer_manager.py',classes=['DecodeStatus','LimitedCapacityDict'],methods={'DetokenizerManager':['trim_matched_stop','_clamp_decode_ids','_grouped_batch_decode','_decode_batch_token_id_output']})
ns['DETOKENIZER_MAX_STATES']=65536;ns['INIT_INCREMENTAL_DETOKENIZATION_OFFSET']=5;ns['OrderedDict']=__import__('collections').OrderedDict;ns['defaultdict']=__import__('collections').defaultdict
load('python/sglang/srt/managers/schedule_batch.py',methods={'Req':['output_ids_through_stop','init_incremental_detokenize']})
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained(str(R/'fixtures/L030-exact-tokenizer'),local_files_only=True,trust_remote_code=False)
def encode(text):return tokenizer.encode(text,add_special_tokens=False)
def add_state(m,rid,prompt_ids,stream=True):
 obj=GenerateFixture(rid=rid,stream=stream,background=False,return_logprob=False,return_sampling_mask=False,log_metrics=False,lora_path=None,is_single=True)
 state=State(out_list=[],finished=False,event=asyncio.Event(),obj=obj,time_stats=TimeFixture());state.prompt_token_ids=prompt_ids;m.rid_to_state[rid]=state;return obj,state
class Upstream:
 def __init__(self,rids,budget,reason=None):
  self.rids=rids;self.budget=budget;self.reason=reason;self.turn=0;self.pos=0;self.prompt_ids=encode('固定组件 prompt 你好.\n');self.queries=[];self.IDs={};self.delivered={r:[] for r in rids}
  for i,r in enumerate(rids):
   self.IDs[r]=encode(('中文 hello 世界 𠮷 🧑🏽‍🚀\n' if i%2==0 else 'English \\"quote\\" text 😀\n')*100)[:budget]
   q=ns['Req']();q.rid=r;q.origin_input_ids_unpadded=array.array('q',self.prompt_ids);q.output_ids=array.array('q');q.finished_len=None;q.surr_offset=q.read_offset=None;q.cur_decode_ids_len=0;q.surr_and_decode_ids=None;q.send_decode_id_offset=q.send_token_offset=0;self.queries.append(q)
  self.detok=ns['DetokenizerManager']();self.detok.tokenizer=tokenizer;self.detok.vocab_size=len(tokenizer);self.detok.disable_tokenizer_batch_decode=False;self.detok.is_tool_call_parser_gpt_oss=False;self.detok.decode_status=ns['LimitedCapacityDict'](65536)
 def next(self):
  end=min(self.budget,self.pos+4);self.turn+=1;fin=end==self.budget;rows=[]
  for q in self.queries:
   q.output_ids.extend(self.IDs[q.rid][self.pos:end]);q.finished_len=end if fin else None;ids,off=q.init_incremental_detokenize();delta=list(ids[q.send_decode_id_offset:]);q.send_decode_id_offset=len(ids);outids=array.array('q',q.output_ids[q.send_token_offset:]);q.send_token_offset=len(q.output_ids);rows.append((q.rid,delta,off,outids))
  if self.turn%2:rows.reverse()
  reasons=[self.reason or {'type':'length','length':self.budget} if fin else None for _ in rows]
  recv=NS(rids=[z[0] for z in rows],decoded_texts=['']*len(rows),decode_ids=[z[1] for z in rows],read_offsets=[z[2] for z in rows],finished_reasons=reasons,skip_special_tokens=[True]*len(rows),spaces_between_special_tokens=[True]*len(rows),no_stop_trim=[False]*len(rows));texts=self.detok._decode_batch_token_id_output(recv)
  fields={x.name:None for x in msgspec.structs.fields(Batch) if x.default is msgspec.NODEFAULT and x.default_factory is msgspec.NODEFAULT};fields.update(rids=recv.rids,http_worker_ipcs=[None]*len(rows),finished_reasons=reasons,output_strs=texts,output_ids=[z[3] for z in rows],prompt_tokens=[len(self.prompt_ids)]*len(rows),completion_tokens=[end]*len(rows),reasoning_tokens=[0]*len(rows),cached_tokens=[0]*len(rows),retraction_counts=[0]*len(rows),spec_verify_ct=[self.turn]*len(rows),spec_num_correct_drafts=[end-self.turn]*len(rows),spec_num_block_accept_tokens=[end]*len(rows),spec_num_cap_tokens=[4*self.turn]*len(rows),spec_correct_drafts_histogram=[[] for _ in rows],spec_cap_lens_histogram=[[] for _ in rows]);self.pos=end;return Batch(**fields)
def chat(m,prepared=None):
 c=Chat();c.tokenizer_manager=m;c.reasoning_parser=c.tool_call_parser=None;c.chat_encoding_spec=None
 for n in ['create_streaming_error_response','create_error_response']:setattr(c,n,types.MethodType(getattr(ns['OpenAIServingBase'],n),c))
 if prepared is not None:
  async def generate_request(adapted,raw):
   async for out in prepared[adapted.rid]:yield out
  m.generate_request=generate_request
 return c
def request_model(usage=True,continuous=False):
 return protocol.ChatCompletionRequest(model='GLM-5.3-component-fixture',messages=[{'role':'user','content':'synthetic protocol fixture'}],stream=True,stream_options={'include_usage':usage,'continuous_usage_stats':continuous},tool_choice='none',return_input_ids_in_sglext=True,return_output_ids_in_sglext=True)
def parse_sse(strings):
 out=[]
 for s in strings:
  for frame in s.split('\n\n'):
   if not frame:continue
   data='\n'.join(line[6:] for line in frame.splitlines() if line.startswith('data: '))
   if data:out.append('[DONE]' if data=='[DONE]' else json.loads(data))
 return out
BOUNDARIES=['Heavy sglang package initializers/serving constructors bypassed; protocol/SSE/usage modules imported completely, manager/chat methods extracted whole','GenerateReqInput and scheduling/model accepted IDs are fixtures, with original incremental detok bodies; no native model or physical P/D','BS/Abort schemas and array hooks are original; decoder registry narrowed to tested carriers, not whole IPC message union','Torch/CUDA proxy namespace sentinels never encoded; no tensor, CUDA/NPU transport, MM, beam or optional logprob branches executed','Request times/logger/metrics/exporter are fixtures; no latency/throughput claims; ragged cap-accept predicate fixed False fixture branch','Raw handler and generator bridge uses prepared waiter fixture; full generate_request preprocessing/tokenization/dispatch not imported','inproc ZeroMQ same process real wire is not cross-host PD or OS process-lifetime evidence']
