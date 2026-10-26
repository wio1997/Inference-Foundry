#!/usr/bin/env python3
"""Exact-source graph admission/bucket semantics, with no native graph objects."""
import ast,bisect,copy,dataclasses,datetime,hashlib,itertools,json,pathlib,sys,types
from typing import Optional
import torch
R=pathlib.Path(__file__).resolve().parents[1];S=R/'sources/installed/python/sglang/srt'
assert not any(pathlib.Path('/dev').glob('davinci*'))
assert 'torch_npu' not in sys.modules
torch.set_num_threads(1);torch.set_num_interop_threads(1)
import numpy as np
OUT=R/'runs/L008-R002-graph-cpu';OUT.mkdir(parents=True,exist_ok=False)
sources={};nodes={};env=globals();dataclass=dataclasses.dataclass
def node(path,name,cls=None):
 p=S/path;b=p.read_bytes();sources[path]=hashlib.sha256(b).hexdigest();t=ast.parse(b)
 items=t.body if cls is None else next(x for x in t.body if isinstance(x,ast.ClassDef) and x.name==cls).body
 n=next(x for x in items if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name==name)
 nodes[(cls+'.' if cls else '')+name]={'file':path,'line':n.lineno,'end_line':n.end_lineno,'ast_sha256':hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()};return copy.deepcopy(n)
def install(body):
 m=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*body],type_ignores=[]);exec(compile(ast.fix_missing_locations(m),'<fixed-graph-source>','exec'),env)
def cls(name,path,original,methods):
 install([ast.ClassDef(name=name,bases=[],keywords=[],body=[node(path,f,original) for f in methods],decorator_list=[])])
common='utils/common.py';base='model_executor/runner/base_cuda_graph_runner.py';decode='model_executor/runner/decode_cuda_graph_runner.py';npu='hardware_backend/npu/graph_runner/npu_graph_runner.py';backend='hardware_backend/npu/graph_runner/npu_cudagraph_backend.py'
install([node('model_executor/runner/shape_key.py','ShapeKey'),*[node(common,f) for f in ['ceil_align','get_cuda_graph_batch_size_alignment','get_cuda_graph_max_batch_size']],node(base,'get_batch_sizes_to_capture')])
cls('BucketProbe',base,'BaseCudaGraphRunner',['_pad_to_bucket'])
cls('AdmissionProbe',decode,'DecodeCudaGraphRunner',['_max_dp_batch_size','_make_graph_key','_resolve_attention_variant','_resolve_lora_variant','can_run_graph'])
cls('MembershipProbe',backend,'NPUCudaGraphBackend',['can_run'])
cls('UpdateProbe',npu,'NPUGraphRunner',['_uses_v2_seq_len_update','_get_update_attr_name','_get_update_attr_type','_update_inputs'])
class AttentionArch:MLA='MLA';MHA='MHA'
NS=types.SimpleNamespace
config_rows=[]
def configure(tbo,gather,tp,cp,requested,width,pool,compile_enabled):
 global get_exec,get_parallel,require_gathered_buffer,get_flags
 get_exec=lambda:NS(overlap=NS(enable_two_batch_overlap=tbo),graph=NS(cuda_graph_config=NS(decode=NS(bs=requested)),torch_compile_max_bs=8))
 get_parallel=lambda:NS(attn_tp_size=tp,attn_cp_size=cp)
 require_gathered_buffer=lambda:gather
 get_flags=lambda:NS(capture=NS(enable_torch_compile=compile_enabled))
 return get_batch_sizes_to_capture(NS(req_to_token_pool=NS(size=pool)),width)
for tbo,gather,tp,cp,width,pool,compile_enabled in itertools.product([False,True],[False,True],[1,2,4,16],[1,2,4],[1,2,4],[1,3,17],[False,True]):
 requested=[1,2,3,4,8,16,32];capture,compile_bs=configure(tbo,gather,tp,cp,requested,width,pool,compile_enabled);alignment=get_cuda_graph_batch_size_alignment();maximum=get_cuda_graph_max_batch_size(pool);effective_width=1 if tbo else width
 assert capture==sorted(set(capture)) and all(0<x<=maximum and x*effective_width%alignment==0 for x in capture)
 assert compile_bs==([x for x in capture if x<=8] if compile_enabled else [])
 for raw in range(maximum+1):
  padded=BucketProbe._pad_to_bucket(raw,capture);assert padded in capture and padded>=raw and not any(raw<=x<padded for x in capture)
 try:BucketProbe._pad_to_bucket(maximum+1,capture)
 except AssertionError:pass
 else:raise AssertionError('Oversize input was not rejected')
 config_rows.append({'tbo':tbo,'gather_dependency':gather,'attn_tp':tp,'attn_cp':cp,'width':width,'pool':pool,'compile_enabled':compile_enabled,'alignment':alignment,'rounded_max':maximum,'capture':capture,'compile':compile_bs})
counterexamples=[]
for gather,tbo in [(True,False),(False,False),(False,True)]:
 before,_=configure(tbo,gather,16,1,[1,4,8,16],4,32,False);after,_=configure(tbo,gather,16,1,[1,2,3,4,8,16],4,32,False)
 counterexamples.append({'gather_dependency':gather,'tbo':tbo,'capture_width':4,'attn_tp':16,'before':before,'after_add_2_3':after,'selected_buckets_for_raw_2_3_before':[BucketProbe._pad_to_bucket(x,before) for x in [2,3]],'selected_buckets_for_raw_2_3_after':[BucketProbe._pad_to_bucket(x,after) for x in [2,3]]})
assert counterexamples[0]['before']==counterexamples[0]['after_add_2_3']==[4,8,16]
assert counterexamples[1]['selected_buckets_for_raw_2_3_after']==[2,3]
membership=MembershipProbe();membership._graphs={ShapeKey(4,None,'nolora','sparse'):object()};key_rows=[]
for size,stream,variant,attn in itertools.product([2,4,8],[None,0,1],[None,'nolora','lora'],[None,'sparse','dense']):
 k=ShapeKey(size,stream,variant,attn);got=membership.can_run(None,k);assert got==(size==4 and stream is None and variant=='nolora' and attn=='sparse');key_rows.append({'key':dataclasses.asdict(k),'member':got})
# Admission uses actual can_run_graph. Native membership and configuration inputs are CPU fixtures.
admit=AdmissionProbe();admit.ragged_verify_mode=False;admit.require_mlp_sync=False;admit.is_encoder_decoder=False;admit.enable_two_batch_overlap=False;admit.record_nolora_graph=True;admit.enable_pdmux=False;admit.attention_graph_variants=NS(select=lambda fb:'sparse');admit.require_mlp_tp_gather=False;admit.max_bs=8;admit.backend=membership
membership._graphs={ShapeKey(x,None,y,'sparse'):object() for x,y in itertools.product([1,2,4,8],['nolora','lora'])}
admission_rows=[]
for width,raw,actual_width,padding,gather,lora in itertools.product([1,4],[0,1,2,3,4,5,8,9],[None,0,1,4,8],[False,True],[False,True],[False,True]):
 admit.captured_req_width=width;admit.disable_padding=not padding;admit.require_mlp_tp_gather=gather;admit.model_runner=NS(spec_algorithm=NS(is_ngram=lambda:False))
 fb=NS(replace_embeds=None,spec_info=None if actual_width is None else NS(num_tokens_per_req=actual_width),batch_size=raw,original_global_num_tokens_cpu=[raw,max(raw,3)],lora_ids=['adapter' if lora else None])
 bs=max(raw,3) if gather else raw;want=(actual_width is None or actual_width==0 or actual_width==width) and (bs<=8 if padding else bs in [1,2,4,8]);got=bool(admit.can_run_graph(fb));assert got==want
 admission_rows.append({'captured_width':width,'raw_bs':raw,'actual_width':actual_width,'padding':padding,'gather_dependency':gather,'lora':lora,'max_dp_request_count':bs,'admitted':got})
guard_rows=[]
admit.captured_req_width=4;admit.disable_padding=False;admit.require_mlp_tp_gather=False;admit.require_mlp_sync=True;admit.is_encoder_decoder=True;admit.enable_two_batch_overlap=True;admit.model_runner=NS(spec_algorithm=NS(is_ngram=lambda:True))
for replace,vote,encoder,tbo,ngram_size in itertools.product([False,True],repeat=5):
 fb=NS(replace_embeds=object() if replace else None,spec_info=NS(num_tokens_per_req=4),batch_size=2,lora_ids=[None],can_run_decode_cuda_graph=vote,encoder_lens=torch.tensor([1,1] if encoder else [1,0]),can_run_tbo=tbo,input_ids=torch.arange(8 if ngram_size else 7))
 got=bool(admit.can_run_graph(fb));want=not replace and vote and encoder and tbo and ngram_size;assert got==want;guard_rows.append({'replace':replace,'dp_vote':vote,'encoder_positive':encoder,'tbo_vote':tbo,'ngram_size_match':ngram_size,'admitted':got})
missing_dp=NS(original_global_num_tokens_cpu=None)
try:AdmissionProbe._max_dp_batch_size(missing_dp)
except RuntimeError as exc:missing_dp_error=str(exc)
else:raise AssertionError('Missing DP request counts must raise')
update_rows=[]
class Spy:
 def update(self,**kwargs):self.calls.append(kwargs)
u=UpdateProbe();u.attr_name={'TARGET_VERIFY':'actual_seq_kvlen','MLA':'actual_seq_lengths_kv'};u.attr_type={'TARGET_VERIFY':[],'MLA':[]}
for if_v2,use_bs,cap_verify,runtime_idle in itertools.product([False,True],repeat=4):
 u.if_use_v2=if_v2;u.use_fias_v2_bsnd=use_bs;u.capture_forward_mode=NS(is_target_verify=lambda:cap_verify);u.runtime_forward_mode=NS(is_idle=lambda:runtime_idle)
 want=if_v2 or (use_bs and cap_verify);assert u._uses_v2_seq_len_update()==want;u.update_attr_name=u._get_update_attr_name();u.update_attr_type=u._get_update_attr_type();u.bs=4;spy=Spy();spy.calls=[];u.graphs={4:spy};values=[1,512,32768];u._update_inputs(values);expected_attr='actual_seq_kvlen' if want else 'actual_seq_lengths_kv';assert spy.calls==[{'cpu_update_input':[{expected_attr:values}]}]
 update_rows.append({'if_use_v2':if_v2,'use_fias_v2_bsnd':use_bs,'captured_target_verify':cap_verify,'runtime_idle_fixture':runtime_idle,'selected_attr':expected_attr,'native_graph':'spy only'})
u.update_attr_type=torch.empty(0);u.update_attr_name='tensor-fixture';spy=Spy();spy.calls=[];u.graphs={4:spy};u._update_inputs([1,512,32768]);tensor=spy.calls[0]['cpu_update_input'][0]['tensor-fixture'];assert tensor.dtype==torch.int32 and tensor.device.type=='cpu' and tensor.tolist()==[1,512,32768]
result={'source_commit':'ce0a8fea5f4245839d174b2b8c55692be1810a7f','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'correctness':'PASS','source_hashes':sources,'extracted_nodes':nodes,'configuration_cases':len(config_rows),'configuration_rows':config_rows,'bucket_addition_cases':counterexamples,'key_membership_cases':len(key_rows),'key_rows':key_rows,'admission_cases':len(admission_rows),'admission_rows':admission_rows,'guard_cases':len(guard_rows),'guard_rows':guard_rows,'missing_dp_error':missing_dp_error,'update_key_cases':len(update_rows),'update_rows':update_rows,'tensor_update_payload':{'dtype':str(tensor.dtype),'device':str(tensor.device),'values':tensor.tolist()},'environment':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'torch_npu_loaded':'torch_npu' in sys.modules},'scope':'Exact-source CPU admission/bucket functions and Python graph.update argument creation only. Explicit injected config, gathered-buffer predicate and backend graph dictionary; no runtime configuration inferred, no native graph/event/replay/all_reduce/device kernels executed. Ragged admission not tested; NPU graph capability and performance remain unknown. All booleans are policy results, not successful graph replay. Overall serving E0.'}
p=OUT/'result.json';p.write_text(json.dumps(result,indent=2)+'\n');receipt={'exit_code':0,'correctness':'PASS','script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'result_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'counts':{k:v for k,v in result.items() if k.endswith('_cases') and isinstance(v,int)},'scope':result['scope']};(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt['counts']))
