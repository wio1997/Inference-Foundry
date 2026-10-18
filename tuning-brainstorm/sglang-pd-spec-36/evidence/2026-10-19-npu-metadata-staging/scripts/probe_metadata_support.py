#!/usr/bin/env python3
"""Exact source predicates and bounded CPU ownership counterexamples; no NPU work."""
import argparse,array,ast,copy,datetime,enum,hashlib,itertools,json,pathlib,sys,types
ap=argparse.ArgumentParser();ap.add_argument('--source-root');ap.add_argument('--stdout',action='store_true');ap.add_argument('--expected-hashes');args=ap.parse_args()
ROOT=pathlib.Path(__file__).resolve().parents[1];S=pathlib.Path(args.source_root) if args.source_root else ROOT/'sources/installed/python/sglang/srt'
expected=json.loads(args.expected_hashes) if args.expected_hashes else {};hashes={};extracted={}
accelerator_nodes=sorted({str(p) for pattern in ['davinci*','devmm_svm','hisi_hdc','ascend*'] for p in pathlib.Path('/dev').glob(pattern)})
assert not accelerator_nodes,'CPU-only probe requires no accelerator device nodes'
import numpy as np
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
assert 'torch_npu' not in sys.modules,'CPU-only probe forbids torch_npu autoload'
def source(path):
 b=(S/path).read_bytes();h=hashlib.sha256(b).hexdigest();hashes[path]=h
 if path in expected:assert expected[path]==h,'Source drift: '+path
 return ast.parse(b)
def get_node(path,name,cls=None):
 t=source(path);items=t.body if cls is None else next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name==cls).body
 n=next(n for n in items if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
 extracted[(cls+'.' if cls else '')+name]={'file':path,'line':n.lineno,'end_line':n.end_lineno,'ast_sha256':hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()};return n
env={'Enum':enum.Enum,'auto':enum.auto,'CustomSpecAlgo':type('UnusedCustomSpecAlgo',(),{}),'np':np,'torch':torch}
def install(nodes):
 m=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*copy.deepcopy(nodes)],type_ignores=[]);exec(compile(ast.fix_missing_locations(m),'<exact-source-cpu-probe>','exec'),env)
reg=source('speculative/spec_registry.py');aliases=next(n for n in reg.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_RESERVED_ALIASES' for t in n.targets));install([aliases])
install([get_node('speculative/spec_info.py','SpeculativeAlgorithm')]);A=env['SpeculativeAlgorithm'];assert A.from_string('NEXTN')==A.EAGLE
for name in ['sglang','sglang.srt','sglang.srt.speculative','sglang.srt.speculative.spec_info']:
 assert name not in sys.modules;mod=types.ModuleType(name);mod.__path__=[];sys.modules[name]=mod
sys.modules['sglang.srt.speculative.spec_info'].SpeculativeAlgorithm=A
install([get_node('managers/overlap_utils.py','decide_needs_cpu_seq_lens')]);gate=env['decide_needs_cpu_seq_lens'];gate_rows=[]
states={'unset':None,'undeclared':object(),'false':types.SimpleNamespace(needs_cpu_seq_lens=False),'true':types.SimpleNamespace(needs_cpu_seq_lens=True)}
for algorithm,tbo,labels in itertools.product([None,'EAGLE','NEXTN','NGRAM','DFLASH','DSPARK'],[False,True],itertools.product(states,repeat=3)):
 backends=[states[k] for k in labels];env['get_exec']=lambda:types.SimpleNamespace(overlap=types.SimpleNamespace(enable_two_batch_overlap=tbo));env['get_spec']=lambda:types.SimpleNamespace(speculative_algorithm=algorithm)
 want=tbo or algorithm=='NGRAM' or any(k in ('undeclared','true') for k in labels);got=gate(backends);assert got==want
 gate_rows.append({'algorithm':algorithm,'tbo':tbo,'backend_flags':list(labels),'needs_cpu_seq_lens':got})
pin=get_node('platforms/npu.py','is_pin_memory_available','NPUDeviceMixin');pin.name='npu_pin_declaration';install([pin]);env['current_platform']=types.SimpleNamespace(is_pin_memory_available=types.MethodType(env['npu_pin_declaration'],object()))
install([get_node('utils/common.py','is_pin_memory_available')]);pin_rows=[]
for device in [None,'npu','npu:0','cpu',torch.device('cpu')]:
 v=env['is_pin_memory_available'](device);assert v is False;pin_rows.append({'device_argument':str(device),'declared_pinned_available':v})
# Select exactly the CPU int32 assignment and subsequent offset block from the real method.
method=get_node('hardware_backend/npu/attention/ascend_backend.py','init_forward_metadata','AscendAttnBackend')
field='self.forward_metadata.seq_lens_cpu_int'
assignment=next(n for n in method.body if isinstance(n,ast.Assign) and any(ast.unparse(t)==field for t in n.targets) and ast.unparse(n.value)=='forward_batch.seq_lens_cpu.int()')
blocks=[n for n in method.body if isinstance(n,ast.If) and any(isinstance(a,ast.AugAssign) and ast.unparse(a.target)==field for a in ast.walk(n))];assert len(blocks)==1
fn=ast.parse('def cpu_metadata_slice(self, forward_batch, compact=None):\n pass').body[0];fn.body=[copy.deepcopy(assignment),copy.deepcopy(blocks[0])];install([fn]);slice_fn=env['cpu_metadata_slice'];alias_rows=[]
for dtype,algorithm,mode,step,draft in itertools.product([torch.int64,torch.int32],[None,'EAGLE','NEXTN','DFLASH','DSPARK'],['target_verify','decode','normal'],[0,1,2],[1,4]):
 initial=[0,511,32768];lengths=torch.tensor(initial,dtype=dtype,device='cpu');f=types.SimpleNamespace(forward_metadata=types.SimpleNamespace(),speculative_step_id=step);algo=A.from_string(algorithm) if algorithm is not None else None
 b=types.SimpleNamespace(seq_lens_cpu=lengths,forward_mode=types.SimpleNamespace(is_target_verify=lambda:mode=='target_verify',is_decode_or_idle=lambda:mode=='decode'),spec_algorithm=algo,spec_info=types.SimpleNamespace(draft_token_num=draft),seq_lens_sum=sum(initial))
 offset=(draft if algo is None or not algo.is_dflash_family() else 0) if mode=='target_verify' else step+1 if mode=='decode' else 0
 slice_fn(f,b);meta=f.forward_metadata.seq_lens_cpu_int;assert meta.tolist()==[x+offset for x in initial]
 changed=lengths.tolist()!=initial;assert changed==(dtype==torch.int32 and offset!=0)
 if dtype==torch.int64:assert meta.data_ptr()!=lengths.data_ptr()
 else:assert meta.data_ptr()==lengths.data_ptr()
 alias_rows.append({'dtype':str(dtype),'algorithm':algorithm,'mode':mode,'step':step,'draft':draft,'offset':offset,'metadata':meta.tolist(),'source_after':lengths.tolist(),'source_changed':changed,'source_sum_stale':int(lengths.sum())!=b.seq_lens_sum})
# Same input to two consecutive draft metadata consumers, as the wrapper does.
draft_sequences=[]
for dtype in [torch.int64,torch.int32]:
 lengths=torch.tensor([10,100],dtype=dtype);saved=[];b=types.SimpleNamespace(seq_lens_cpu=lengths,forward_mode=types.SimpleNamespace(is_target_verify=lambda:False,is_decode_or_idle=lambda:True),spec_algorithm=A.EAGLE,spec_info=types.SimpleNamespace(draft_token_num=4))
 for step in [0,1]:
  f=types.SimpleNamespace(forward_metadata=types.SimpleNamespace(),speculative_step_id=step);slice_fn(f,b);saved.append(f.forward_metadata.seq_lens_cpu_int)
 got=[x.tolist() for x in saved];want=[[11,101],[12,102]]
 assert (got==want)==(dtype==torch.int64)
 draft_sequences.append({'dtype':str(dtype),'source_after':lengths.tolist(),'saved_metadata_after_two_steps':got,'snapshot_equivalent_to_int64_baseline':got==want})
install([get_node('utils/common.py','flatten_arrays_to_pinned_cpu'),get_node('managers/overlap_utils.py','resolve_forward_inputs'),get_node('managers/overlap_utils.py','_resolve_spec_extras','FutureMap')]);env['_DEBUG_ASSERT']=False
flatten=env['flatten_arrays_to_pinned_cpu'];resolve=env['resolve_forward_inputs'];stage_rows=[]
for pieces in [[[1,2,3]],[[],[4,5],[]],[[7],[8,9]],[[0]*512],[[]]]:
 parts=[array.array('q',x) for x in pieces];want=[v for x in pieces for v in x];tensor=flatten(parts,False);assert tensor.device.type=='cpu' and tensor.dtype==torch.int64 and tensor.tolist()==want and not tensor.is_pinned()
 for a in parts:
  if a:a[0]=999
 assert tensor.tolist()==want
 stage_rows.append({'part_lengths':[len(x) for x in parts],'flattened_count':len(want),'caller_mutation_isolated':True,'pinned':False})
borrowed=[]
for variant in ['current-concatenate','unconditional-borrowed-view']:
 part=array.array('q',[11,22,33]);tensor=flatten([part],False) if variant=='current-concatenate' else torch.from_numpy(np.frombuffer(part,dtype=np.int64));part[0]=999
 batch=types.SimpleNamespace(prefill_input_ids_cpu=tensor,device='cpu',mix_running_indices=None,enable_overlap=True,spec_algorithm=A.EAGLE,spec_info=None,input_ids=None)
 fmap=types.SimpleNamespace(spec_algo=A.EAGLE);fmap._resolve_spec_extras=types.MethodType(env['_resolve_spec_extras'],fmap);resolve(batch,fmap)
 got=batch.input_ids.tolist();assert got==([11,22,33] if variant=='current-concatenate' else [999,22,33]);assert batch.prefill_input_ids_cpu is None
 borrowed.append({'variant':variant,'input_ids_at_deferred_resolve':got,'preserves_prepared_snapshot':got==[11,22,33]})
script_sha=globals().get('_IF_SCRIPT_SHA256') or hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
result={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'correctness':'PASS','exit_code':0,'script_sha256':script_sha,'source_hashes':hashes,'extracted_nodes':extracted,'compile_placeholder':'CustomSpecAlgo class-default placeholder only; no tested method uses it','gate_cases':len(gate_rows),'gate_rows':gate_rows,'npu_pin_declaration':pin_rows,'metadata_slice_lines':[assignment.lineno,blocks[0].lineno,blocks[0].end_lineno],'alias_cases':len(alias_rows),'alias_rows':alias_rows,'counterfactual_source_mutations':sum(x['source_changed'] for x in alias_rows),'draft_sequences':draft_sequences,'staging_cases':stage_rows,'borrowed_view_counterexample':borrowed,'environment':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'torch_npu_loaded':'torch_npu' in sys.modules,'accelerator_device_nodes':accelerator_nodes,'all_computed_tensors':'CPU'},'scope':'Exact-source CPU predicates and selected metadata statements only. compact=None, no graph/FIA kernels. H2D is represented by resolve to CPU, so no DMA/event/pin-performance or NPU correctness claim. Counterfactual int32 alias and borrowed-view failures reject unconditional equivalence, not all possible optimized ownership designs. Overall serving remains E0.'}
if args.stdout:print(json.dumps(result))
else:
 o=ROOT/'runs/L007-R003-cpu-support';o.mkdir(parents=True,exist_ok=False);(o/'result.json').write_text(json.dumps(result,indent=2)+'\n');(o/'receipt.json').write_text(json.dumps({'script_sha256':script_sha,'source_hashes':hashes,'result_sha256':hashlib.sha256((o/'result.json').read_bytes()).hexdigest(),'exit_code':0,'correctness':'PASS','scope':result['scope']},indent=2)+'\n');print(json.dumps({'gate_cases':len(gate_rows),'alias_cases':len(alias_rows),'counterfactual_source_mutations':result['counterfactual_source_mutations'],'correctness':'PASS'}))
