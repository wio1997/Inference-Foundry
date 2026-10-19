#!/usr/bin/env python3
"""Exact Python wrapper contract only; native class base and events are not loaded."""
import ast,copy,datetime,difflib,hashlib,json,pathlib,types
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L009-R005-wrapper-contract';O.mkdir(parents=True,exist_ok=False)
package=R/'sources/installed-packages/torch_npu/npu/streams.py';tree=ast.parse(package.read_bytes());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Stream');methods={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)};env={}
module=ast.Module(body=[ast.ClassDef(name='PythonWrapperOnly',bases=[],keywords=[],decorator_list=[],body=[copy.deepcopy(methods[n]) for n in ['wait_event','wait_stream']])],type_ignores=[]);exec(compile(ast.fix_missing_locations(module),'<installed-stream-python-wrappers>','exec'),env);API=env['PythonWrapperOnly']
rows=[]
for producer_tag in ['empty_alt','q_work_enqueued','mla_work_enqueued']:
 trace=[]
 class EventSpy:
  def wait(self,receiver):trace.append(['event.wait','main',producer_tag])
 class AltStreamSpy:
  def record_event(self):trace.append(['alt.record_event',producer_tag]);return EventSpy()
 receiver=API();alt=AltStreamSpy()
 try:receiver.wait_event(alt)
 except AttributeError as exc:original={'error_type':type(exc).__name__,'message':str(exc)}
 else:raise AssertionError('Stream-like object without .wait unexpectedly accepted')
 assert trace==[]
 receiver.wait_stream(alt);assert trace==[['alt.record_event',producer_tag],['event.wait','main',producer_tag]]
 rows.append({'alt_fixture':producer_tag,'original_wait_event_stream_argument':original,'candidate_wait_stream_trace':trace,'native_event_created':False})
# Ordinary wait_event(Event) remains valid in the same Python wrapper contract.
trace=[];receiver.wait_event(EventSpy());assert trace==[['event.wait','main','mla_work_enqueued']]
relative='python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py';src=R/'sources/installed'/relative;before=src.read_text();old='torch.npu.current_stream().wait_event(m.alt_stream)';new='torch.npu.current_stream().wait_stream(m.alt_stream)';assert before.count(old)==1;after=before.replace(old,new);ast.parse(after)
candidate=R/'candidates/L009';candidate.mkdir(parents=True,exist_ok=True);patch=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+relative,tofile='b/'+relative));(candidate/'mlapo_wait_stream.patch').write_text(patch)
result={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'correctness':'PASS','source_commit':'ce0a8fea5f4245839d174b2b8c55692be1810a7f','package_version':'2.10.0.post6','source_hashes':{'torch_npu/npu/streams.py':hashlib.sha256(package.read_bytes()).hexdigest(),relative:hashlib.sha256(src.read_bytes()).hexdigest()},'wrapper_methods':{n:{'line':methods[n].lineno,'end_line':methods[n].end_lineno,'ast_sha256':hashlib.sha256(ast.dump(methods[n],include_attributes=False).encode()).hexdigest()} for n in ['wait_event','wait_stream']},'cases':rows,'positive_event_control':True,'candidate':{'path':str(candidate/'mlapo_wait_stream.patch'),'sha256':hashlib.sha256(patch.encode()).hexdigest(),'patched_python_parses':True,'production_source_unchanged':hashlib.sha256(src.read_bytes()).hexdigest()==hashlib.sha256(before.encode()).hexdigest(),'status':'NOT_APPLIED / NATIVE_VALIDATION_PENDING'},'scope':'Only installed Python method bodies compiled on object base; Event/Stream fixtures implement documented minimal interfaces. This reproduces a conditional Python contract mismatch, not a native Stream instance failure. Native import in isolated R004 was UNAVAILABLE due missing libascend_hal.so. No NPU event ordering, buffer lifetime, graph correctness, numerical result or speed measured. Candidate is retained for review only, not accepted for production. Overall serving E0.'}
(O/'result.json').write_text(json.dumps(result,indent=2)+'\n');(O/'receipt.json').write_text(json.dumps({'correctness':'PASS','exit_code':0,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'result_sha256':hashlib.sha256((O/'result.json').read_bytes()).hexdigest(),'cases':len(rows),'scope':result['scope']},indent=2)+'\n');print(json.dumps({'cases':len(rows),'positive_event_control':True,'candidate_applied':False}))
