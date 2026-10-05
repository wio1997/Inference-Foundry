"""GPT critical source check; model-free CPU arithmetic/control research."""
from pathlib import Path
import argparse,ast,types,json,hashlib,sys
ap=argparse.ArgumentParser();ap.add_argument("--output-dir",required=True,type=Path);args=ap.parse_args()
root=Path(__file__).parent;out=args.output_dir;assert out.is_absolute();out.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(root));from grammar_draft_control import validate_immediate_drafts,CORE_SHA,SCHEDULER_SHA
def ref(f):
 raw=Path(f).read_bytes();return dict(path=str(f),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
assert ref(root/"v1_engine_core.py")["sha256"]==CORE_SHA and ref(root/"v1_core_sched_scheduler.py")["sha256"]==SCHEDULER_SHA
def native_method(file,cls,name):
 tree=ast.parse((root/file).read_text());c=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name==cls)
 f=next(n for n in c.body if isinstance(n,ast.FunctionDef)and n.name==name);f.returns=None
 for a in f.args.args:a.annotation=None
 ns=dict(TYPE_CHECKING=False,GrammarOutput=lambda ids,masks:types.SimpleNamespace(structured_output_request_ids=ids,grammar_bitmask=masks))
 exec(compile(ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[])),"<installed-native-method>",'exec'),ns)
 return ns[name]
mask_fn=native_method("v1_structured_output___init__.py","StructuredOutputManager","grammar_bitmask")
get_fn=native_method("v1_core_sched_scheduler.py","Scheduler","get_grammar_bitmask")
filter_fn=native_method("v1_core_sched_scheduler.py","Scheduler","update_draft_token_ids_in_output")
class Mask:
 def __init__(self,n):self.rows=[None]*n;self.shape=(n,1)
 def __getitem__(self,index):
  m=Mask(0);m.rows=self.rows[index];m.shape=(len(m.rows),1);return m
 def numpy(self):return self.rows
class Grammar:
 pieces={455:"get",709:"ge",68852:"_weather",154847:"<arg_key>"}
 target="get_weather<arg_key>"
 def __init__(self):self.history=[]
 def prefix(self):return "".join(self.pieces[t]for t in self.history)
 def allowed(self):return [t for t,s in self.pieces.items()if self.target.startswith(self.prefix()+s)]
 def is_terminated(self):return self.prefix()==self.target
 def accept_tokens(self,req,tokens):
  for token in tokens:
   if token not in self.allowed():return False
   self.history.append(token)
  return True
 def rollback(self,n):
  if n:del self.history[-n:]
 def validate_tokens(self,tokens):
  valid=[]
  for token in tokens:
   if not self.accept_tokens("CPU", [token]):break
   valid.append(token)
  self.rollback(len(valid));return valid
def fixture(drafts=None,pending=False,structured=True):
 grammar=Grammar();req=types.SimpleNamespace(use_structured_output=structured,is_prefill_chunk=False,is_finished=lambda:False,structured_output_request=types.SimpleNamespace(grammar=grammar))
 manager=types.SimpleNamespace(vllm_config=types.SimpleNamespace(num_speculative_tokens=2,scheduler_config=types.SimpleNamespace(max_num_seqs=2),model_config=types.SimpleNamespace(is_diffusion=False)),_grammar_bitmask=Mask(6),fill_bitmask_parallel_threshold=128,enable_in_reasoning=False,_get_reasoner=lambda r:None,should_fill_bitmask=lambda r:True,should_advance=lambda r:True)
 def fill(batch):
  for gram,index,apply in batch:manager._grammar_bitmask.rows[index]=dict(allow_all=not apply,allowed=gram.allowed()if apply else None)
 manager._fill_bitmasks=fill;manager.grammar_bitmask=types.MethodType(mask_fn,manager)
 scheduler=types.SimpleNamespace(requests={"req":req},structured_output_manager=manager)
 scheduler.update_draft_token_ids_in_output=types.MethodType(filter_fn,scheduler)
 scheduler.get_grammar_bitmask=types.MethodType(get_fn,scheduler)
 calls=[]
 def take():calls.append("native_get");return drafts
 engine=types.SimpleNamespace(scheduler=scheduler,model_executor=types.SimpleNamespace(take_draft_token_ids=take),check_for_draft_tokens=True)
 output=types.SimpleNamespace(pending_structured_output_tokens=pending,scheduled_spec_decode_tokens={"req":[-1,-1]},has_structured_output_requests=True,num_scheduled_tokens={"req":3})
 return engine,output,grammar,calls
drafts=types.SimpleNamespace(req_ids=["req"],draft_token_ids=[[455,68852]])
engine,output,grammar,calls=fixture(drafts)
before=engine.scheduler.get_grammar_bitmask(output).grammar_bitmask
evidence=validate_immediate_drafts(engine,output)
after=engine.scheduler.get_grammar_bitmask(output).grammar_bitmask
assert before[-1]["allowed"]==[455,709]and before[1]["allow_all"]
assert after[-1]["allowed"]==[154847]and not any(m["allow_all"]for m in after)
assert evidence["after"]=={"req":[455,68852]}and output.num_invalid_spec_tokens=={}and calls==["native_get"]and not grammar.history
assert not hasattr(engine,"batch_queue")
checks=["installed native filter+mask AST methods constrain immediate bonus after real valid drafts","original grammar state rollback preserved; helper never touches queue/FIFO/resource/config"]
for pending,structured in [(True,True),(False,False)]:
 e,o,g,c=fixture(drafts,pending,structured);assert validate_immediate_drafts(e,o)is None and not c and o.scheduled_spec_decode_tokens=={"req":[-1,-1]}
checks.append("deferred path and ordinary requests perform no added native draft getter")
for bad in [None,types.SimpleNamespace(req_ids=["other"],draft_token_ids=[[455,68852]]),types.SimpleNamespace(req_ids=["req","req"],draft_token_ids=[[455,68852],[455,68852]]),types.SimpleNamespace(req_ids=["req"],draft_token_ids=[[455,-1]])]:
 e,o,g,c=fixture(bad)
 try:validate_immediate_drafts(e,o)
 except RuntimeError:pass
 else:raise AssertionError("missing/stale/ambiguous draft coverage accepted")
 assert o.scheduled_spec_decode_tokens=={"req":[-1,-1]}and not g.history
checks.append("absent/mismatched/duplicate/invalid draft identity fails before native grammar mask; no silent fallback")
e,o,g,c=fixture(types.SimpleNamespace(req_ids=["req"],draft_token_ids=[[709,68852]]));validate_immediate_drafts(e,o)
assert o.scheduled_spec_decode_tokens=={"req":[709,-1]}and o.num_invalid_spec_tokens=={"req":1}and not g.history
checks.append("native grammar prefix filtering and invalid-draft padding unchanged")
proof=dict(owner="GPT/root critical isolated source check",CPU_ONLY=True,native_imports_or_SDK_or_NPU_calls=0,GPU_inference_requests=0,active_runtime_changed=False,synthetic_lexical_FSM=True,native_AST_methods=["get_grammar_bitmask","grammar_bitmask","update_draft_token_ids_in_output"],checks=checks,placeholder_masks=before,candidate_masks=after,candidate=ref(root/"grammar_draft_control.py"),probe=ref(Path(__file__)),limitations=["Actual Run236 immediate/deferred branch and placeholder/draft vector are not observed; conditional mechanism only","Synthetic lexical FSM models known tokenizer prefix, not actual xgrammar/native GPU replay","PP interleaved cohort draft coverage/freshness and native RPC ordering need bounded real E2E","Candidate not installed; no performance/function KEEP or global capacity claim"])
f=out/"critical_cpu_proof.json";f.write_text(json.dumps(proof,indent=2)+"\n");print(json.dumps(dict(CPU_conditional_VALID=True,checks=len(checks),**ref(f))))
