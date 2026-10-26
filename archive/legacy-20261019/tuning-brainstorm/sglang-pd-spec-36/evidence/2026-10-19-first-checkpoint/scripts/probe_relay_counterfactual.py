#!/usr/bin/env python3
"""CPU state differential of exact installed publish/resolve methods. No NPU timing."""
import ast,datetime,enum,hashlib,itertools,json,pathlib,platform,sys,types
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'runs/L002-R003-relay-counterfactual';OUT.mkdir(parents=True,exist_ok=False)
src=ROOT/'sources/installed/python/sglang/srt/managers/overlap_utils.py';b=src.read_bytes();tree=ast.parse(b)
fm_node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FutureMap')
env={'torch':torch,'_DEBUG_ASSERT':False,'_is_hip':False}
mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fm_node],type_ignores=[])
exec(compile(ast.fix_missing_locations(mod),str(src),'exec'),env);FutureMap=env['FutureMap']
class EventFixture:
 def __init__(self):self.record_count=0;self.wait_count=0
 def record(self):self.record_count+=1
 def wait(self):self.wait_count+=1
algo=types.SimpleNamespace(is_some=lambda:True)
def new_map():
 f=FutureMap.__new__(FutureMap);f.device='cpu';f.spec_algo=algo;f.req_pool_size=8
 f.new_seq_lens_buf=torch.full((8,),-777,dtype=torch.int64);f.needs_confidence_relay=False
 f.needs_cpu_seq_lens=True;f.publish_ready=EventFixture();f._publish_fresh=False;f.fwd_prepare_d2h_stream=None
 return f
def resolve(f,slots,mirror):
 f.needs_cpu_seq_lens=mirror
 x=types.SimpleNamespace(spec_info=types.SimpleNamespace(future_indices=torch.tensor(slots,dtype=torch.int64)),seq_lens=None,seq_lens_cpu=None,seq_lens_sum=None)
 f.resolve_seq_lens_cpu(x);return x
rows=[];negative=[]
for mirror in [False,True]:
 for perm in itertools.permutations([1,2,3]):
  for deltas in itertools.product([1,2,4],repeat=3):
   f=new_map();old={1:100,2:200,3:300};want={s:old[s]+deltas[s-1] for s in old}
   f.publish(torch.tensor([1,2,3]),torch.tensor([want[s] for s in [1,2,3]],dtype=torch.int32))
   x=resolve(f,perm,mirror);expected=[want[s] for s in perm]
   assert x.seq_lens.tolist()==expected
   if mirror:assert x.seq_lens_cpu.tolist()==expected and x.seq_lens_sum==sum(expected)
   else:assert x.seq_lens_cpu is None and x.seq_lens_sum is None
   assert (f.publish_ready.record_count,f.publish_ready.wait_count)==(1,1)
   # Request 2 is cancelled; survivors re-order, then the same slot is assigned to a new request.
   survivor=resolve(f,[3,1],mirror);assert survivor.seq_lens.tolist()==[want[3],want[1]]
   f.publish(torch.tensor([2]),torch.tensor([17],dtype=torch.int64))
   x=resolve(f,[2,3,1],mirror);assert x.seq_lens.tolist()==[17,want[3],want[1]]
   # Empty DP batch must not record a publish event or touch the buffers.
   snapshot=f.new_seq_lens_buf.clone();recorded=f.publish_ready.record_count
   f.publish(torch.tensor([],dtype=torch.int64),torch.tensor([],dtype=torch.int64))
   assert torch.equal(snapshot,f.new_seq_lens_buf) and recorded==f.publish_ready.record_count
   rows.append({'mirror':mirror,'permutation':list(perm),'effective_length_increments':list(deltas),'post_reuse':[17,want[3],want[1]]})
   if any(d!=1 for d in deltas):
    bad=new_map();bad.publish(torch.tensor([1,2,3]),torch.tensor([old[s]+1 for s in [1,2,3]]));wrong=resolve(bad,perm,mirror).seq_lens.tolist();assert wrong!=expected
    negative.append({'mutation':'replace true lengths with old+1','expected':expected,'wrong':wrong,'detected':True})
   stale=new_map();stale.publish(torch.tensor([1,2,3]),torch.tensor([want[s] for s in [1,2,3]]));wrong_stale=resolve(stale,[2,3,1],mirror).seq_lens.tolist()
   assert wrong_stale!=[17,want[3],want[1]]
   negative.append({'mutation':'omit new request slot publication','stale_slot_2':want[2],'expected_slot_2':17,'detected':True})
assert len(rows)==324
result={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':'ce0a8fea5f4245839d174b2b8c55692be1810a7f','source_sha256':hashlib.sha256(b).hexdigest(),'methods':{n.name:{'line':n.lineno,'end_line':n.end_lineno} for n in fm_node.body if isinstance(n,ast.FunctionDef) and n.name in ['publish','resolve_seq_lens_cpu']},'python':sys.version,'torch':torch.__version__,'platform':platform.platform(),'device':'cpu','cases':len(rows),'negative_controls_detected':len(negative),'correctness':'PASS','exit_code':0,'scope':'Exact publish/resolve method bodies. __init__ not executed; synchronous CPU fixture supplies tensors and fake event. This checks value/index/dtype/CPU-mirror semantics only, not NPU async ordering, graph execution, full cancellation/EOS, bonus/hidden/topk stash or end-to-end correctness. No latency, throughput or speedup claim.','rows':rows,'negative_controls':negative}
(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
(OUT/'receipt.json').write_text(json.dumps({'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(b).hexdigest(),'result_sha256':hashlib.sha256((OUT/'result.json').read_bytes()).hexdigest(),'command':[sys.executable,str(pathlib.Path(__file__).resolve())],'exit_code':0,'correctness':'PASS','performance_claim':False},indent=2)+'\n')
print(json.dumps({'cases':len(rows),'negative_controls_detected':len(negative),'correctness':'PASS'}))
