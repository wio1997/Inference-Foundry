"""CPU same-state audit/count/mapping equivalence and safety guard gate."""
import ast,json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
root=Path('/data/wio/Inference_Foundry')
sys.path.insert(0,str(root/'scripts'))
import loop081_deferred_audit_patch_run677 as edit
original=(root/'evidence/20260929_loop081_bound/run677/preflight/meta_patch_state/handoff.orig').read_text()
candidate=(root/'bootstrap/vllm_dspark_handoff.py').read_text();assert candidate==edit.patch(original)
def cls(text):
 tree=ast.parse(text);node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DirectDSparkHandoff')
 node.body=[n for n in node.body if isinstance(n,ast.FunctionDef) and n.name in ('_refresh_draft_context_slots','_flush_slot_refresh_audit')]
 scope={'torch':torch,'FixedDecodeState':SimpleNamespace}
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'audit_gate','exec'),scope)
 return scope['DirectDSparkHandoff']
Old,New=cls(original),cls(candidate)
def make(kind,on=False):
 obj=kind.__new__(kind);obj.config=SimpleNamespace(batch_size=12);obj._refresh_draft_slots=True
 obj.slot_refresh_audit=[];obj._pending_slot_refresh_audit=[];obj._defer_slot_audit=on
 obj._group_slot_bindings=[(gid,(torch.arange(12*128).view(12,128)%11).clone(),torch.zeros(96,dtype=torch.int64),bs) for gid,bs in ((2,2),(3,8))]
 return obj
old,off,on=make(Old),make(New),make(New,True)
for cycle in range(10):
 state=SimpleNamespace(cycle_index=cycle,target_positions=(torch.arange(96)+cycle*8).clone())
 for obj in (old,off,on):obj._refresh_draft_context_slots(state)
 for a,b,c in zip(old._group_slot_bindings,off._group_slot_bindings,on._group_slot_bindings):assert torch.equal(a[2],b[2]) and torch.equal(a[2],c[2])
 assert len(on._pending_slot_refresh_audit)==min(cycle+1,8)*2
assert old.slot_refresh_audit==off.slot_refresh_audit and len(old.slot_refresh_audit)==16
assert on.slot_refresh_audit==[]
# Results must not alias mutable mapping, table or state.
for _,table,mapping,_ in on._group_slot_bindings:table.zero_();mapping.fill_(-99)
on._flush_slot_refresh_audit();assert on.slot_refresh_audit==old.slot_refresh_audit and not on._pending_slot_refresh_audit
on._flush_slot_refresh_audit();assert on.slot_refresh_audit==old.slot_refresh_audit
for issue in ('negative_position','large_position','negative_block'):
 for kind,mode in ((Old,False),(New,False),(New,True)):
  obj=make(kind,mode);pos=torch.arange(96)
  if issue=='negative_position':pos[0]=-1
  if issue=='large_position':pos[0]=100000
  if issue=='negative_block':obj._group_slot_bindings[0][1].fill_(-1)
  try:obj._refresh_draft_context_slots(SimpleNamespace(cycle_index=0,target_positions=pos))
  except RuntimeError:pass
  else:raise AssertionError(issue)
  assert torch.count_nonzero(obj._group_slot_bindings[0][2])==0
# Preserve guard AST nodes exactly, including their location before the gather.
def safety(text):
 tree=ast.parse(text);fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_refresh_draft_context_slots')
 return [ast.dump(n,include_attributes=False) for n in ast.walk(fn) if isinstance(n,ast.If) and any(isinstance(x,ast.Raise) and 'draft group' in ast.unparse(x) for x in ast.walk(n))]
assert safety(original)==safety(candidate)
print(json.dumps({'pass':True,'audit_rows_exact':16,'mapping_exact_cycles':10,'snapshot_mutation_isolated':True,'flush_idempotent':True,'immediate_safety_checks_exact':True,'negative_cases':9,'scope':'CPU same-state arithmetic; live all8 correctness and full Product still required'}))
