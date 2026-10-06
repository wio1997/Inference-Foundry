"""H5 policy/None contract and whole-source AST equivalence outside events."""
import ast
import copy
import hashlib
import json
import types
from pathlib import Path


class UndoEvents(ast.NodeTransformer):
    def __init__(self, shared=False):
        self.shared=shared

    def visit_ImportFrom(self,node):
        if node.module=='vllm_ascend.utils':
            node.names=[x for x in node.names if x.name!='maybe_record_moe_event']
            if not node.names:return None
        return node

    def visit_FunctionDef(self,node):
        if node.name=='maybe_record_moe_event':return None
        if self.shared and node.name=='forward':
            assert isinstance(node.body[0],ast.If) and ast.unparse(node.body[0].test)=='self.multistream_overlap'
            node.body.pop(0)
        return self.generic_visit(node)

    def visit_AnnAssign(self,node):
        if self.shared and isinstance(node.target,ast.Name) and node.target.id=='before_routed_experts':
            node.annotation=ast.parse('torch.npu.Event',mode='eval').body
        return self.generic_visit(node)

    def visit_Call(self,node):
        if isinstance(node.func,ast.Name) and node.func.id=='maybe_record_moe_event':
            assert not node.args and not node.keywords
            return ast.parse('torch.npu.current_stream().record_event()',mode='eval').body
        if self.shared and isinstance(node.func,ast.Name) and node.func.id=='maybe_wait_event' and node.lineno in (309,370,401):
            # Only the three mandatory waits changed. The two pre-existing
            # optional guards in DP/SP paths keep their original structure.
            return ast.parse('torch.npu.current_stream().wait_event(fused_moe_evts.before_routed_experts)',mode='eval').body
        return self.generic_visit(node)


def check(root):
    identities=json.loads((root/'moe_event_patch_identity.json').read_text())
    rows=[]
    for name,x in identities.items():
        old=(root/'event_originals'/name).read_bytes();new=(root/'event_candidate'/name).read_bytes()
        assert hashlib.sha256(old).hexdigest()==x['original_sha256']
        assert hashlib.sha256(new).hexdigest()==x['candidate_sha256']
        a=ast.parse(old.decode());b=UndoEvents(name.endswith('/shared_experts.py')).visit(ast.parse(new.decode()))
        assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False),name
        rows.append(dict(file=name,non_event_AST_identical=True))
    tree=ast.parse((root/'event_candidate/utils.py').read_text())
    helper=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='maybe_record_moe_event')
    shared=ast.parse((root/'event_candidate/ops/fused_moe/shared_experts.py').read_text())
    cls=next(x for x in shared.body if isinstance(x,ast.ClassDef) and x.name=='AscendSharedExperts')
    forward=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='forward')
    wait=next(x for x in forward.body if isinstance(x,ast.FunctionDef) and x.name=='maybe_wait_event')
    calls=[];sentinel=object();config=types.SimpleNamespace(multistream_overlap_shared_expert=False)
    stream=types.SimpleNamespace(record_event=lambda:(calls.append('record') or sentinel),wait_event=lambda e:calls.append(('wait',e)))
    torch=types.SimpleNamespace(npu=types.SimpleNamespace(current_stream=lambda:(calls.append('current_stream') or stream)))
    ns=dict(torch=torch,get_ascend_config=lambda:config)
    nodes=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),copy.deepcopy(helper),copy.deepcopy(wait)]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'candidate_event_contract','exec'),ns)
    for enabled in (False,True):
        config.multistream_overlap_shared_expert=enabled;calls.clear()
        event=ns['maybe_record_moe_event']();ns['maybe_wait_event'](event)
        if enabled:assert event is sentinel and calls==['current_stream','record','current_stream',('wait',sentinel)]
        else:assert event is None and calls==[]
    guard=copy.deepcopy(forward.body[0]);assert isinstance(guard,ast.If)
    code=compile(ast.fix_missing_locations(ast.Module(body=[guard],type_ignores=[])),'consumer_guard','exec')
    for enabled,event,should_fail in [(False,None,False),(True,sentinel,False),(True,None,True)]:
        ns=dict(self=types.SimpleNamespace(multistream_overlap=enabled),fused_moe_evts=types.SimpleNamespace(before_routed_experts=event))
        failed=False
        try:exec(code,ns)
        except AssertionError:failed=True
        assert failed==should_fail
    return dict(passed=True,source_checks=rows,policy_false_no_stream_or_event_call=True,
                policy_true_original_record_wait=True,none_consumer_safe=True,
                true_consumer_missing_event_rejected=True,device_or_model_loaded=False)


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    out=check(root);(root/'moe_event_cpu_correctness.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
