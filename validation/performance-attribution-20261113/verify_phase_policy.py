"""只核对源码插入位置及主机状态契约；不导入SDK，不运行调度器或模型。"""
from pathlib import Path
from types import SimpleNamespace as NS
import ast
import hashlib
import json
import tempfile
from phase_policy import PhasePolicy,transformed_method


def verified_source(source):
    transformed=transformed_method(source)
    tree=ast.parse(source);cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Scheduler')
    original=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='schedule')
    modified=ast.parse(transformed).body[0]
    gate_lines=[]
    for node in ast.walk(modified):
        if isinstance(node,ast.If) and isinstance(node.test,ast.UnaryOp) and isinstance(node.test.operand,ast.Call):
            call=node.test.operand
            if isinstance(call.func,ast.Attribute) and call.func.attr=='allows':gate_lines.append(node.lineno)
    assert len(gate_lines)==2
    allocations=[n.lineno for n in ast.walk(modified) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='allocate_slots']
    assert len(allocations)==2 and all(a<b for a,b in zip(sorted(gate_lines),sorted(allocations)))
    record=next(n.lineno for n in ast.walk(modified) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='record' and isinstance(n.func.value,ast.Name) and n.func.value.id=='_ap')
    advance=next(n.lineno for n in ast.walk(modified) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='_update_after_schedule')
    assert record<advance

    def private_name(node):return isinstance(node,ast.Name) and node.id.startswith('_ap')
    class RemoveInserted(ast.NodeTransformer):
        def visit_Assign(self,node):
            if any(private_name(t) for t in node.targets):return None
            if any(isinstance(t,ast.Subscript) and private_name(t.value) for t in node.targets):return None
            return self.generic_visit(node)
        def visit_If(self,node):
            if any(private_name(x) for x in ast.walk(node.test)):return None
            return self.generic_visit(node)
        def visit_Expr(self,node):
            if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and private_name(node.value.func.value):return None
            return self.generic_visit(node)
    restored=RemoveInserted().visit(modified)
    assert ast.dump(restored,include_attributes=False)==ast.dump(original,include_attributes=False)
    try:transformed_method(source+'\n')
    except ValueError:pass
    else:raise AssertionError('unmatched SDK source accepted')
    return {'allocation_gates':len(gate_lines),'original_method_AST_restored_exactly':True,
            'metadata_record_before_async_progress':True,'source_drift_rejected':True,
            'transformed_method_sha256':hashlib.sha256(transformed.encode()).hexdigest()}


def metadata_contracts():
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
        on=PhasePolicy(True,'fixed-source',Path(directory)/'on.json')
        off=PhasePolicy(False,'fixed-source',Path(directory)/'off.json')
        prefill=NS(request_id='prefix-hit',has_encoder_inputs=False,num_preemptions=0,
            num_computed_tokens=640,num_tokens=1024,num_output_placeholders=0,num_prompt_tokens=1024,
            spec_token_ids=[],is_prefill_chunk=True)
        decode=NS(request_id='async-decode',has_encoder_inputs=False,num_preemptions=0,
            num_computed_tokens=1030,num_tokens=1025,num_output_placeholders=6,num_prompt_tokens=1024,
            spec_token_ids=[-1]*5,is_prefill_chunk=False)
        before=[dict(vars(q)) for q in (prefill,decode)]
        assert on.phase(prefill,640,384)=='prefill'
        assert on.phase(decode,1030,6)=='spec_decode'
        assert on.phase(prefill,1023,6,True)=='spec_decode'
        assert not on.allows('prefill','spec_decode') and off.allows('prefill','spec_decode')
        output=NS(num_scheduled_tokens={'prefix-hit':384,'async-decode':6},
            total_num_scheduled_tokens=390,scheduled_spec_decode_tokens={'async-decode':[-1]*5},preempted_req_ids=set())
        scheduler=NS(requests={q.request_id:q for q in (prefill,decode)})
        tags={'prefix-hit':'prefill','async-decode':'spec_decode'}
        off.record(scheduler,output,tags,off.begin())
        assert off.counts['mixed_contract_steps']==1
        try:on.record(scheduler,output,tags,on.begin())
        except RuntimeError:pass
        else:raise AssertionError('enabled mixed output not rejected before progress')
        assert [dict(vars(q)) for q in (prefill,decode)]==before
        class DeviceLike:
            def __int__(self):raise AssertionError('device read forbidden')
        try:on.phase(prefill,DeviceLike(),384)
        except TypeError:pass
        else:raise AssertionError('nonhost scalar accepted')
        off.close();off.close()
        snapshot=json.loads((Path(directory)/'off.json').read_text())
        assert snapshot['closed'] and snapshot['events'][0]['total_scheduled_tokens']==390
        assert [dict(vars(q)) for q in (prefill,decode)]==before
    return {'prefix_hit_and_async_placeholder_classification':True,'padded_waiting_spec_handled':True,
            'off_preserves_mixed_metadata':True,'on_rejects_mixed_metadata_before_progress':True,
            'request_state_unmodified_by_policy_observation':True,'nonhost_scalar_rejected_without_conversion':True,
            'shutdown_journal_idempotent':True,'actual_scheduler_executed':False}


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    code=(root/'fixed-sdk/scheduler.py').read_text()
    result={'source':verified_source(code),'host_metadata':metadata_contracts(),
            'scope':'static original-method preservation and host metadata contracts;no actual SDK schedule or model execution',
            'new_npu_runs':0,'actual_sdk_runtime_validated':False,'goal_complete':False}
    (root/'phase-policy-source-checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))
