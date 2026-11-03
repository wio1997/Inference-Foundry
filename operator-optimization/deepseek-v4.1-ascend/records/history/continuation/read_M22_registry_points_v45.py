"""Read only registry/workspace/proxy cases raised by current graph/bank evidence."""
import ast
from datetime import datetime,timezone
import hashlib
from finalize_origin_coverage_v9 import O,W,read,write


def main():
    assert not(O/'REVIEW-CHECKPOINT-v45.json').exists()
    p=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend/compilation/acl_graph.py'
    b=p.read_bytes();sha=hashlib.sha256(b).hexdigest();assert sha=='8e1c269e331d005f1361acc5cec24402bff7ae39b75e952ea890da7812804955'
    tree=ast.parse(b);lines=b.decode().splitlines()
    write(O/'PROFILE-CODE-QUESTION-GATE-v45.json',dict(
        status='PROFILE_FIRST_M22_REMAINING28_REGISTRY_WORKSPACE_GUARDS',previous_turn_classification='PROGRESS:v44placement41guardaudit',
        profile_facts=['Current same-capture321SMLA+208QLI named metadata DIRECT placement established; bank/native update identities not established.',
                       '80graph executions/64phase-linked/16unbound retained; current compact receipts do not expose complete registry/workspace/task/handle generations.'],
        question='What namespaces/keys/owner lifetimes and update arguments constrain graph-param/workspace reuse, and which remaining28graph-file guards are already CPU/proxy/diagnostic behavior?',
        allowed_new_source=['Same-SHA13top-level registry/workspace/update/error helpers,GraphParams field declaration,ACLGraphWrapper.__getattr__ only.',
                            'Exact registry globals/error marker constants only; no native SDK or whole runner audit.'],
        new_source_read_allowed=True,no_model_torch_NPU_service_health_work=True,whole_module_closed=False,overall_complete=False))
    names=['_is_stream_resource_capture_error','_is_old_hdk_capture_error','weak_ref_workspaces','update_full_graph_params',
        'set_graph_params','update_graph_params_workspaces','get_graph_params','set_draft_graph_params',
        'update_draft_graph_params_workspaces','get_draft_graph_params','set_draft_graph_prefill_params',
        'update_draft_graph_prefill_params_workspaces','get_draft_graph_prefill_params']
    selected=[]
    for name in names:
        node=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name==name)
        assert node.end_lineno-node.lineno+1<=40
        selected.append(dict(owner='',name=name,start=node.lineno,end=node.end_lineno,source=[f'{i+1}: {lines[i]}'for i in range(node.lineno-1,node.end_lineno)]))
    for clsname,method in [('ACLGraphWrapper','__getattr__')]:
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef)and x.name==clsname)
        node=next(x for x in cls.body if isinstance(x,ast.FunctionDef)and x.name==method)
        assert node.end_lineno-node.lineno+1<=20
        selected.append(dict(owner=clsname,name=method,start=node.lineno,end=node.end_lineno,source=[f'{i+1}: {lines[i]}'for i in range(node.lineno-1,node.end_lineno)]))
    fields=next(x for x in tree.body if isinstance(x,ast.ClassDef)and x.name=='GraphParams')
    declarations=[dict(start=fields.lineno,end=fields.end_lineno,source=[f'{i+1}: {lines[i]}'for i in range(fields.lineno-1,fields.end_lineno)])]
    keys={'_graph_params','_draft_graph_params','_draft_graph_prefill_params','_STREAM_RESOURCE_ERROR_CODE','_STREAM_RESOURCE_ERROR_MARKERS','_OLD_HDK_CAPTURE_ERROR_MARKERS'}
    for node in tree.body:
        if isinstance(node,(ast.Assign,ast.AnnAssign)):
            targets=node.targets if isinstance(node,ast.Assign)else[node.target]
            if any(isinstance(t,ast.Name)and t.id in keys for t in targets):
                assert node.end_lineno-node.lineno+1<=20
                declarations.append(dict(start=node.lineno,end=node.end_lineno,source=[f'{i+1}: {lines[i]}'for i in range(node.lineno-1,node.end_lineno)]))
    write(O/'M22-registry-workspace-point-source-v45.json',dict(generated_utc=datetime.now(timezone.utc).isoformat(),
        path=str(p),sha256=sha,functions=selected,declarations=declarations,no_model_or_torch_import=True,
        scope='14small functions incl3branch-free getters+GraphParams/global declarations. No native backend update implementation read.'))
    for x in declarations+selected:print('\n'.join(x['source']))


if __name__=='__main__':main()
