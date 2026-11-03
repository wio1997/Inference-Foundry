"""Bounded source supplement for saved graph profile identity/lifetime questions."""
import ast
from datetime import datetime, timezone
import hashlib

from finalize_origin_coverage_v9 import O, W, read, write


def main():
    assert not(O/'REVIEW-CHECKPOINT-v44.json').exists()
    old=read(O/'profile-raised-source-contract-v16.json')
    entry=next(f for f in old['excerpts']if f['path'].endswith('/compilation/acl_graph.py'))
    p=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend/compilation/acl_graph.py'
    b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==entry['sha256']
    tree=ast.parse(b);lines=b.decode().splitlines()
    write(O/'PROFILE-CODE-QUESTION-GATE-v44.json',dict(
        status='PROFILE_FIRST_M22_CAPTURE_REPLAY_LIFETIME',previous_turn_classification='PROGRESS:v43neighbor63guardreconciliation',
        profile_facts=['Current80graph executions with64phase-associated/16unbound; no input bank addresses in compact current receipt.',
                       'Saved effective main capture sizes[8,16,24],copy_inputs=False; draft outer wrapper excluded separately.',
                       'Saved worker event profiles do not establish event-handle or captured-buffer generations.'],
        questions=['Which captured inputs/outputs and owner/address checks are retained, and what first-capture vs replay mutation is observed by optimization candidates?',
                   'Which update/replay synchronization branches can be ruled in/out by the actual saved graph options, and which exact predicate definition is needed?',
                   'How much of original M22 source scope is genuinely analyzed versus wrapper-only or adjacent-worker navigation?'],
        allowed_new_source=['Same-SHA ACLGraphEntry fields; wrapper selected constructor assignments only.',
                            'Same-SHA __call__ missing capture blocks175-185/189-207/215-244/245-264 only; runtime dispatch/replay ranges already in v16 are reused.',
                            'Same-SHA _updatable_graph_replay and use_updatable_graph small methods only.'],
        no_repository_wide_source_audit=True,no_model_torch_NPU_service_health_work=True,whole_module_closed=False,overall_complete=False))
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef)and x.name=='ACLGraphWrapper')
    data=next(x for x in tree.body if isinstance(x,ast.ClassDef)and x.name=='ACLGraphEntry')
    assert data.end_lineno-data.lineno+1<=20
    ranges=[(data.lineno,data.end_lineno),(175,185),(189,207),(215,244),(245,264)]
    pieces=[dict(start=a,end=z,source=[f'{i+1}: {lines[i]}'for i in range(a-1,z)])for a,z in ranges]
    functions=[]
    for scope,name in [(cls.body,'_updatable_graph_replay')]:
        node=next(x for x in scope if isinstance(x,ast.FunctionDef)and x.name==name)
        assert node.end_lineno-node.lineno+1<=40
        functions.append(dict(name=name,start=node.lineno,end=node.end_lineno,
            source=[f'{i+1}: {lines[i]}'for i in range(node.lineno-1,node.end_lineno)]))
    init=next(x for x in cls.body if isinstance(x,ast.FunctionDef)and x.name=='__init__')
    fields={'runtime_mode','is_debugging_mode','use_eagle','attn_backend','enable_enpu','aclgraph_options','vllm_config'}
    selected=[]
    for node in ast.walk(init):
        if isinstance(node,(ast.Assign,ast.AnnAssign)):
            targets=node.targets if isinstance(node,ast.Assign)else[node.target]
            if any(isinstance(t,ast.Attribute)and isinstance(t.value,ast.Name)and t.value.id=='self'and t.attr in fields for t in targets):
                assert node.end_lineno-node.lineno+1<=10
                selected.append(dict(start=node.lineno,end=node.end_lineno,source=[f'{i+1}: {lines[i]}'for i in range(node.lineno-1,node.end_lineno)]))
    predicate_import=next(x for x in tree.body if isinstance(x,ast.ImportFrom)and any(a.name=='use_updatable_graph'for a in x.names))
    assert predicate_import.level==2 and predicate_import.module=='utils'
    predicate_path=p.parents[1]/'utils.py'
    predicate=dict(import_module='..utils',name='use_updatable_graph',local_file=str(predicate_path),local_file_available=predicate_path.exists())
    if predicate_path.exists():
        pb=predicate_path.read_bytes();pt=ast.parse(pb);plines=pb.decode().splitlines()
        node=next(x for x in pt.body if isinstance(x,ast.FunctionDef)and x.name=='use_updatable_graph')
        assert node.end_lineno-node.lineno+1<=40
        predicate.update(sha256=hashlib.sha256(pb).hexdigest(),start=node.lineno,end=node.end_lineno,
                         source=[f'{i+1}: {plines[i]}'for i in range(node.lineno-1,node.end_lineno)],actual166_hash_attested_this_turn=False)
    gate=read(O/'PROFILE-CODE-QUESTION-GATE-v44.json');gate['resolved_predicate_import']='..utils:use_updatable_graph; exact small predicate only if local source exists';write(O/'PROFILE-CODE-QUESTION-GATE-v44.json',gate)
    write(O/'M22-capture-lifetime-point-source-v44.json',dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),path=str(p),sha256=entry['sha256'],
        ranges=pieces,functions=functions,constructor_statements=selected,update_predicate=predicate,
        reused_v16_ranges=entry['ranges'],no_model_or_torch_import=True,
        scope='Missing capture blocks/entry fields/selected ctor/small replay helper only. No full worker/capture file audit.'))
    for x in functions+selected+pieces:
        print('\n'.join(x['source']))
    if 'source'in predicate:print('\n'.join(predicate['source']))


if __name__=='__main__':main()
