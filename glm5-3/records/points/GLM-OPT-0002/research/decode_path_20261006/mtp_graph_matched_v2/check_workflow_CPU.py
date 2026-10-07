"""Execute actual comparison/finalization AST with CPU-only lifecycle doubles."""
import ast
from copy import deepcopy
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'pd_compare.py'
tree=ast.parse(SOURCE.read_text())
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('decide','effective_decision','transition','recover','workflow')]


def case(failure):
    with TemporaryDirectory() as temporary:
        root=Path(temporary);parent=root/'diagnostic';parent.mkdir()
        owner=root/'owner.json';owner.write_text(json.dumps(dict(run_id='CPU',status='running',owner={})))
        for name,row in [('state.json',dict(status='completed')),('correctness_reduced.json',dict(passed=True)),('retained_witness.json',dict(H11_rows=[dict(transition=3) for _ in range(16)]))]:
            (parent/name).write_text(json.dumps(row))
        writes={};requests=[];calls=[];state=dict(mode=0,last=0,transition=3,epoch='candidate',injected=False)
        def write(name,row):writes[name]=deepcopy(row)
        def path(value):return owner if str(value)=='/data/tiankuan/wio/glm52-pd/controller-owner.json' else Path(value)
        def rpc(host,action,*values):
            calls.append((host,action,values))
            if action=='epoch':return dict(epoch=state['epoch'])
            if action=='switch':state['mode']=values[0];return {}
            if action=='launch':state['epoch']=values[0];return {}
            terminal='retained_mode_warm' in requests
            if terminal and not state['injected'] and failure in ('witness','guard') and action==('witness' if failure=='witness' else 'guardfresh'):
                state['injected']=True;raise RuntimeError('injected terminal '+failure)
            if action=='witness':return dict(H11_rows=[dict(transition=state['transition']) for _ in range(16)])
            return dict(health=200,idle=True)
        def measured(label,complete=False):
            requests.append(label)
            if label=='retained_mode_warm' and failure=='warm' and not state['injected']:
                state['injected']=True;raise RuntimeError('injected terminal warm')
            if label=='H6_H5_recovery_warm' and failure=='recovery':raise RuntimeError('injected recovery warm')
            if label=='retained_mode_warm' and failure=='recovery' and not state['injected']:
                state['injected']=True;raise RuntimeError('injected terminal warm before recovery')
            if state['mode']!=state['last']:
                state['last']=state['mode'];state['transition']+=1
            if label=='B1_complete' and failure=='measurement':raise RuntimeError('injected measurement')
            b=label.startswith('B')
            return dict(D_wall_s=.8 if b else .9,PD_wall_s=.9 if b else 1.0,P_wall_s=.1,TPOT_ms=35 if b else 40,
                workload_signature=dict(drafts=11,accepted=11),
                cache_by_endpoint=dict(P=dict(hbm_queries=58,hbm_hits=0,ext_queries=58,ext_hits=0),D=dict(hbm_queries=58,hbm_hits=0,ext_queries=58,ext_hits=58)))
        d=NS(rpc=rpc,ready=lambda:None,x=NS(),oldclient=NS(),PLAN={})
        m=NS(write=write,live=lambda owner:True)
        env=dict(Path=path,DIAGNOSTIC=parent,ROOT=root,PLAN=dict(run_id='CPU',initial_transition=3),json=json,math=math,PODS=['P','D'],
                 d=d,m=m,unused_recovery=lambda:None,measured_request=measured)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),env)
        raised=None
        try:env['workflow']()
        except RuntimeError as error:raised=str(error)
        final=writes['decision.json'];recoveries=sum(action=='recoverycleanup' for _,action,_ in calls)
        if failure is None:
            assert final['H11_research_stack'] and final['verdict']=='POSITIVE_LIMITED_RESEARCH'
            assert raised is None and len(requests)==10 and recoveries==0
            assert writes['retained_stack.json']['H11']
        else:
            assert final['verdict']=='INVALIDATED' and not final['H11_research_stack'] and raised
            assert recoveries<=1 and len(requests)<=11
            if failure in ('warm','witness','guard','recovery'):assert recoveries==1
            if failure=='measurement':assert recoveries==0
            if failure!='recovery':assert not writes['retained_stack.json']['H11']
        if failure!='measurement':assert writes['comparison_evidence.json']['H11_research_stack']
        assert d.x.ROOT==root and d.x.PLAN['run_id']=='CPU'
        return dict(failure=failure,final_verdict=final['verdict'],requests=len(requests),recoveries=recoveries)


def main():
    cases=[case(x) for x in (None,'warm','witness','guard','measurement','recovery')]
    row=dict(passed=True,CPU_only=True,HTTP_NPU_model_requests=0,actual_workflow_AST=True,
        actual_recovery_AST=True,positive_published_after_terminal=True,terminal_failures_invalidate=True,
        bounded_single_recovery=True,new_warm_transitions=True,request_roots_routed=True,cases=cases,
        limitations='Lifecycle and request doubles, not runtime/performance proof.',
        source_sha256=__import__('hashlib').sha256(SOURCE.read_bytes()).hexdigest())
    (HERE/'workflow_CPU_result.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row))


if __name__=='__main__':main()
