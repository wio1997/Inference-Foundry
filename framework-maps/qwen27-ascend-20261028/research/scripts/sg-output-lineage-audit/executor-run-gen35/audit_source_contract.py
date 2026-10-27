"""Pin and check the saved installed scheduler's step-preserving handoff."""
from pathlib import Path
import ast,json,hashlib
D=Path(__file__).resolve().parent;R=D.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt=json.loads((D/'installed-source/download.json').read_text());facts=[]
for row in receipt['files']:
 p=D/'installed-source'/row['relative_path'];assert sha(p)==row['sha256']==row['frozen_git_sha256']and row['installed_equals_frozen_git']

def function(name,file):
 p=D/'installed-source'/file;nodes=[n for n in ast.walk(ast.parse(p.read_text()))if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))and n.name==name];assert len(nodes)==1
 return nodes[0]
def assignments(fn,target,value):
 return [n for n in ast.walk(fn)if isinstance(n,ast.Assign)and any(ast.unparse(x)==target for x in n.targets)and ast.unparse(n.value)==value]
def calls(fn,expr):return[n for n in ast.walk(fn)if isinstance(n,ast.Call)and ast.unparse(n)==expr]
a=function('run_batch','scheduler.py');v=assignments(a,'batch.forward_iter','self.forward_ct');assert len(v)==1;facts.append({'file':'scheduler.py','lines':[v[0].lineno],'fact':'run_batch explicitly assigns batch.forward_iter = self.forward_ct'})
a=function('event_loop_overlap','scheduler.py')
for expr in ['self.result_queue.append((batch.copy(), batch_result))','self.process_batch_result(tmp_batch, tmp_result)']:
 v=calls(a,expr);assert len(v)==1;facts.append({'file':'scheduler.py','lines':[v[0].lineno],'fact':expr})
a=function('copy','schedule_batch.py');v=[n for n in ast.walk(a)if isinstance(n,ast.Call)and ast.unparse(n.func)=='ScheduleBatch'];assert len(v)==1
kw={k.arg:k.value for k in v[0].keywords}
for name in ['forward_iter','forward_mode']:assert ast.unparse(kw[name])=='self.'+name
facts.append({'file':'schedule_batch.py','lines':[kw['forward_iter'].lineno,kw['forward_mode'].lineno],'fact':'ScheduleBatch.copy preserves explicit forward_iter and forward_mode in returned scheduler-result batch'})
a=function('process_batch_result','scheduler.py');v=calls(a,'self.batch_result_processor.process_batch_result_decode(batch, result)');assert len(v)==1;facts.append({'file':'scheduler.py','lines':[v[0].lineno],'fact':'decode processing receives the same batch and result arguments'})
# Byte-for-byte historical frozen source corroboration for token forwarding.
p=D/'installed-source/detokenizer_manager.py';frozen=R/'final-assembly/source-path-normalization/upstream/sgl-project_sglang/python/sglang/srt/managers/detokenizer_manager.py';assert sha(p)==sha(frozen)
a=function('handle_batch_token_id_out','detokenizer_manager.py');v=[n for n in ast.walk(a)if isinstance(n,ast.Call)and ast.unparse(n.func)=='BatchStrOutput'];assert len(v)==1
kw={k.arg:k.value for k in v[0].keywords}
assert ast.unparse(kw['output_ids'])=='recv_obj.output_ids'and ast.unparse(kw['rids'])=='recv_obj.rids'
facts.append({'file':'detokenizer_manager.py','lines':[kw['output_ids'].lineno,kw['rids'].lineno],'fact':'BatchStrOutput forwards the received exact output_ids and rids'})
x={'task_id':'Q27-FULL-MAP-20261025','generation':35,'component_commit':receipt['source']['component_commit'],'scope':'source-level contract; current installed bytes equal historical pinned commit, same historical container/image; not proof of complete model numerical equivalence','source':receipt['source'],'files':[{'path':str((D/'installed-source'/row['relative_path']).relative_to(R)),'sha256':row['sha256']}for row in receipt['files']],'facts':facts,'source_receipt_sha256':sha(D/'installed-source/download.json'),'source_sha256':sha(Path(__file__))}
(D/'source-contract.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');print('Three installed source files equal pinned commit; six step/token forwarding facts checked')
