#!/usr/bin/env python3
"""Execute exact installed predicates, isolated from heavyweight imports. CPU only."""
import ast,datetime,enum,hashlib,itertools,json,pathlib,types
ROOT=pathlib.Path(__file__).resolve().parents[1];SRC=ROOT/'sources/installed/python/sglang/srt'
OUT=ROOT/'runs/L002-R001-predicates';OUT.mkdir(parents=True,exist_ok=False)
hashes={};nodes={}
def load(path,name):
 p=SRC/path;b=p.read_bytes();hashes[str(path)]=hashlib.sha256(b).hexdigest()
 t=ast.parse(b);n=next(n for n in ast.walk(t) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
 nodes[name]={'path':str(path),'line':n.lineno,'end_line':n.end_lineno};return n
# SpeculativeAlgorithm's class body references CustomSpecAlgo as a default; spec_registry.py is not in the installed source subset, and the probed methods never use it.
env={'Enum':enum.Enum,'auto':enum.auto,'CustomSpecAlgo':type('CustomSpecAlgo',(),{})}
def compile_node(n):
 m=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),n],type_ignores=[])
 exec(compile(ast.fix_missing_locations(m),'<exact-installed-ast>','exec'),env)
compile_node(load(pathlib.Path('speculative/spec_info.py'),'SpeculativeAlgorithm'))
compile_node(load(pathlib.Path('managers/schedule_batch.py'),'grammar_needs_sync'))
compile_node(load(pathlib.Path('managers/scheduler.py'),'is_disable_overlap_for_batch'))
compile_node(load(pathlib.Path('managers/scheduler.py'),'_dispatch_event_loop_once'))
A=env['SpeculativeAlgorithm'];rows=[]
expected_support={'EAGLE','EAGLE3','FROZEN_KV_MTP','STANDALONE','DFLASH','DSPARK'}
for a in A:assert a.supports_grammar_overlap()==(a.name in expected_support),a
for a,grammar,decode,queued,dp_sync,consecutive in itertools.product(list(A),[False,True],[False,True],[0,1],[False,True],[False,True]):
 flag=types.SimpleNamespace(get=lambda:consecutive)
 env['envs']=types.SimpleNamespace(SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP=flag)
 mode=types.SimpleNamespace(is_extend=lambda:not decode,is_decode=lambda:decode)
 batch=types.SimpleNamespace(spec_algorithm=a,has_grammar=grammar,forward_mode=mode,is_extend_in_batch=not decode)
 batch.grammar_needs_sync=types.MethodType(env['grammar_needs_sync'],batch)
 scheduler=types.SimpleNamespace(require_mlp_sync=dp_sync,result_queue=[None]*queued)
 got=bool(env['is_disable_overlap_for_batch'](scheduler,batch,batch))
 expected=bool((consecutive and not decode) or (a.name!='NONE' and grammar and a.name not in expected_support and decode and queued))
 assert got==expected,(a.name,grammar,decode,queued,dp_sync,consecutive,got,expected)
 rows.append({'algorithm':a.name,'has_grammar':grammar,'decode':decode,'queue_size':queued,'dp_sync':dp_sync,'consecutive_flag':consecutive,'disable_overlap':got})
dispatch=[]
for role,pp,overlap in itertools.product(['null','prefill','decode'],[1,2],[False,True]):
 role_enum=enum.Enum('DisaggregationMode',{'NULL':'null','PREFILL':'prefill','DECODE':'decode'});env['DisaggregationMode']=role_enum
 env['get_parallel']=lambda:types.SimpleNamespace(pp_size=pp)
 obj=types.SimpleNamespace(disaggregation_mode=role_enum(role),enable_pdmux=False,enable_overlap_mlx=False,enable_overlap=overlap)
 called=[]
 names=['event_loop_pdmux','event_loop_pp','event_loop_overlap_mlx','event_loop_overlap','event_loop_normal','event_loop_pp_disagg_prefill','event_loop_overlap_disagg_prefill','event_loop_normal_disagg_prefill','event_loop_pp_disagg_decode','event_loop_overlap_disagg_decode','event_loop_normal_disagg_decode']
 for n in names:setattr(obj,n,lambda n=n:called.append(n))
 env['_dispatch_event_loop_once'](obj)
 expected=('event_loop_pp' if pp>1 else 'event_loop_overlap' if overlap else 'event_loop_normal')+('' if role=='null' else '_disagg_'+role)
 assert called==[expected],(role,pp,overlap,called,expected)
 dispatch.append({'role':role,'pp_size':pp,'overlap':overlap,'called':called})
v={'scope':'Exact AST predicates only; no imports of SGLang, no devices, no grammar FSM/collectives executed; does not prove NPU or PD service support.','compile_placeholders':['CustomSpecAlgo'],'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':'ce0a8fea5f4245839d174b2b8c55692be1810a7f','source_hashes':hashes,'nodes':nodes,'grammar_support':{a.name:a.supports_grammar_overlap() for a in A},'grammar_cases':len(rows),'dispatch_cases':len(dispatch),'grammar_rows':rows,'dispatch_rows':dispatch,'correctness':'PASS','exit_code':0}
(OUT/'result.json').write_text(json.dumps(v,indent=2)+'\n');(OUT/'receipt.json').write_text(json.dumps({'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'result_sha256':hashlib.sha256((OUT/'result.json').read_bytes()).hexdigest(),'exit_code':0,'correctness':'PASS','performance_claim':False},indent=2)+'\n');print(json.dumps({'grammar_cases':len(rows),'dispatch_cases':len(dispatch),'correctness':'PASS'}))
