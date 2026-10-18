#!/usr/bin/env python3
"""L006 controlled baseline or paired candidate experiment, using reviewed helpers."""
import ast,datetime,hashlib,importlib.util,json,pathlib,sys,types
import numpy as np
R=pathlib.Path(__file__).resolve().parents[1];phase=sys.argv[1];assert phase in ['baseline','candidate']
tag='L006-R001-baseline' if phase=='baseline' else 'L006-R002-candidate';O=R/'runs'/tag;O.mkdir(parents=True,exist_ok=False)
hp=R/'scripts/probe_full_consumer_cpu.py';sp=importlib.util.spec_from_file_location('reviewed_l005_helpers',hp);lib=importlib.util.module_from_spec(sp);sp.loader.exec_module(lib)
S=R/'sources/installed/python/sglang/srt';cp=R/'candidates/L006/direct_native_columns.py';gp=S/'disaggregation/common/utils.py';tp=S/'disaggregation/mooncake/conn.py'
group=lib.find_top_level_function(ast.parse(gp.read_text()),'group_concurrent_contiguous');consumer=lib.find_method(ast.parse(tp.read_text()),'MooncakeKVManager','_transfer_data')
ct=ast.parse(cp.read_text());functions=[n for n in ct.body if isinstance(n,ast.FunctionDef)]
env={'np':np,'concurrent':lib.concurrent};exec(lib.compile_nodes([group,consumer,*functions]),env)
B=env['candidate_sender'];C=env['columns_sender'];chosen=B if phase=='baseline' else C
lib.RUNTIME.update({'base_env':env,'candidate_env':env,'base_sender':B,'candidate_sender':chosen,'real_group':env['group_concurrent_contiguous'],'range_metadata':env['range_metadata'],'transfer_fn':env['_transfer_data']})
l004p=R/'runs/L004-R001-descriptor-cpu/result.json';ref=json.loads(l004p.read_text())
correctness=lib.run_correctness(lib.build_cases(),ref);assert correctness['status']=='PASS'
extra=[]
if phase=='candidate':
 original=env['installed_sender'];spy_calls=[]
 def spy(*args):spy_calls.append(args);return 73
 x=np.arange(10000,10008,dtype=np.int32);y=np.arange(20000,20008,dtype=np.int32)
 try:
  env['installed_sender']=spy
  for flag,value in [('pp_size',2),('is_hybrid_mla_backend',True),('enable_custom_mem_pool',True)]:
   f=lib.make_fixture(lib.CaptureEngine());setattr(f,flag,value);before=len(spy_calls);assert lib.call_sender(C,f,x,y)==73
   a=spy_calls[-1];assert len(spy_calls)==before+1 and a[0] is f and a[1]==lib.SESSION and a[2] is x and a[3] is f.dst and a[4] is y and a[5:] == (None,None,None,None,None)
   extra.append({'kind':'fallback-dispatch-only','flag':flag,'value':value,'status':73,'original_sender_forwarded_arguments':True})
 finally:env['installed_sender']=original
 for label,xx,yy in [('empty-layers-nonempty-pages',x,y),('empty-layers-empty-pages',x[:0],y[:0])]:
  a=lib.execute_case(B,lib.make_fixture(lib.CaptureEngine(),0),xx,yy);b=lib.execute_case(C,lib.make_fixture(lib.CaptureEngine(),0),xx,yy)
  assert a['exception']==b['exception']==None and a['status']==b['status']==0 and a['native_calls']==b['native_calls']==0;extra.append({'kind':label,'correctness':'PASS'})
 f=lib.make_fixture(lib.CaptureEngine());f.pp_size=2
 a=lib.execute_case(env['installed_sender'],f,x,y,np.array([7],dtype=np.int32));b=lib.execute_case(C,lib.make_fixture(lib.CaptureEngine()),x,y,np.array([7],dtype=np.int32))
 assert a['exception']==b['exception'] and a['exception'][0]=='NotImplementedError';extra.append({'kind':'HiSparse-rejection','correctness':'PASS'})
gc_start=lib.gc_state();assert gc_start['enabled'] and gc_start['callbacks']==0
bench=lib.run_benchmark();gc_stop=lib.gc_state();assert gc_start['enabled']==gc_stop['enabled'] and gc_start['threshold']==gc_stop['threshold'] and gc_stop['callbacks']==0
inputs={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [hp,cp,gp,tp,l004p]}
result={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run':tag,'phase':phase,'source_commit':'ce0a8fea5f4245839d174b2b8c55692be1810a7f','inputs':inputs,'variants':{'baseline':'unchanged L004 range sender plus exact real consumer','candidate':'same range sender (baseline control)' if phase=='baseline' else 'guarded direct native columns; untargeted modes forward to original installed sender'},'correctness':correctness,'additional_checks':extra,'benchmark':bench,'gc_start':gc_start,'gc_stop':gc_stop,'scope':'Local CPU metadata fixture only; native engine is no-op. Original production code unchanged. Each phase is an actual separate execution. Candidate phase uses paired interleaved B/C samples for causal comparison; baseline phase contains only repeated B controls. No native transport, NPU or PD serving gain inferred.','exit_code':0}
(O/'result.json').write_text(json.dumps(result,indent=2)+'\n')
rc={'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'phase':phase,'inputs':inputs,'result_sha256':hashlib.sha256((O/'result.json').read_bytes()).hexdigest(),'exit_code':0,'correctness':'PASS'};(O/'receipt.json').write_text(json.dumps(rc,indent=2)+'\n')
print(json.dumps({'phase':phase,'correctness':'PASS','cases':correctness['cases_total'],'measurements':[{'case':m['case'],'thread_cpu_ratio':m['thread_cpu']['ab_ratio'],'wall_ratio':m['wall']['ab_ratio']} for m in bench['cases']]}))
