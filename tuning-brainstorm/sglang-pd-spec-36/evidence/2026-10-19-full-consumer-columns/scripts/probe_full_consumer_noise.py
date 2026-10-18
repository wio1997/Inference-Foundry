#!/usr/bin/env python3
"""Focused L005 repetition: interleave A/A blocks with A/B for the noisy case."""
import ast,datetime,hashlib,importlib.util,json,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'runs/L005-R002-interleaved-noise';OUT.mkdir(parents=True,exist_ok=False)
p=ROOT/'scripts/probe_full_consumer_cpu.py';spec=importlib.util.spec_from_file_location('l005_reviewed_helpers',p);lib=importlib.util.module_from_spec(spec);spec.loader.exec_module(lib)
src=ROOT/'sources/installed/python/sglang/srt';paths={'ascend':src/'disaggregation/ascend/conn.py','utils':src/'disaggregation/common/utils.py','consumer':src/'disaggregation/mooncake/conn.py','candidate':ROOT/'runs/L004-R001-descriptor-cpu/candidate-method.py'}
nodes={
 'send':lib.find_method(ast.parse(paths['ascend'].read_text()),'AscendKVManager','send_kvcache'),
 'group':lib.find_top_level_function(ast.parse(paths['utils'].read_text()),'group_concurrent_contiguous'),
 'transfer':lib.find_method(ast.parse(paths['consumer'].read_text()),'MooncakeKVManager','_transfer_data'),
 'ranges':lib.find_top_level_function(ast.parse(paths['candidate'].read_text()),'range_metadata'),
 'candidate':lib.find_top_level_function(ast.parse(paths['candidate'].read_text()),'candidate_sender')}
base_env={'np':np,'concurrent':lib.concurrent};candidate_env={'np':np,'concurrent':lib.concurrent}
exec(lib.compile_nodes([nodes['group'],nodes['send'],nodes['transfer']]),base_env)
exec(lib.compile_nodes([nodes['ranges'],nodes['candidate']]),candidate_env)
lib.RUNTIME.update({'base_env':base_env,'candidate_env':candidate_env,'base_sender':base_env['send_kvcache'],'candidate_sender':candidate_env['candidate_sender'],'transfer_fn':base_env['_transfer_data'],'real_group':base_env['group_concurrent_contiguous'],'range_metadata':candidate_env['range_metadata']})
A=lib.RUNTIME['base_sender'];B=lib.RUNTIME['candidate_sender'];x,y=lib.bench_arrays(4096,True)
fa=lib.make_fixture(lib.CaptureEngine());fb=lib.make_fixture(lib.CaptureEngine())
a=lib.execute_case(A,fa,x,y);b=lib.execute_case(B,fb,x,y);expected=lib.oracle(fa,x,y);cols=lib.oracle_columns(expected)
assert a['exception']==b['exception']==None and a['status']==b['status']==0
assert a['native_calls']==b['native_calls']==1
assert (a['src'],a['dst'],a['lengths'])==(b['src'],b['dst'],b['lengths'])==cols
payload_bytes=sum(v[2] for v in expected);count=len(expected)
del fa,fb,a,b,expected,cols
fa=lib.make_fixture(lib.CountOnlyEngine());fb=lib.make_fixture(lib.CountOnlyEngine());gc_before=lib.gc_state();assert gc_before['enabled']
for _ in range(3):lib.measure_sample(A,fa,x,y,1);lib.measure_sample(B,fb,x,y,1)
aa=[];ab=[];blocks=[]
for i in range(13):
 order=['AA','AB'] if i%2==0 else ['AB','AA'];block={'block':i,'order':order,'samples':{}}
 for kind in order:
  if kind=='AA':
   first=lib.measure_sample(A,fa,x,y,1);second=lib.measure_sample(A,fa,x,y,1);pair={'a':first,'b':second,'order':'AA'};aa.append(pair)
  elif i%2==0:
   first=lib.measure_sample(A,fa,x,y,1);second=lib.measure_sample(B,fb,x,y,1);pair={'a':first,'b':second,'order':'AB'};ab.append(pair)
  else:
   second=lib.measure_sample(B,fb,x,y,1);first=lib.measure_sample(A,fa,x,y,1);pair={'a':first,'b':second,'order':'BA'};ab.append(pair)
  assert first['native_calls']==second['native_calls']==1
  block['samples'][kind]=pair
 blocks.append(block)
stats={}
for key in ['wall_ns_per_call','thread_cpu_ns_per_call']:
 stats[key]={'baseline':lib.describe([v['a'] for v in ab],key),'candidate':lib.describe([v['b'] for v in ab],key),'aa_ratio':lib.paired_ratio([(v['a'],v['b']) for v in aa],key),'ab_ratio':lib.paired_ratio([(v['a'],v['b']) for v in ab],key)}
gc_after=lib.gc_state();assert gc_before['enabled']==gc_after['enabled'] and gc_before['threshold']==gc_after['threshold'] and gc_before['callbacks']==gc_after['callbacks']
result={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'case':'fragmented','N':4096,'R':4096,'layers':78,'descriptor_count':count,'payload_bytes':payload_bytes,'native_calls_per_sample':1,'correctness':'PASS','source_commit':'ce0a8fea5f4245839d174b2b8c55692be1810a7f','inputs':{str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [p,*paths.values()]},'protocol':'13 blocks; AA and AB block order alternates; AB and BA inner order alternates. Same methods, fixture and workload as L005-R001. Focused follow-up is justified by unresolved wall-time variability. No samples removed.','blocks':blocks,'stats':stats,'gc_before':gc_before,'gc_after':gc_after,'scope':'Local CPU only, exact Python consumer, no-op native engine; no callbacks/tracemalloc in timing; no native conversion/network/NPU/PD baseline. This is a new noise-measurement run, not a replay of the first benchmark.','exit_code':0}
(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
receipt={'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'result_sha256':hashlib.sha256((OUT/'result.json').read_bytes()).hexdigest(),'inputs':result['inputs'],'exit_code':0,'correctness':'PASS'}
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'correctness':'PASS','stats':stats}))
