#!/usr/bin/env python3
"""Exact sender AST versus range-metadata candidate, CPU fixture only."""
import ast,copy,datetime,hashlib,itertools,json,math,pathlib,platform,random,statistics,sys,time,tracemalloc,types
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'runs/L004-R001-descriptor-cpu';OUT.mkdir(parents=True,exist_ok=False)
S=ROOT/'sources/installed/python/sglang/srt';paths=['disaggregation/ascend/conn.py','disaggregation/common/utils.py']
hashes={p:hashlib.sha256((S/p).read_bytes()).hexdigest() for p in paths}
def node(path,name):return next(n for n in ast.walk(ast.parse((S/path).read_text())) if isinstance(n,ast.FunctionDef) and n.name==name)
env={'np':np}
def install(n):
 m=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),n],type_ignores=[])
 exec(compile(ast.fix_missing_locations(m),'<isolated-sender-ast>','exec'),env)
group_node=node(paths[1],'group_concurrent_contiguous');install(group_node)
baseline_node=node(paths[0],'send_kvcache');baseline_node.name='baseline_sender';install(baseline_node)
candidate_group_source='''
def range_metadata(src_indices, dst_indices):
    if src_indices.size == 0 or dst_indices.size == 0:
        return []
    if src_indices.size != dst_indices.size:
        raise ValueError("group_concurrent_contiguous requires equal-length src/dst index arrays, "
                         f"got {src_indices.size} and {dst_indices.size}")
    brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
    starts = np.concatenate((np.array([0], dtype=np.int64), brk))
    ends = np.concatenate((brk, np.array([src_indices.size], dtype=np.int64)))
    return [(int(src_indices[start]), int(dst_indices[start]), int(end-start))
            for start, end in zip(starts, ends)]
'''
install(ast.parse(candidate_group_source).body[0])
candidate_node=copy.deepcopy(baseline_node);candidate_node.name='candidate_sender';mutations=0
for i,n in enumerate(candidate_node.body):
 if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='group_concurrent_contiguous':
  candidate_node.body[i]=ast.parse('transfer_ranges = range_metadata(prefill_kv_indices, dst_kv_indices)').body[0];mutations+=1
 if isinstance(n,ast.FunctionDef) and n.name=='set_transfer_blocks':
  candidate_node.body[i]=ast.parse('''
def set_transfer_blocks(src_ptr, dst_ptr, item_len):
    transfer_blocks = []
    for src_start, dst_start, run_len in transfer_ranges:
        src_addr = src_ptr + src_start * item_len
        dst_addr = dst_ptr + dst_start * item_len
        length = item_len * run_len
        transfer_blocks.append((src_addr, dst_addr, length))
    return transfer_blocks
''').body[0];mutations+=1
assert mutations==2;install(candidate_node)
base=env['baseline_sender'];candidate=env['candidate_sender'];group=env['group_concurrent_contiguous'];ranges=env['range_metadata']
(OUT/'candidate-method.py').write_text(candidate_group_source+'\n'+ast.unparse(ast.fix_missing_locations(candidate_node))+'\n')
def fixture(layers,capture=True):
 s=types.SimpleNamespace(pp_size=1,is_hybrid_mla_backend=False,enable_custom_mem_pool=False)
 s.kv_args=types.SimpleNamespace(kv_data_ptrs=[2**40+i*2**32 for i in range(layers)],kv_item_lens=[128 if i%2==0 else 64 for i in range(layers)])
 s.dst=[2**46+i*2**32 for i in range(layers)];s.blocks=None
 s._validate_envelope_kv_layout=lambda *a:None
 def transfer(session,blocks):
  if capture:s.blocks=blocks
  return 0
 s._transfer_data=transfer;return s
def call(fn,s,src,dst,hisparse=None):return fn(s,'CPU_FIXTURE_NO_TRANSFER',src,s.dst,dst,None,dst_device_kv_indices=hisparse)
def oracle(s,src,dst):
 if not len(src) or not len(dst):return []
 if len(src)!=len(dst):raise ValueError
 out=[]
 for a,b,width in zip(s.kv_args.kv_data_ptrs,s.dst,s.kv_args.kv_item_lens):
  previous=None
  for x,y in zip(src,dst):
   x,y=int(x),int(y)
   if previous is not None and x==previous[0]+1 and y==previous[1]+1:
    q=out[-1];out[-1]=(q[0],q[1],q[2]+width)
   else:out.append((a+x*width,b+y*width,width))
   previous=(x,y)
 return out
rng=random.Random(20261019);checks=[]
def check(src,dst,layers,label,hisparse=None):
 a=fixture(layers);b=fixture(layers);ea=eb=None
 try:ra=call(base,a,src,dst,hisparse)
 except Exception as e:ea=(type(e).__name__,str(e))
 try:rb=call(candidate,b,src,dst,hisparse)
 except Exception as e:eb=(type(e).__name__,str(e))
 assert ea==eb,(label,ea,eb)
 if ea is None:
  expected=oracle(a,src,dst);assert ra==rb==0 and a.blocks==b.blocks==expected,label
  blocks=len(expected);payload=sum(x[2] for x in expected)
 else:blocks=payload=None
 checks.append({'case':label,'source_count':len(src),'destination_count':len(dst),'layers':layers,'descriptor_count':blocks,'payload_bytes':payload,'exception':ea})
for n in [0,1,2,8,31,4096]:
 x=np.arange(10000,10000+n,dtype=np.int32);y=np.arange(20000,20000+n,dtype=np.int32)
 check(x,y,78,'contiguous-'+str(n));check(x*2,y*3,3,'fragmented-'+str(n))
 check(x,y*2,3,'destination-gaps-'+str(n))
check(np.arange(10000,42768,dtype=np.int32),np.arange(50000,82768,dtype=np.int32),78,'large-contiguous-32768')
check(np.array([],dtype=np.int32),np.array([7],dtype=np.int32),3,'empty-source-nonempty-destination')
check(np.array([7],dtype=np.int32),np.array([],dtype=np.int32),3,'nonempty-source-empty-destination')
check(np.array([7,8],dtype=np.int32),np.array([9],dtype=np.int32),3,'mismatched-nonempty')
check(np.array([7],dtype=np.int32),np.array([9],dtype=np.int32),3,'hisparse-rejection',np.array([9],dtype=np.int32))
for i in range(128):
 n=rng.randrange(1,100);src=np.array([10000+rng.randrange(256) for _ in range(n)],dtype=np.int32);dst=np.array([20000+rng.randrange(256) for _ in range(n)],dtype=np.int32)
 if i%4==0:src=src.astype(np.int64);dst=dst.astype(np.int64)
 if i%4==1:
  src=np.repeat(src,2)[::2];dst=np.repeat(dst,2)[::2];src.setflags(write=False);dst.setflags(write=False)
 check(src,dst,3,'random-'+str(i))
bench_cases=[('short-contiguous',8,False,128),('long-contiguous',4096,False,16),('large-contiguous',32768,False,4),('fragmented',4096,True,1)]
measurements=[]
def summary(xs):
 ys=sorted(xs);return {'median_ns':statistics.median(xs),'min_ns':min(xs),'max_ns':max(xs),'iqr_ns':float(np.percentile(ys,75)-np.percentile(ys,25))}
def ci_ratio(pairs):
 logs=[math.log(a/b) for a,b in pairs];r=random.Random(20261019)
 bs=sorted(math.exp(statistics.mean(r.choices(logs,k=len(logs)))) for _ in range(4000))
 return {'ratio_of_geomeans':math.exp(statistics.mean(logs)),'bootstrap_95_interval':[bs[99],bs[3899]]}
for name,n,fragmented,repeats in bench_cases:
 src=np.arange(10000,10000+n,dtype=np.int32);dst=np.arange(100000,100000+n,dtype=np.int32)
 if fragmented:src=src*2;dst=dst*3
 f=fixture(78,capture=False)
 def timed(fn):
  t=time.perf_counter_ns()
  for _ in range(repeats):assert call(fn,f,src,dst)==0
  return (time.perf_counter_ns()-t)/repeats
 for _ in range(3):timed(base);timed(candidate)
 aa=[(timed(base),timed(base)) for _ in range(5)]
 ab=[];orders=[]
 for i in range(9):
  if i%2==0:a=timed(base);b=timed(candidate);order='AB'
  else:b=timed(candidate);a=timed(base);order='BA'
  ab.append((a,b));orders.append(order)
 peaks={}
 for label,fn in [('baseline_group',group),('range_group',ranges)]:
  tracemalloc.start();result=fn(src,dst);current,peak=tracemalloc.get_traced_memory();tracemalloc.stop();peaks[label]={'current_bytes':current,'peak_bytes':peak};del result
 r=len(ranges(src,dst));aa_stats=ci_ratio(aa);ab_stats=ci_ratio(ab)
 measurements.append({'case':name,'N_page_indices':n,'R_paired_contiguous_runs':r,'layers':78,'input_dtype':str(src.dtype),'input_read_bytes':src.nbytes+dst.nbytes,'fixture_repeats_per_sample':repeats,'aa_pairs_ns':aa,'ab_pairs_ns':ab,'ab_order':orders,'baseline':summary([a for a,b in ab]),'candidate':summary([b for a,b in ab]),'aa_ratio':aa_stats,'ab_ratio':ab_stats,'group_tracemalloc':peaks,'local_diagnostic':'SEPARATED_FROM_AA' if ab_stats['bootstrap_95_interval'][0]>max(1,aa_stats['bootstrap_95_interval'][1]) else 'INCONCLUSIVE','descriptor_count_both':78*r,'transfer_count_both':1})
result={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':'ce0a8fea5f4245839d174b2b8c55692be1810a7f','source_hashes':hashes,'baseline_method_line':baseline_node.lineno,'group_method_line':group_node.lineno,'candidate_ast_changes':mutations,'candidate_method_sha256':hashlib.sha256((OUT/'candidate-method.py').read_bytes()).hexdigest(),'correctness':'PASS','correctness_cases':len(checks),'cases':checks,'numpy':np.__version__,'python':sys.version,'platform':platform.platform(),'clock':'perf_counter_ns','measurements':measurements,'scope':'pp=1, non-hybrid, custom-mem-pool false CPU fixture. Exact sender AST uses no-op envelope validation and no-op transfer; no addresses dereferenced, no NPU, no model, no network transfer. Same descriptors/bytes/order, no claim of serving support or speedup. No production source changed. CPU microbenchmark is diagnostic only; global serving evidence remains E0.','cost_model':'Both perform O(N) paired diff; old metadata boxes O(N) page values in per-run lists, candidate boxes O(R) start/length values. Ordered descriptor construction remains O(layers*R); payload bytes and transfer calls are unchanged. HBM/link physical rates and serving exposure are UNKNOWN.','exit_code':0}
(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
(OUT/'receipt.json').write_text(json.dumps({'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'result_sha256':hashlib.sha256((OUT/'result.json').read_bytes()).hexdigest(),'candidate_method_sha256':result['candidate_method_sha256'],'source_hashes':hashes,'exit_code':0,'correctness':'PASS','production_source_changed':False},indent=2)+'\n')
print(json.dumps({'correctness_cases':len(checks),'cases':[{'case':m['case'],'local_ratio':m['ab_ratio'],'diagnostic':m['local_diagnostic']} for m in measurements]}))
