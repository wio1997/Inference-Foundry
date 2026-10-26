import csv,glob,json,statistics
from pathlib import Path
base=Path('/data/wio/Inference_Foundry/evidence/20260926_loop060_resource/run246/profile')
cand=Path('/data/wio/Inference_Foundry/evidence/20260926_loop062_nongmm/run260/profile')
shape_keys={'"134217728"','"8192,16384"'}
def census(root):
 windows=[]
 for rank in range(8):
  folders=sorted(glob.glob(str(root/f'rank{rank}_*/ASCEND_PROFILER_OUTPUT')))
  assert folders,(root,rank)
  f=Path(folders[-1]);trace=json.loads((f/'trace_view.json').read_text());scopes=sorted((e for e in trace if e.get('name')=='extreme::target' and e.get('cat')=='cpu_op'),key=lambda e:float(e['ts']))
  assert len(scopes)==2,(f,len(scopes))
  with (f/'kernel_details.csv').open() as h: rows=list(csv.DictReader(h))
  for cyc,scope in enumerate(scopes):
   start=float(scope['ts']);end=start+float(scope['dur'])
   selected=[r for r in rows if start<=float(r.get('Start Time(us)','0').strip() or 0)<end]
   big=[r for r in selected if r['Name']=='aclnnInplaceCopy_TensorMoveAiCore_TensorMove' and r['Input Shapes'] in shape_keys and r['Input Data Types']=='DT_BF16']
   hcom=[r for r in selected if r['Name'].startswith('hcom_')]
   windows.append({'rank':rank,'cycle':cyc,'scope_ms':float(scope['dur'])/1000,'big_copy_count':len(big),'big_copy_shapes':sorted(r['Input Shapes'] for r in big),'big_copy_task_sum_ms':sum(float(r['Duration(us)']) for r in big)/1000,'big_copy_read_GB':sum(float(r['aiv_read_main_memory_datas(KB)'])*1024 for r in big)/1e9,'big_copy_write_GB':sum(float(r['aiv_write_main_memory_datas(KB)'])*1024 for r in big)/1e9,'hcom_count':len(hcom)})
 return windows
b=census(base);c=census(cand)
result={'status':'valid' if len(b)==len(c)==16 and all(x['big_copy_count']==2 for x in b) and all(x['big_copy_count']==0 for x in c) and all(x['hcom_count']==265 for x in b) and sum(x['hcom_count']==264 for x in c)>=15 else 'invalid','baseline':{'windows':len(b),'median_scope_ms':statistics.median(x['scope_ms'] for x in b),'median_big_copy_task_sum_ms':statistics.median(x['big_copy_task_sum_ms'] for x in b),'median_big_copy_read_GB':statistics.median(x['big_copy_read_GB'] for x in b),'median_big_copy_write_GB':statistics.median(x['big_copy_write_GB'] for x in b)},'candidate':{'windows':len(c),'median_scope_ms':statistics.median(x['scope_ms'] for x in c),'median_big_copy_task_sum_ms':statistics.median(x['big_copy_task_sum_ms'] for x in c)},'scope_delta_ms':statistics.median(x['scope_ms'] for x in c)-statistics.median(x['scope_ms'] for x in b),'limit':'Synchronized profiler target scopes differ across runs; one rank-cycle includes three extra HCCL rows from asynchronous timing overlap; cause of removed large copies established by shape/count, but scope delta is not a formal E2E gain and may include profiling variance. Graph task time is not additive wall time.','windows':{'base':b,'candidate':c}}
print(json.dumps(result,indent=2))
