#!/usr/bin/env python3
"""Clip Level1 MemoryAccess kernel rows to real target graph scopes."""
import argparse,csv,glob,json,statistics
from collections import defaultdict
from pathlib import Path
def classify(n):
 if n.startswith('aclnnGroupedMatmulSwigluQuantWeightNzV2_'):return 'gmm1'
 if n.startswith('aclnnGroupedMatmulWeightNz_'):return 'gmm2'
 if n.startswith('aclnnQuantMatmulWeightNz_'):return 'quant_matmul'
 if n=='Compressor':return 'compressor'
 if n.startswith('aclnnScatterNdUpdateSk_'):return 'scatter_sk'
 if n=='SparseAttnSharedkv':return 'sparse_attention'
 if n.startswith('hcom_'):return 'communication'
 return 'other'
def val(row,k):
 s=row.get(k,'').strip()
 try:return float(s)
 except:return 0.0
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--profile-dir',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 windows=[];invalid=[];sources=[]
 for rank in range(8):
  folders=sorted(glob.glob(str(a.profile_dir/f'rank{rank}_*/ASCEND_PROFILER_OUTPUT')))
  if not folders: invalid.append(dict(rank=rank,reason='missing_profile'));continue
  for capture,folder in enumerate(folders):
   folder=Path(folder);trace=folder/'trace_view.json';csvpath=folder/'kernel_details.csv'
   if not(trace.exists() and csvpath.exists()):invalid.append(dict(rank=rank,capture=capture,reason='missing_export'));continue
   sources.append(str(folder))
   events=json.loads(trace.read_text())
   scopes=sorted((e for e in events if e.get('name')=='extreme::target' and e.get('cat')=='cpu_op'),key=lambda e:float(e['ts']))
   if len(scopes)!=2:invalid.append(dict(rank=rank,capture=capture,reason=f'target_scope_count_{len(scopes)}'));continue
   with csvpath.open(newline='') as f:
    reader=csv.DictReader(f);headers=reader.fieldnames or []
    needed=['Start Time(us)','Name','Duration(us)','aic_read_main_memory_datas(KB)','aic_write_main_memory_datas(KB)','aiv_read_main_memory_datas(KB)','aiv_write_main_memory_datas(KB)']
    if any(k not in headers for k in needed):invalid.append(dict(rank=rank,capture=capture,reason='missing_memory_columns',headers=headers));continue
    kernel=list(reader)
   for cycle,scope in enumerate(scopes):
    start=float(scope['ts']);end=start+float(scope['dur']);families=defaultdict(lambda:dict(count=0,read_KB=0.0,write_KB=0.0,aic_read_KB=0.0,aiv_read_KB=0.0,aic_write_KB=0.0,aiv_write_KB=0.0,kernel_sum_ms=0.0))
    for row in kernel:
     ts=val(row,'Start Time(us)')
     if not start<=ts<end:continue
     name=row['Name'];fam=classify(name);v=families[fam]
     v['count']+=1
     for core in ('aic','aiv'):
      v[f'{core}_read_KB']+=val(row,f'{core}_read_main_memory_datas(KB)')
      v[f'{core}_write_KB']+=val(row,f'{core}_write_main_memory_datas(KB)')
     v['read_KB']=v['aic_read_KB']+v['aiv_read_KB']
     v['write_KB']=v['aic_write_KB']+v['aiv_write_KB']
     v['kernel_sum_ms']+=val(row,'Duration(us)')/1000
    expected={'gmm1':43,'gmm2':43,'quant_matmul':236,'compressor':62,'scatter_sk':126,'sparse_attention':43}
    bad={k:dict(actual=families[k]['count'],expected=n) for k,n in expected.items() if families[k]['count']!=n}
    if bad:invalid.append(dict(rank=rank,capture=capture,cycle=cycle,reason='family_count',detail=bad));continue
    if any(families[k]['read_KB']<=0 for k in ('gmm1','gmm2','quant_matmul','compressor','sparse_attention')) or families['scatter_sk']['write_KB']<=0:
     invalid.append(dict(rank=rank,capture=capture,cycle=cycle,reason='missing_positive_family_read'));continue
    windows.append(dict(rank=rank,capture=capture,cycle=cycle,scope_ms=(end-start)/1000,families=dict(families)))
 latest=max((w['capture'] for w in windows),default=-1)
 selected=[w for w in windows if w['capture']==latest]
 # Do not silently substitute warmup captures for the measured cohort.
 status='valid' if len(selected)>=12 and len({w['rank'] for w in selected})==8 else 'invalid_insufficient_latest_capture'
 keys=sorted({k for w in selected for k in w['families']})
 summary={}
 for k in keys:
  rows=[w['families'].get(k,{}) for w in selected]
  summary[k]={name:{'median':statistics.median(v),'min':min(v),'max':max(v)} for name,v in
              ((name,[r.get(name,0) for r in rows]) for name in ('count','read_KB','write_KB','aic_read_KB','aiv_read_KB','aic_write_KB','aiv_write_KB','kernel_sum_ms'))}
 out=dict(status=status,all_valid_windows=len(windows),latest_capture=latest,latest_valid_windows=len(selected),
          latest_ranks=sorted({w['rank'] for w in selected}),summary=summary,invalid=invalid[:40],sources=sources,
          windows=selected,limit='Level1 profiler/synchronized diagnostic; bytes sum AIC and AIV task counters, not necessarily unique compulsory HBM bytes. No HCCL link bytes or formal E2E conclusion.')
 a.output.write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(dict(status=status,all_valid_windows=len(windows),latest_valid_windows=len(selected),
                       gmm_read_GB=(summary.get('gmm1',{}).get('read_KB',{}).get('median',0)+summary.get('gmm2',{}).get('read_KB',{}).get('median',0))*1024/1e9,
                       family_read_GB={k:round(v['read_KB']['median']*1024/1e9,3) for k,v in summary.items()},
                       invalid=len(invalid)),indent=2))
if __name__=='__main__':main()
