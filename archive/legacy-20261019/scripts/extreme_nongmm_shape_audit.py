#!/usr/bin/env python3
"""Audit exact non-GMM graph shapes without interpreting task time as exposed latency."""
import argparse,csv,glob,json,statistics
from collections import defaultdict
from pathlib import Path

def num(x):
 try:return float(x)
 except (TypeError,ValueError):return 0.0

def family(n):
 if n.startswith('aclnnQuantMatmulWeightNz_'):return 'quant_matmul'
 if n=='Compressor':return 'compressor'
 if n=='SparseAttnSharedkv':return 'sparse_attention'
 if n.startswith('aclnnMatmul_'):return 'plain_matmul'
 if n.startswith('aclnnTransposeBatchMatMul_'):return 'transpose_matmul'
 return None

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--profile-dir',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 windows=[]
 for rank in range(8):
  folders=sorted(glob.glob(str(a.profile_dir/f'rank{rank}_*/ASCEND_PROFILER_OUTPUT')));assert len(folders)==5
  folder=Path(folders[-1]);events=json.loads((folder/'trace_view.json').read_text())
  scopes=sorted((e for e in events if e.get('name')=='extreme::target' and e.get('cat')=='cpu_op'),key=lambda e:float(e['ts']));assert len(scopes)==2
  with (folder/'kernel_details.csv').open(newline='') as f: rows=list(csv.DictReader(f))
  for cycle,scope in enumerate(scopes):
   start=float(scope['ts']);end=start+float(scope['dur']);d=defaultdict(lambda:dict(count=0,read_GB=0.,write_GB=0.,duration_sum_ms=0.))
   for r in rows:
    t=num(r['Start Time(us)']);fam=family(r['Name'])
    if fam is None or not start<=t<end:continue
    key=fam+'|'+r['Input Shapes'].strip('"')
    x=d[key];x['count']+=1;x['read_GB']+=(num(r['aic_read_main_memory_datas(KB)'])+num(r['aiv_read_main_memory_datas(KB)']))*1024/1e9
    x['write_GB']+=(num(r['aic_write_main_memory_datas(KB)'])+num(r['aiv_write_main_memory_datas(KB)']))*1024/1e9
    x['duration_sum_ms']+=num(r['Duration(us)'])/1000
   windows.append(dict(rank=rank,cycle=cycle,shapes=dict(d)))
 assert len(windows)==16
 keys=sorted({k for w in windows for k in w['shapes']})
 summary={}
 for k in keys:
  values=[w['shapes'].get(k,{}) for w in windows]
  m={f:statistics.median(v.get(f,0) for v in values) for f in ('count','read_GB','write_GB','duration_sum_ms')}
  m['counter_rate_TBps']=m['read_GB']/m['duration_sum_ms'] if m['duration_sum_ms'] else None
  summary[k]=m
 out=dict(status='valid_shape_audit',window_count=len(windows),summary=summary,windows=windows,
          limit='Counter rate is read bytes divided by summed profiler task durations of one shape, not attainable bandwidth or exposed E2E time.')
 a.output.write_text(json.dumps(out,indent=2)+'\n')
 for k,v in sorted(summary.items(),key=lambda x:-x[1]['read_GB']):
  print(k,'count',v['count'],'read_GB',round(v['read_GB'],3),'task_ms',round(v['duration_sum_ms'],3),'counter_TBps',round(v['counter_rate_TBps'],3))
if __name__=='__main__':main()
