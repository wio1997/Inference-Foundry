#!/usr/bin/env python3
"""Attribution and non-binding traffic sensitivity for Run247 latest graph windows."""
import argparse,csv,glob,json,statistics
from collections import defaultdict
from pathlib import Path

def num(s):
 try:return float(s)
 except (TypeError,ValueError):return 0.0

def family(n):
 if n.startswith('aclnnGroupedMatmulSwigluQuantWeightNzV2_'):return 'gmm1'
 if n.startswith('aclnnGroupedMatmulWeightNz_'):return 'gmm2'
 if n.startswith('aclnnQuantMatmulWeightNz_'):return 'quant_matmul'
 if n=='Compressor':return 'compressor'
 if n.startswith('aclnnScatterNdUpdateSk_'):return 'scatter_sk'
 if n=='SparseAttnSharedkv':return 'sparse_attention'
 if n.startswith('hcom_'):return 'hcom'
 if n.startswith(('AivKernel','HcPre','HcPost')):return 'comm_associated_aiv'
 if n.startswith('aclnnMatmul_'):return 'plain_matmul'
 if n.startswith('aclnnTransposeBatchMatMul_'):return 'transpose_matmul'
 if n.startswith('aclnnInplaceCopy_'):return 'inplace_copy'
 if n=='VllmQuantLightningIndexer':return 'indexer'
 return 'remaining'

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--profile-dir',type=Path,required=True)
 ap.add_argument('--run247-analysis',type=Path,required=True)
 ap.add_argument('--output',type=Path,required=True)
 a=ap.parse_args()
 source=json.loads(a.run247_analysis.read_text())
 assert source['status']=='valid' and source['latest_valid_windows']==16
 windows=[]
 for rank in range(8):
  folders=sorted(glob.glob(str(a.profile_dir/f'rank{rank}_*/ASCEND_PROFILER_OUTPUT')))
  assert len(folders)==5
  folder=Path(folders[-1])
  trace=json.loads((folder/'trace_view.json').read_text())
  scopes=sorted((e for e in trace if e.get('name')=='extreme::target' and e.get('cat')=='cpu_op'),key=lambda e:float(e['ts']))
  assert len(scopes)==2
  with (folder/'kernel_details.csv').open(newline='') as f: rows=list(csv.DictReader(f))
  for cycle,scope in enumerate(scopes):
   start=float(scope['ts']);end=start+float(scope['dur'])
   groups=defaultdict(lambda:dict(count=0,read_GB=0.,write_GB=0.,duration_sum_ms=0.))
   for row in rows:
    t=num(row['Start Time(us)'])
    if not start<=t<end:continue
    q=groups[family(row['Name'])]
    q['count']+=1
    q['read_GB']+=(num(row['aic_read_main_memory_datas(KB)'])+num(row['aiv_read_main_memory_datas(KB)']))*1024/1e9
    q['write_GB']+=(num(row['aic_write_main_memory_datas(KB)'])+num(row['aiv_write_main_memory_datas(KB)']))*1024/1e9
    q['duration_sum_ms']+=num(row['Duration(us)'])/1000
   windows.append(dict(rank=rank,cycle=cycle,scope_ms=(end-start)/1000,groups=dict(groups)))
 for w,s in zip(windows,sorted(source['windows'],key=lambda w:(w['rank'],w['cycle']))):
  assert (w['rank'],w['cycle'])==(s['rank'],s['cycle'])
  assert abs(sum(v['read_GB'] for v in w['groups'].values())-sum(v['read_KB'] for v in s['families'].values())*1024/1e9)<1e-5
 summary={}
 for k in sorted({k for w in windows for k in w['groups']}):
  vals=[w['groups'].get(k,{}) for w in windows]
  summary[k]={f:{'median':statistics.median(v.get(f,0) for v in vals),'min':min(v.get(f,0) for v in vals),'max':max(v.get(f,0) for v in vals)} for f in ('count','read_GB','write_GB','duration_sum_ms')}
 all_read=sum(v['read_GB']['median'] for v in summary.values())
 all_write=sum(v['write_GB']['median'] for v in summary.values())
 # These are dimensional sensitivities, not hardware limits.
 sensitivity={str(bw):dict(gross_counter_GB=all_read+all_write,idealized_ms=(all_read+all_write)/bw) for bw in (1.0,1.3,1.6)}
 out=dict(status='valid_attribution_nonbinding_sensitivity',windows=windows,summary=summary,sensitivity_TBps=sensitivity,
          limitations=['AIC+AIV counters are task-reported transactions, not uniquely compulsory HBM bytes.',
                       'Communication-associated names are a naming inference; HCCL link bytes remain unavailable.',
                       'Kernel duration sums and profiled target scopes are not exposed E2E savings.',
                       'Idealized bandwidth sensitivity does not represent attainable full-graph throughput or hardware upper bound.'])
 a.output.write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'windows':len(windows),'all_read_GB':all_read,'all_write_GB':all_write,'other_breakdown':{k:round(summary[k]['read_GB']['median'],3) for k in ('plain_matmul','transpose_matmul','inplace_copy','comm_associated_aiv','indexer','remaining')},'sensitivity':sensitivity},indent=2))
if __name__=='__main__':main()
