#!/usr/bin/env python3
"""Read existing window-2 exports for vocab-sized head/sampling families only."""
import collections
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import statistics

BASE=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/official-20261030T084726Z-cb252d1d/analysis/operator-export/copies')
def ns(x):
 v=Decimal(x.strip())*1000;assert v==int(v);return int(v)
sources=[]
for spec in SPECS:
 p=BASE/spec['source_dir']/'ASCEND_PROFILER_OUTPUT/kernel_details.csv'
 assert hashlib.sha256(p.read_bytes()).hexdigest()==spec['sha256']
 groups=collections.defaultdict(list);examples=collections.defaultdict(list)
 with p.open(newline='') as f:
  for x in csv.DictReader(f):
   t=x['Type'];shapes=x['Input Shapes']+';'+x['Output Shapes']
   vocab=any(v in shapes for v in ('129280','16160','16192'))
   name_match=bool(re.search(r'reject|argmax|arg_max|sample',t,re.I))
   if not (vocab or name_match):continue
   duration=ns(x['Duration(us)'])
   if duration<=0:continue
   key=(t,x['Input Shapes'],x['Output Shapes'],x['Input Data Types'],x['Output Data Types'],x['Accelerator Core'])
   groups[key].append(duration)
   if len(examples[key])<3:examples[key].append({k:x[k] for k in ('Model ID','Stream ID','Task ID','Name','Start Time(us)','Duration(us)')})
 results=[]
 for key,values in groups.items():
  ordered=sorted(values);hist=collections.Counter(values)
  results.append({'type':key[0],'input_shapes':key[1],'output_shapes':key[2],'input_dtypes':key[3],'output_dtypes':key[4],'core_label':key[5],
   'count':len(values),'sum_ns':sum(values),'min_ns':min(values),'median_ns':statistics.median(values),'p95_ns':ordered[(95*len(ordered)+99)//100-1],'max_ns':max(values),
   'duration_histogram_ns':dict(sorted(hist.items())),'first_examples':examples[key]})
 sources.append({'rank':spec['rank'],'source_dir':spec['source_dir'],'csv_sha256':spec['sha256'],'groups':results})
print(json.dumps({'scope':'same official capture, later window, all16rank; exact shapes or sampler-name families',
 'limits':['Shape compatibility and count ratios are not tensor identity or request/replay binding.','Group sums and kernel durations are not exposed critical-path communication or end-to-end savings.','Selection is lexical on exact exported dimensions/names, not all runtime consumers or all operators.'],
 'rows':sources},indent=2))
