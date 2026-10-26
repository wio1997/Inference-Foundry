"""Parse cumulative ITEM_TRACE snapshots from a diagnostic service log."""
import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--log',required=True);p.add_argument('--out',required=True);a=p.parse_args()
pat=re.compile(r'Worker_TP(\d+)_EP\d+.*ITEM_TRACE steps=(\d+) rows=(\[.*\])')
rows=defaultdict(dict)
for line in Path(a.log).open(errors='replace'):
 m=pat.search(line)
 if not m:continue
 rank,step=int(m[1]),int(m[2])
 parsed=ast.literal_eval(m[3])
 rows[rank][step]={f'{k[0]}:{k[1]}:{k[2]}':{'count':n,'sum_ms':total,'max_ms':maximum} for k,n,total,maximum in parsed}
summary={'snapshots_by_rank':{str(r):sorted(v) for r,v in rows.items()},'ranks':{}}
for rank,snaps in sorted(rows.items()):
 report={}
 for step in (56,112,184):
  if step in snaps:report[str(step)]={k:v for k,v in snaps[step].items() if 'dsa_cp.py:1008:' in k or 'dsa_cp.py:1009:' in k}
 for start,end,label in ((56,112,'warm_approx'),(112,184,'cold_approx')):
  if start in snaps and end in snaps:
   change={}
   for key,new in snaps[end].items():
    if 'dsa_cp.py:1008:' not in key and 'dsa_cp.py:1009:' not in key:continue
    old=snaps[start].get(key,{'count':0,'sum_ms':0})
    change[key]={'count':new['count']-old['count'],'sum_ms':round(new['sum_ms']-old['sum_ms'],3)}
   report[label]=change
 summary['ranks'][str(rank)]=report
Path(a.out).write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary['ranks'],indent=2))
