#!/usr/bin/env python3
"""Read-only Run287 layer2 storage-alias census, without claiming live byte closure."""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--ownership-dir',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
files=sorted(a.ownership_dir.glob('rank*_cohort*.jsonl'))
if len(files)!=40:raise SystemExit(f'expected 40 rank-cohort files, got {len(files)}')
rows=[]
layouts=[]
representative=None
for f in files:
 with f.open() as h:
  first=json.loads(next(h))
 views=first.get('layer2_cache_views')
 if not views:raise SystemExit(f'missing layer2 views {f}')
 groups={}
 for v in views:
  key=(int(v['storage_ptr']),int(v['storage_nbytes']))
  x=groups.setdefault(key,{'storage_ptr':key[0],'storage_nbytes':key[1],
                             'names':[],'aliases':set(),'views':[]})
  x['names'].append(v['name']);x['aliases'].update(v['aliases'])
  item={k:v[k] for k in ('name','dtype','shape','stride','storage_offset','data_ptr')}
  x['views'].append(item)
 if len(groups)!=3:raise SystemExit(f'expected three distinct backing stores {f}: {len(groups)}')
 normalized=[]
 for g in groups.values():
  g['names']=sorted(g['names']);g['aliases']=sorted(g['aliases'])
  g['views'].sort(key=lambda x:x['name'])
  normalized.append(g)
 normalized.sort(key=lambda x:(x['storage_nbytes'],x['names']))
 if representative is None:
  representative=normalized
 layout=json.dumps([(g['storage_nbytes'],g['names'],g['aliases'],
                     [(v['name'],v['dtype'],v['shape'],v['stride'],v['storage_offset']) for v in g['views']])
                    for g in normalized],sort_keys=True)
 layouts.append(layout)
 rows.append({'rank':first['rank'],'cohort':first['local_constructor_cohort_seq'],
              'source':str(f),'storage_ptrs':[g['storage_ptr'] for g in normalized],
              'layout_sha256':hashlib.sha256(layout.encode()).hexdigest(),
              'total_unique_backing_bytes':sum(g['storage_nbytes'] for g in normalized)})
patterns={}
for key in layouts:
 patterns.setdefault(key,0);patterns[key]+=1
summary={'rank_cohort_files':len(rows),'unique_storage_layout_patterns':len(patterns),
         'pattern_counts':list(patterns.values()),
         'total_unique_backing_bytes_per_rank_cohort':sorted(set(r['total_unique_backing_bytes'] for r in rows)),
         'representative_backings':representative if rows else [],
         'limits':'same storage indicates alias potential only; interval provenance, native write/read domains, last-writer order and cross-cycle liveness remain unknown'}
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps({'status':'read_only_alias_census','summary':summary,'rows':rows},indent=2)+'\n')
print(json.dumps(summary,indent=2))
