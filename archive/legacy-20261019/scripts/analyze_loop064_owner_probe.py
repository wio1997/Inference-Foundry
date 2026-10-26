#!/usr/bin/env python3
"""Summarize original-path layer2 request ownership and physical-page envelope."""
import argparse,json,collections
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True);a=ap.parse_args()
files=sorted((a.run_dir/'ownership').glob('rank*_cohort*.jsonl'))
summary=[]
for p in files:
 rows=[json.loads(x) for x in p.read_text().splitlines() if x]
 if not rows:continue
 rank=rows[0]['rank']; cohort=rows[0]['local_constructor_cohort_seq']
 assert all(r['rank']==rank and r['local_constructor_cohort_seq']==cohort for r in rows)
 assert [r['cycle'] for r in rows]==list(range(len(rows)))
 geometry=collections.Counter((tuple(r['global_qsl']),tuple(r['local_qsl']),tuple(r['owner_request_slots']),r['owner_full_update_rows']) for r in rows)
 changes=collections.Counter()
 page_data={}
 compressed_intersections=collections.Counter()
 derived_output_rows=collections.Counter()
 first_storage=rows[0].get('layer2_cache_views')
 for r in rows:
  start=r['start_pos'];qsl=r['global_qsl'];owners=set(r['owner_request_slots'])
  for f in r['physical_frontiers']:
   layer=f['layer'];changes[layer]+=int(f['changed'])
   if 'physical_pages_by_request' in f:
    page_data[layer]=f['physical_pages_by_request']
   if layer.endswith('.attn') or layer.endswith('indexer.k_cache'):
    table=page_data[layer]
    block=f['block_size'];ratio=f['ratio']
    owner_history=set()
    nonowner_new=set()
    for entry in table:
     i=entry['request_slot'];lo,hi=entry['logical_interval'];pages=entry['physical_pages']
     valid={v for v in pages if v>=0}
     if i in owners:owner_history|=valid
     else:
      first=start[i]//ratio;end=(start[i]+qsl[i+1]-qsl[i])//ratio
      derived_output_rows[(layer,i,end-first)]+=1
      for j in range(first,end):
       logical=j//block
       if lo<=logical<hi:nonowner_new.add(pages[logical-lo])
    compressed_intersections[(layer, bool(owner_history & nonowner_new))]+=1
 summary.append({'rank':rank,'local_constructor_cohort_seq':cohort,'cycles':len(rows),
  'geometry_signatures':[{'count':v,'global_qsl':list(k[0]),'local_qsl':list(k[1]),'owners':list(k[2]),'owner_full_update_rows':k[3]} for k,v in geometry.items()],
  'owner_sets':sorted({tuple(r['owner_request_slots']) for r in rows}),
  'active_mask_variants':sorted({tuple(r['active_mask']) for r in rows}),
  'max_emitted_by_slot':[max(r['emitted_token_count'][i] for r in rows) for i in range(12)],
  'frontier_snapshot_changes':dict(changes),
  'compressed_owner_history_nonowner_new_page_intersection':[
   {'layer':layer,'intersects':flag,'cycles':v} for (layer,flag),v in compressed_intersections.items()],
  'derived_nonowner_compressed_output_rows_histogram':[
   {'layer':layer,'request_slot':i,'rows':cnt,'cycles':v} for (layer,i,cnt),v in derived_output_rows.items()],
  'layer2_cache_views':first_storage})
out={'status':'read_only_original_path_geometry_and_page_envelope','rank_cohort_files':len(files),
 'rank_cohort_rows':len(summary),'summaries':summary,
 'limits':['local constructor cohort sequence is not Runtime cohort identity','compressed page sets conservatively include partial/unwritten prefix pages','derived compressed output rows/physical pages are not native scatter proof','state historical recursive read scope unknown','DSpark, prefix-refcount, connector and next-cohort consumers unproven','probe forces Host synchronizations and yields no performance number']}
(a.run_dir/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'rank_cohort_rows':len(summary),'cycles_by_rank_cohort':[(x['rank'],x['local_constructor_cohort_seq'],x['cycles']) for x in summary],
 'geometry_violations':[x for x in summary if len(x['geometry_signatures'])!=1]},indent=2))
