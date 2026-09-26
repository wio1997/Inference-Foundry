#!/usr/bin/env python3
"""Summarize Run285 eager CP-fork same-prestate raw numerical screen."""
import argparse,json
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True);a=ap.parse_args()
rows=[]
for rank in range(8):
 p=a.run_dir/'pages'/f'rank{rank}.json'
 if not p.exists():
  rows.append({'rank':rank,'status':'missing'});continue
 d=json.loads(p.read_text());events=d['rows'];x=events[0].get('cp_fork_parity') if events else None
 if not x:
  rows.append({'rank':rank,'status':'missing_cp_fork_parity','cycles':len(events)});continue
 row={'rank':rank,'status':'present','modes':x['modes'],'cache_views':x['cache_views']}
 for key in ('a0_repeat_0_2','a0_repeat_0_4','overlap_vs_a0','immediate_vs_a0','overlap_vs_immediate'):
  c=x[key]
  row[key]={'argmax_equal':c['argmax_equal'],'argmax_total':c['argmax_total'],
   'counts_exact':c['counts']['exact'],'tokens_exact':c['tokens']['exact'],
   'logits':{k:c['logits'][k] for k in ('exact','different','max_abs')},
   'hidden':{k:c['hidden'][k] for k in ('exact','different','max_abs')},
   'aux_max_abs':max((v['max_abs'] or 0) for v in c['aux_hidden']) if c['aux_hidden'] else None,
   'cache_diffs':[{'name':v['name'],'different':v['different'],'max_abs':v['max_abs']} for v in c['post_cache'] if not v['exact']]}
 rows.append(row)
valid=[r for r in rows if r['status']=='present']
out={'run':'Run285','status':'raw_same_state_screen_not_formal','rank_count':len(valid),'rows':rows,
 'a0_repeat_integer_stable':all(r[k]['counts_exact'] and r[k]['tokens_exact'] and r[k]['argmax_equal']==r[k]['argmax_total'] for r in valid for k in ('a0_repeat_0_2','a0_repeat_0_4')) if len(valid)==8 else None,
 'candidate_integer_parity':all(r['overlap_vs_a0']['counts_exact'] and r['overlap_vs_a0']['tokens_exact'] and r['overlap_vs_a0']['argmax_equal']==r['overlap_vs_a0']['argmax_total'] for r in valid) if len(valid)==8 else None,
 'interpretation':'No automatic tolerance inferred from these summaries. Inspect off self-noise, all tensor differences, mode branch evidence, and Graph closure before promotion.'}
(a.run_dir/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
