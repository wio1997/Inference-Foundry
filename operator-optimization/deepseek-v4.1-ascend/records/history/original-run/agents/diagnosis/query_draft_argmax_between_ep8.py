#!/usr/bin/env python3
"""Read four inter-ArgMax intervals per fixed EP8 window from frozen CSV."""
import argparse,csv,hashlib,json
from collections import Counter
from decimal import Decimal
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--task-root',required=True);p.add_argument('--trace-root',required=True);p.add_argument('--anchors-json',required=True);a=p.parse_args()
T,R=Path(a.task_root),Path(a.trace_root)
assert hashlib.sha256(a.anchors_json.encode()).hexdigest()=='6270f9cb785b75cdfbe2442498631a5b0933941b552257c5220046ac55eaf5ce'
x=json.loads(a.anchors_json);rank='dp1_pp0_tp0_dcp0_ep8_rank0_727_20261028020238580_ascend_pt';assert x['rank']==rank
manifest=T/'evidence/TRACE-EXPORT-SHA256.json'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
assert sha(manifest)=='45ac6c7d22cc20601df2592ce0d78406df24a612f4aa56045566354d9ff46139'
expected={r['rank']:r['kernel_csv_sha256'] for r in json.loads(manifest.read_text())['exports']};csv_path=R/rank/'ASCEND_PROFILER_OUTPUT/kernel_details.csv'
assert csv_path.stat().st_size<=50*1024*1024 and sha(csv_path)==x['csv_sha256']==expected[rank]
intervals=[]
for w in x['windows']:
 assert w['ordinal'] in (6,7) and len(w['strict_draft_brackets'])==3
 argmax=sorted([e for e in w['events'] if e['type']=='ArgMaxV2' and e['relation_to_strict_draft_brackets']==['after']*3],key=lambda e:e['start_rel_ns'])
 assert len(argmax)==5 and all(e['complete_in_window'] and e['stream']=='47' for e in argmax)
 for i,(prev,nxt) in enumerate(zip(argmax,argmax[1:]),1):
  start=w['start_ns']+prev['end_rel_ns'];end=w['start_ns']+nxt['start_rel_ns'];assert start<end
  intervals.append({'ordinal':w['ordinal'],'gap_index':i,'start_ns':start,'end_ns':end,'main_stream':'47',
    'prev_argmax':{'task_id':prev['task_id'],'start_ns':w['start_ns']+prev['start_rel_ns'],'end_ns':start},
    'next_argmax':{'task_id':nxt['task_id'],'start_ns':end,'end_ns':w['start_ns']+nxt['end_rel_ns']},'events':[]})
assert len(intervals)==8
anchors={}
for g in intervals:
 for edge in ('prev_argmax','next_argmax'):
  t=g[edge];anchors[(g['ordinal'],t['task_id'],t['start_ns'],t['end_ns'])]=[]
with csv_path.open(newline='') as f:
 for z in csv.DictReader(f):
  s=int(Decimal(z['Start Time(us)'].strip())*1000);e=s+int(Decimal(z['Duration(us)'].strip())*1000)
  if e<=s:continue
  if z['Type']=='ArgMaxV2' and z['Stream ID'].strip()=='47':
   for key,hits in anchors.items():
    if (z['Task ID'].strip(),s,e)==(key[1],key[2],key[3]):
     hits.append({'type':z['Type'],'stream':z['Stream ID'].strip(),'model':z['Model ID'].strip(),
                  'task_id':z['Task ID'].strip(),'start_ns':s,'end_ns':e})
  for g in intervals:
   if s>=g['end_ns'] or e<=g['start_ns']:continue
   stream=z['Stream ID'].strip();category='unknown' if not stream.isdigit() else 'main' if stream==g['main_stream'] else 'other_numeric'
   g['events'].append({'type':z['Type'],'stream':stream,'model':z['Model ID'].strip(),'task_id':z['Task ID'].strip(),'stream_category':category,
      'start_rel_ns':s-g['start_ns'],'end_rel_ns':e-g['start_ns'],
      'clipped_start_rel_ns':max(s,g['start_ns'])-g['start_ns'],'clipped_end_rel_ns':min(e,g['end_ns'])-g['start_ns'],
      'crosses_start':s<g['start_ns'],'crosses_end':e>g['end_ns']})
assert len(anchors)==10 and all(len(hits)==1 and hits[0]['model']=='4294967295' for hits in anchors.values())
def union(es):
 spans=sorted((z['clipped_start_rel_ns'],z['clipped_end_rel_ns']) for z in es);merged=[]
 for a,b in spans:
  if not merged or a>merged[-1][1]:merged.append([a,b])
  else:merged[-1][1]=max(merged[-1][1],b)
 return sum(b-a for a,b in merged)
for g in intervals:
 assert len(g['events'])<=80,'event cap exceeded; do not truncate'
 g['events'].sort(key=lambda z:(z['start_rel_ns'],z['end_rel_ns'],z['type']))
 g['gap_ns']=g['end_ns']-g['start_ns'];g['all_rows_union_ns']=union(g['events']);assert g['all_rows_union_ns']<=g['gap_ns']
 g['uncovered_by_csv_rows_ns']=g['gap_ns']-g['all_rows_union_ns']
 g['group_union_ns']={k:union([z for z in g['events'] if z['stream_category']==k]) for k in ('main','other_numeric','unknown')}
 g['type_counts']=dict(Counter(z['type'] for z in g['events']))
print(json.dumps({'scope':'EP8 two ordinal windows, four adjacent post-bracket ArgMax intervals each, positive-duration intersecting CSV rows only',
 'anchors_sha256':'6270f9cb785b75cdfbe2442498631a5b0933941b552257c5220046ac55eaf5ce','csv_sha256':expected[rank],
 'rank':rank,'matched_anchor_rows':[dict(ordinal=key[0],**hits[0]) for key,hits in sorted(anchors.items())],
 'intervals':intervals,'limits':['Clipped union is only coverage by exported CSV rows; unreported time is not established device idle or host delay.',
 'Rows can be duplicate/summary records and different stream categories can overlap; only all-row union avoids double counting.',
 'Same stream, temporal order or adjacent ArgMax does not bind data, token, request, phase, causality or K=5 sampling.',
 'Excluded zero-duration events, internal activities and other trace tables may contain more; no cross-rank timestamps, user content or NPU calls.']},ensure_ascii=False,indent=2))
