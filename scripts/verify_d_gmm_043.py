#!/usr/bin/env python3
"""Validate recorded GMM sequence slices and counters, not device causality."""
import gzip,hashlib,json,math,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FILES={'counter':('counters.json.gz','e50dead51a02affaff6870e9a4eb6a64896cb290dc0567f8957d93828e78f811'),
       'temporal':('temporal.json.gz','53a1cb239d6d9ae56380bc2c802bc7d878d4d633e17c7749ee05bdfccb9bc6f0')}


def need(ok,msg):
 if not ok:raise ValueError(msg)


def load(kind):
 name,digest=FILES[kind];raw=gzip.decompress((ROOT/'evidence/d_gmm_043'/name).read_bytes());need(hashlib.sha256(raw).hexdigest()==digest,'raw SHA mismatch')
 text=raw.decode();need(not re.search(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])',text),'address-like data');need(not re.search(r'(?<!\w)/(?:home|root|data|data1|task|tmp|models)(?:/|\b)',text),'private path')
 return json.loads(text)


def verify():
 t=load('temporal');c=load('counter');r={x['rank']:x for x in t['ranks']};need(set(r)==set(range(8)) and t['rank_count']==8,'temporal ranks')
 slots={}
 for rank,row in r.items():
  need(row['engine']=='sg' and row['csv_expected_sha256']==row['csv_actual_sha256'] and row['frozen_sha_size_mtime_match'] is True and row['stable_during_hash_and_parse'] is True,'source pin')
  need(row['csv_rows_scanned']==45016 and row['selected_rows']==720 and len(row['results'])==2,'row count')
  for x in row['results']:
   prefix=x['input_shape'].split(';')[0];need(prefix in ('8192,2048','8192,4096') and x['event_count']==360 and len(x['slices'])==6,'signature')
   need(all(y['count']==60 for y in x['slices']),'slice size');slots[(rank,prefix)]=x
 wins={}
 for shape in ('8192,2048','8192,4096'):
  for metric in ('p50_us','p90_us'):
   n=sum(slots[(0,shape)]['slices'][i][metric]>slots[(rank,shape)]['slices'][i][metric] for rank in range(1,8) for i in range(6))
   need(n==42,'unexpected ordinal comparison');wins[shape+' '+metric]=n
 ranks={(x['engine'],x['rank']):x for x in c['ranks']};need(len(ranks)==len(c['ranks'])==16 and c['all_signature_counts_match'] is True,'counter coverage')
 for key,row in ranks.items():
  need(len(row['signatures'])==2 and row['csv_actual_sha256']==row['csv_expected_sha256'] and row['count_validation']=='pass','counter source')
  for x in row['signatures']:
   o=x['counter_observations'];need(o['rows']==360,'counter count')
   for m in o['all_events'].values():need(m['available']+m['unavailable']==360,'missing counter coverage')
 r0=next(x for x in ranks[('sg',0)]['signatures'] if x['input_shape'].startswith('8192,2048;'))['counter_observations']['all_events']
 need(abs(r0['Wait Time(us)']['p50']-1.2)<1e-9 and abs(r0['aicore_time(us)']['p50']-60.3285)<1e-6,'reference counter mismatch')
 return {'temporal_ranks':8,'slices':96,'wins_of_42':wins,'counter_ranks':16,'rank0_wait_p50_us':1.2,'rank0_aicore_p50_us':60.3285,'scope':'independent ordinal slices and uncalibrated CSV fields; not same step or root cause'}


if __name__=='__main__':
 try:print(json.dumps(verify(),ensure_ascii=False,indent=2))
 except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError) as e:raise SystemExit('FAILED: '+str(e))
