#!/usr/bin/env python3
"""Summarize historical KV gauges at formal-run scrape endpoints only.

Offline; output contains no raw HTTP body, labels, model name, request text,
identity, private location or absolute timestamp. Conditional capacity arithmetic
is not an admission or active-request measurement.
"""
import argparse,hashlib,json,re
from decimal import Decimal,InvalidOperation
from pathlib import Path
from summarize_stage_snapshots_056 import labels as parse_labels
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/kv_endpoints_057/summary_v4.json'
STARTUP='evidence/matched_SG_G181/p/functional_PD_g182/startup_d_server_info.json'
STARTUP_KEYS=('max_total_num_tokens','max_running_requests','page_size','num_reserved_decode_tokens','context_length','disable_radix_cache','disaggregation_decode_enable_radix_cache','enable_hisparse','disaggregation_decode_host_receive_threshold')
ROOTS={32768:'evidence/matched_SG_G181/first_32K_formal/run',65536:'evidence/matched_SG_G181/64K/formal/run'}
GAUGES=('max_total_num_tokens','kv_available_tokens','kv_evictable_tokens','kv_used_tokens','num_used_tokens','full_token_usage','token_usage','prealloc_full_token_usage','pending_prealloc_token_usage','mamba_available_tokens','mamba_evictable_tokens','mamba_used_tokens','num_running_reqs','num_queue_reqs','num_decode_prealloc_queue_reqs','num_prefill_bootstrap_queue_reqs','num_decode_transfer_queue_reqs')
LINE=re.compile(r'^sglang:([A-Za-z0-9_]+)\{([^}]*)\}\s+(\S+)(?:\s+\S+)?$')


def need(x,m):
 if not x:raise ValueError(m)


def series(body,role):
 result={};identity={}
 for line in body.splitlines():
  m=LINE.match(line)
  if not m or m.group(1) not in GAUGES:continue
  name,raw,value=m.groups();ls=parse_labels(raw)
  if ls.get('engine_type')!=role or any(ls.get(k)!='0' for k in ('tp_rank','moe_ep_rank','pp_rank')):continue
  need(name not in result,'gauge not unique')
  try:v=Decimal(value)
  except InvalidOperation:raise ValueError('invalid gauge') from None
  need(v.is_finite() and v>=0,'non-finite or negative gauge');result[name]=float(v);identity[name]=tuple(sorted(ls.items()))
 need(set(result)==set(GAUGES),'gauge missing');return result,identity


def source(name,hashes):
 p=Path(name);need(not p.is_absolute() and '..' not in p.parts and p.as_posix() in hashes,'unsafe path');raw=(ROOT/p).read_bytes();sha=hashlib.sha256(raw).hexdigest();need(sha==hashes[p.as_posix()],'source SHA mismatch');return json.loads(raw),sha


def summarize():
 hashes={}
 for line in (ROOT/'SHA256SUMS').read_text().splitlines():
  h,name=line.split('  ',1);need(name not in hashes,'duplicate checksum');hashes[name]=h
 startup,startup_sha=source(STARTUP,hashes);found={k:[] for k in STARTUP_KEYS}
 def visit(x):
  if isinstance(x,dict):
   for k,v in x.items():
    if k in found and isinstance(v,(int,float,bool)):found[k].append(v)
    visit(v)
  elif isinstance(x,list):
   for v in x:visit(v)
 visit(startup);need(all(v and len(set(v))==1 for v in found.values()),'startup whitelist ambiguous');startup_values={k:v[0] for k,v in found.items()};need(startup_values['max_total_num_tokens']==345856 and startup_values['num_reserved_decode_tokens']==512 and startup_values['page_size']==128,'startup configuration mismatch')
 samples=[];models={}
 for size,directory in ROOTS.items():
  for phase in ('cold','hot'):
   for bound in ('before','after'):
    d,sha=source(f'{directory}/{bound}_{phase}_{size}_server_snapshot.json',hashes)
    for role in ('prefill','decode'):
     m=d[role]['metrics'];need(m['status']==200,'metrics HTTP not 200');values,identity=series(m['body'],role);key=(size,phase,role)
     if bound=='before':models[key]=identity
     else:need(models[key]==identity,'gauge label changed within endpoint pair')
     cap=int(values['max_total_num_tokens']);samples.append({'input_tokens':size,'phase':phase,'boundary':bound,'role':role,'gauge_scope':'tp0_ep0_pp0','source_sha256':sha,'gauges':values,'arithmetic_only':{'free_plus_evictable':values['kv_available_tokens']+values['kv_evictable_tokens'],'free_plus_evictable_plus_active':values['kv_available_tokens']+values['kv_evictable_tokens']+values['kv_used_tokens'],'ideal_full_unshared_input_slots_floor':cap//size,'input_tokens_for_16_simultaneous_full_unshared_requests':16*size,'reserve512_slots_if_same_pool':cap//(size+512) if role=='decode' else None,'input_plus_reserve512_for_16_if_same_pool':16*(size+512) if role=='decode' else None,'assumptions':'all requests require simultaneously resident, distinct full inputs in the same indicated pool; the separate +512 example additionally assumes each such request reserves 512 slots in this pool; ignores sharing, actual admission, layer pool mapping, outputs and other memory constraints'}})
 return {'schema':'glm53-formal-kv-endpoint-gauges-v2','startup_config':startup_values,'startup_source_sha256':startup_sha,'samples':samples,'limitations':['Instantaneous gauges at before/after HTTP scrape only, not in-run peaks or same time as histogram events','Gauge self-description of free/evictable/used does not prove their sum is a realizable allocatable budget for a particular request','Counterfactual context count assumes full distinct contexts simultaneously share the indicated max_total_num_tokens pool and no other reservations; C16 client concurrency does not establish this state','No same-request identity, runtime source binding, real prefix sharing/locks, evictability, full/SWA pool mapping, or hidden admission gates','No cross-framework causation, request wait, TTFT contribution or optimization benefit']}


def checks(d):
 need(len(d['samples'])==16,'endpoint coverage');seen=set()
 for s in d['samples']:
  k=(s['input_tokens'],s['phase'],s['boundary'],s['role']);need(k not in seen,'duplicate endpoint');seen.add(k);g=s['gauges'];a=s['arithmetic_only'];need(g['max_total_num_tokens']==(292864 if s['role']=='prefill' else 345856),'unexpected pool capacity')
  need(a['free_plus_evictable_plus_active']==g['max_total_num_tokens'],'endpoint arithmetic mismatch');need(g['num_running_reqs']==0 and g['num_decode_prealloc_queue_reqs']==0,'non-empty endpoint');need(a['ideal_full_unshared_input_slots_floor']<16,'ideal budget observation changed')


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--write',action='store_true');ap.add_argument('--check',action='store_true');a=ap.parse_args();d=summarize();checks(d);text=json.dumps(d,ensure_ascii=False,indent=2)+'\n'
 if a.write:
  OUT.parent.mkdir(parents=True,exist_ok=True)
  with OUT.open('x',encoding='utf8') as f:f.write(text)
 if a.check:need(OUT.read_text(encoding='utf8')==text,'summary mismatch')
 print(json.dumps({'status':'pass','endpoints':len(d['samples']),'sha256':hashlib.sha256(text.encode()).hexdigest(),'wrote':a.write,'checked':a.check}))


if __name__=='__main__':
 try:main()
 except (OSError,ValueError,TypeError,KeyError,json.JSONDecodeError) as e:raise SystemExit('FAILED: '+str(e))
