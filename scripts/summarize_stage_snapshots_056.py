#!/usr/bin/env python3
"""Offline aggregate histograms from frozen SG formal/vLLM diagnostic scrapes.

Reads only allowlisted metric names/labels; no raw response, model name, IDs,
absolute timestamps, request/response contents, or private paths are emitted.
"""
import argparse,datetime,hashlib,json,math,re
from decimal import Decimal,InvalidOperation
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/stage_snapshots_056/summary_v7.json'
SG_ROOTS={32768:'evidence/matched_SG_G181/first_32K_formal/run',65536:'evidence/matched_SG_G181/64K/formal/run'}
V_ROOT='evidence/G4_profile_G138/job/capture_control'
SG_BASES=('sglang:per_stage_req_latency_seconds','sglang:kv_transfer_bootstrap_ms','sglang:kv_transfer_alloc_ms','sglang:kv_transfer_latency_ms','sglang:queue_time_seconds','sglang:e2e_request_latency_seconds','sglang:time_to_first_token_seconds','sglang:inter_token_latency_seconds','sglang:request_time_per_output_token_seconds')
SG_STAGES={'prefill_bootstrap','prefill_forward','prefill_transfer_kv_cache','chunked_prefill','decode_prepare','decode_bootstrap','decode_transferred','decode_waiting','fake_output'}
SG_RANK_BASES=set(SG_BASES[:5])
SG_GLOBAL_BASES=set(SG_BASES[5:])
V_BASES=('vllm:request_queue_time_seconds','vllm:request_prefill_time_seconds','vllm:time_to_first_token_seconds','vllm:request_inference_time_seconds','vllm:request_decode_time_seconds')
METRIC=re.compile(r'^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{(.*)\})?\s+(\S+)(?:\s+\S+)?$')
LABEL=re.compile(r'\s*([A-Za-z_][A-Za-z0-9_]*)="((?:\\.|[^"\\])*)"\s*(?:,|$)')


def need(v,m):
 if not v:raise ValueError(m)


def labels(raw):
 if not raw:return {}
 pos=0;out={}
 while pos<len(raw):
  m=LABEL.match(raw,pos);need(m is not None,'malformed label');key,val=m.groups();need(key not in out,'duplicate label');out[key]=val;pos=m.end()
 return out


def val(s):
 try:x=Decimal(s)
 except InvalidOperation:raise ValueError('invalid metric value') from None
 need(x.is_finite(),'non-finite metric');return x


def parse(body,bases,role,sg):
 out={};process=[]
 for line in body.splitlines():
  if line.startswith('#'):continue
  m=METRIC.match(line)
  if not m:continue
  name,raw,v=m.groups()
  if name=='process_start_time_seconds':process.append(val(v));continue
  kind=next((k for k in ('_created','_count','_sum','_bucket') if name.endswith(k) and name[:-len(k)] in bases),None)
  if not kind:continue
  base=name[:-len(kind)];ls=labels(raw);stage=ls.get('stage','none')
  if sg:
   if ls.get('engine_type')!=role:continue
   if base in SG_RANK_BASES:
    if any(ls.get(k)!='0' for k in ('tp_rank','moe_ep_rank','pp_rank')):continue
    if base=='sglang:per_stage_req_latency_seconds' and stage not in SG_STAGES:continue
    if base!='sglang:per_stage_req_latency_seconds' and stage!='none':continue
   elif base in SG_GLOBAL_BASES:
    need(not any(k in ls for k in ('tp_rank','moe_ep_rank','pp_rank')),'unexpected global rank labels')
    if base!='sglang:inter_token_latency_seconds' and ls.get('is_streaming')!='true':continue
    if base=='sglang:inter_token_latency_seconds':need('is_streaming' not in ls,'unexpected global stream label')
   else:continue
  elif stage!='none':continue
  bound=ls.pop('le',None);identity=(base,stage,tuple(sorted(ls.items())));obj=out.setdefault(identity,{'buckets':{}})
  if kind=='_bucket':
   need(bound is not None and bound not in obj['buckets'],'invalid bucket');need(bound=='+Inf' or val(bound)>=0,'invalid bound');obj['buckets'][bound]=val(v)
  else:need(bound is None and kind not in obj,'duplicate aggregate');obj[kind]=val(v)
 need(len(process)<=1,'process clock not unique')
 return out,process[0] if process else None


def between(a,b,bases,role,sg):
 x,xp=parse(a,bases,role,sg);y,yp=parse(b,bases,role,sg);need(set(x)==set(y) and x,'counter label sets differ');output=[];seen=set()
 for key in sorted(x):
  base,stage,_=key;need((base,stage) not in seen,'more than one rank0 target series');seen.add((base,stage));u=x[key];v=y[key]
  need(all(t in u and t in v for t in ('_sum','_count')) and set(u['buckets'])==set(v['buckets']) and '+Inf' in u['buckets'],'incomplete histogram')
  count=v['_count']-u['_count'];total=v['_sum']-u['_sum'];need(count>=0 and count==count.to_integral_value() and total>=Decimal('-0.000001'),'negative counter delta');need(count!=0 or abs(total)<Decimal('0.000001'),'sum changed without count')
  ub=u.get('_created');vb=v.get('_created');need((ub is None)==(vb is None),'created presence mismatch');created='not_recorded' if ub is None else ('same_at_endpoints' if ub==vb else 'changed_at_endpoints');need(created!='changed_at_endpoints','created changed')
  bounds=sorted(u['buckets'],key=lambda z:Decimal('Infinity') if z=='+Inf' else val(z));bucket=[];last=Decimal(0);unit=Decimal(1000) if base.endswith('_ms') else Decimal(1)
  for bound in bounds:
   n=v['buckets'][bound]-u['buckets'][bound];need(n>=last and n==n.to_integral_value() and n<=count,'invalid cumulative bucket delta');last=n;bucket.append({'upper_s':None if bound=='+Inf' else float(val(bound)/unit),'cumulative_count':int(n)})
  need(last==count,'bucket/count mismatch');qs={}
  for pct in (50,90,99):
   target=math.ceil(int(count)*pct/100)
   if not target:qs[str(pct)]=None;continue
   for i,z in enumerate(bucket):
    if z['cumulative_count']>=target:
     qs[str(pct)]={'lower_exclusive_s':bucket[i-1]['upper_s'] if i else None,'upper_inclusive_s':z['upper_s']};break
  output.append({'metric':base,'stage':stage,'metric_scope':('global_streaming' if base in SG_GLOBAL_BASES and base!='sglang:inter_token_latency_seconds' else 'global_unspecified_stream' if base=='sglang:inter_token_latency_seconds' else 'rank0' if sg else 'endpoint_aggregate'),'delta_count':int(count),'delta_sum_s':float(total/unit),'mean_s':float(total/unit/count) if count else None,'created_check':created,'buckets':bucket,'nearest_rank_quantile_bounds':qs})
 bootstrap_stage='prefill_bootstrap' if role=='prefill' else 'decode_bootstrap'
 target={(base,stage):tuple((k,v) for k,v in ls if k!='stage') for base,stage,ls in x if (base,stage) in {('sglang:per_stage_req_latency_seconds',bootstrap_stage),('sglang:kv_transfer_alloc_ms','none'),('sglang:kv_transfer_bootstrap_ms','none')}}
 same_nonstage=(len(target)==3 and len(set(target.values()))==1) if sg else None
 process='not_recorded' if xp is None and yp is None else 'same_at_endpoints' if xp is not None and xp==yp else 'changed_or_incomplete'
 need(process!='changed_or_incomplete','process start changed');return output,process,same_nonstage


def source(path,manifest):
 p=Path(path);need(not p.is_absolute() and '..' not in p.parts and p.as_posix() in manifest,'unsafe source');b=(ROOT/p).read_bytes();sha=hashlib.sha256(b).hexdigest();need(sha==manifest[p.as_posix()],'source SHA mismatch');return json.loads(b),sha


def duration(t0,t1):
 a=datetime.datetime.fromisoformat(t0.replace('Z','+00:00'));b=datetime.datetime.fromisoformat(t1.replace('Z','+00:00'));need(a.utcoffset() is not None and b.utcoffset() is not None and b>=a,'invalid scrape time');return (b-a).total_seconds()


def summarize():
 manifest={}
 for line in (ROOT/'SHA256SUMS').read_text().splitlines():
  sha,path=line.split('  ',1);need(path not in manifest,'duplicate manifest key');manifest[path]=sha
 out=[]
 for size,sg_root in SG_ROOTS.items():
  for phase in ('cold','hot','between'):
   start=f'after_cold_{size}_server_snapshot.json' if phase=='between' else f'before_{phase}_{size}_server_snapshot.json'
   end=f'before_hot_{size}_server_snapshot.json' if phase=='between' else f'after_{phase}_{size}_server_snapshot.json'
   before,bs=source(f'{sg_root}/{start}',manifest);after,as_=source(f'{sg_root}/{end}',manifest)
   for role in ('prefill','decode'):
    a=before[role]['metrics'];b=after[role]['metrics'];need(a['status']==b['status']==200,'SG scrape failed');rows,proc,labels_same=between(a['body'],b['body'],SG_BASES,role,True)
    out.append({'framework':'sglang','input_tokens':size,'capture':'between_formal' if phase=='between' else 'formal','phase':phase,'role':role,'rank_scope':'selected histograms individually label rank0 or global streaming','nominal_scored_requests':None if phase=='between' else 32,'snapshot_elapsed_s':duration(a['at'],b['at']),'source_sha256':[bs,as_],'process_start_check':proc,'bootstrap_nonstage_labels_same':labels_same,'histograms':rows})
 for role in ('p','d'):
  before,bs=source(f'{V_ROOT}/{role}_metrics_before.json',manifest);after,as_=source(f'{V_ROOT}/{role}_metrics_after.json',manifest);need(before['status']==after['status']==200 and before['url']==after['url'],'vLLM scrape failed or endpoint changed');rows,proc,_=between(before['body'],after['body'],V_BASES,role,False)
  out.append({'framework':'vllm','input_tokens':32768,'capture':'diagnostic','phase':'profile','role':'prefill' if role=='p' else 'decode','rank_scope':'endpoint_aggregate','nominal_diagnostic_requests':16,'snapshot_elapsed_s':duration(before['utc'],after['utc']),'source_sha256':[bs,as_],'process_start_check':proc,'histograms':rows})
 return {'schema':'glm53-stage-snapshot-deltas-v1','scopes':out,'limitations':['SG formal scrape counts differ from 32 scored client requests; individual sample identities and background traffic unknown','SG G181 runtime timer definitions and process/counter reset status not SHA-bound; created/process metrics absent','vLLM G138 metrics are a separate non-scored 16-request profiling window, not the vLLM formal run','Per-role/stage histograms may overlap; labels/counts do not establish common samples, request pairing, or synchronized P/D clocks','Histogram bounds use nearest-rank order statistic ceil(p*n); open lower/closed upper when present, not exact linear-interpolated quantiles; first lower and +Inf upper are unknown','Histogram bounds rely on no-reset assumptions','No cross-framework phase deltas, P+D sums, wire-time attribution, causality or E2E benefit inferred']}


def checks(d):
 need(len(d['scopes'])==14,'scope coverage')
 for s in d['scopes']:
  need(s['snapshot_elapsed_s']>0 and len(s['histograms'])>0,'empty scope')
  for x in s['histograms']:need(x['delta_count']>=0 and (x['mean_s'] is None)==(x['delta_count']==0),'counter invariant')
 def get(f,c,p,r,stage):
  s=next(x for x in d['scopes'] if x['framework']==f and x['input_tokens']==32768 and x['capture']==c and x['phase']==p and x['role']==r);return next(x for x in s['histograms'] if x['stage']==stage and 'per_stage_req_latency' in x['metric'])
 cold=get('sglang','formal','cold','decode','decode_bootstrap');need(cold['delta_count']==33,'SG cold stage count')
 threshold=next(z for z in cold['buckets'] if z['upper_s'] is not None and abs(z['upper_s']-25.10818416014034)<1e-8);need(threshold['cumulative_count']==11,'SG cold stage bucket')
 need(get('sglang','formal','hot','decode','decode_bootstrap')['delta_count']==35,'SG hot stage count')
 decode_scope=next(x for x in d['scopes'] if x['framework']=='sglang' and x['input_tokens']==32768 and x['phase']=='cold' and x['role']=='decode');alloc=next(x for x in decode_scope['histograms'] if x['metric']=='sglang:kv_transfer_alloc_ms');finite=next(z for z in alloc['buckets'] if z['upper_s']==2.5);need(alloc['delta_count']==33 and finite['cumulative_count']==11,'SG cold alloc bucket')
 for role,phase,expected in (('prefill','cold',35),('decode','cold',35),('prefill','hot',34),('decode','hot',33)):
  scope=next(x for x in d['scopes'] if x['framework']=='sglang' and x['input_tokens']==65536 and x['phase']==phase and x['role']==role);stage='prefill_bootstrap' if role=='prefill' else 'decode_bootstrap';row=next(x for x in scope['histograms'] if x['stage']==stage);need(row['delta_count']==expected,'SG 64K stage count')
 for s in d['scopes']:
  if s['framework']=='vllm':need(s['process_start_check']=='same_at_endpoints' and all(x['created_check']=='same_at_endpoints' and x['delta_count']==16 for x in s['histograms']),'vLLM endpoint stability')
  else:need(s['bootstrap_nonstage_labels_same'] is True,'SG non-stage labels differ')


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--write',action='store_true');ap.add_argument('--check',action='store_true');args=ap.parse_args();d=summarize();checks(d);text=json.dumps(d,ensure_ascii=False,indent=2)+'\n'
 if args.write:
  OUT.parent.mkdir(parents=True,exist_ok=True)
  with OUT.open('x',encoding='utf8') as f:f.write(text)
 if args.check:need(OUT.read_text(encoding='utf8')==text,'summary mismatch')
 print(json.dumps({'status':'pass','scopes':len(d['scopes']),'sha256':hashlib.sha256(text.encode()).hexdigest(),'wrote':args.write,'checked':args.check}))


if __name__=='__main__':
 try:main()
 except (OSError,ValueError,TypeError,KeyError,InvalidOperation,StopIteration,json.JSONDecodeError) as e:raise SystemExit('FAILED: '+str(e))
