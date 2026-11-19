#!/usr/bin/env python3
"""Summarize published, redacted D log into conditional numeric bins.

The log has no timezone. Formal-window alignment here is explicitly an
unverified UTC and cross-host-clock hypothesis, not a proven sample identity.
The redacted source differs from its original SHA. No raw lines, requests,
IDs, labels, absolute timestamps, or private paths are emitted.
"""
import argparse,collections,datetime,hashlib,json,math,re,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/decode_log_058/summary_v3.json'
LOG='evidence/matched_SG_G181/128K_failed_warm/2202/capture/service_server.log'
LOG_SHA='c26fe6559c4b6c406414f42c80e200347ac581c581d64101bc6d1b8e1d308efa'
ROOTS={32768:'evidence/matched_SG_G181/first_32K_formal/run',65536:'evidence/matched_SG_G181/64K/formal/run'}
TS=re.compile(r'\b(20\d{2}-\d\d-\d\d)[ T](\d\d:\d\d:\d\d)(?:[.,](\d{1,6}))?')
PREFIX=re.compile(r'^\[(20\d{2}-\d\d-\d\d)[ T](\d\d:\d\d:\d\d)(?:[.,](\d{1,6}))? TP0 EP0\]\s+Decode batch\b')
PATTERNS={k:re.compile(v+r'(?=\s*(?:,|$))') for k,v in {'batch_running':r'#running-req:\s*(\d+)','full_tokens':r'#full token:\s*(\d+)','full_usage':r'full token usage:\s*(\d+\.\d{2})','prealloc_queue':r'#prealloc-req:\s*(\d+)','transfer_queue':r'#transfer-req:\s*(\d+)','mamba_num':r'mamba num:\s*(\d+)','ordinary_queue':r'#queue-req:\s*(\d+)'}.items()}


def need(v,m):
 if not v:raise ValueError(m)


def load_verified(path,hashes):
 p=Path(path);need(not p.is_absolute() and '..' not in p.parts and p.as_posix() in hashes,'unsafe input');data=(ROOT/p).read_bytes();h=hashlib.sha256(data).hexdigest();need(h==hashes[p.as_posix()],'SHA mismatch');return json.loads(data),h


def quantile(xs,q):
 xs=sorted(xs);need(xs and 0<=q<=1,'bad quantile');i=(len(xs)-1)*q;lo=math.floor(i);hi=math.ceil(i);return xs[lo]+(xs[hi]-xs[lo])*(i-lo)


def in_window(t,start,end):
 return start<=t<end


def metrics(samples):
 if not samples:return None
 return {name:{'min':min(s[name] for s in samples),'p50':quantile([s[name] for s in samples],.5),'max':max(s[name] for s in samples)} for name in PATTERNS}


def parse_line(line):
 pos=line.find('Decode batch')
 if pos<0:return None
 m=PREFIX.match(line);need(m is not None,'unsupported decode log prefix or precision')
 t=datetime.datetime.fromisoformat(m.group(1)+'T'+m.group(2)+'.'+(m.group(3) or '0').ljust(6,'0'))
 result={}
 for name,rx in PATTERNS.items():
  found=list(rx.finditer(line,pos));need(len(found)==1,'missing or duplicate expected log field');v=float(found[0].group(1));need(math.isfinite(v) and v>=0,'invalid log field');result[name]=v
 need(result['full_usage']<=1.01,'full usage range');return t,result


def summarize():
 hashes={}
 for line in (ROOT/'SHA256SUMS').read_text().splitlines():
  h,n=line.split('  ',1);need(n not in hashes,'duplicate SHA record');hashes[n]=h
 need(hashes.get(LOG)==LOG_SHA,'log not pinned');p=ROOT/LOG;need(p.is_file() and not p.is_symlink(),'unsafe log source');before=p.stat();raw=p.read_bytes();need(hashlib.sha256(raw).hexdigest()==LOG_SHA,'published log changed');after=p.stat();need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'log changed while reading')
 prov,_=load_verified('provenance/artifacts.json',hashes);matches=[x for x in prov if x.get('artifact')==LOG];need(len(matches)==1 and matches[0]['original_sha256']!=matches[0]['published_sha256'] and matches[0]['published_sha256']==LOG_SHA,'redaction provenance mismatch')
 windows={}
 for size,base in ROOTS.items():
  for ph in ('cold','hot'):
   a,ah=load_verified(f'{base}/before_{ph}_{size}_server_snapshot.json',hashes);b,bh=load_verified(f'{base}/after_{ph}_{size}_server_snapshot.json',hashes);x=a['decode']['metrics'];y=b['decode']['metrics'];need(x['status']==y['status']==200,'bad metrics response');start=datetime.datetime.fromisoformat(x['at'].replace('Z','+00:00'));end=datetime.datetime.fromisoformat(y['at'].replace('Z','+00:00'));need(start.utcoffset() is not None and end.utcoffset() is not None and end>start,'bad window time');windows[(size,ph)]=(start,end,[ah,bh])
 allrows=[];seconds_precision=0
 for line in raw.decode('utf8',errors='strict').splitlines():
  row=parse_line(line)
  if row is not None:allrows.append(row);seconds_precision+=PREFIX.match(line).group(3) is None
 need(allrows,'no decode rows');first_time=min(t for t,_ in allrows);last_time=max(t for t,_ in allrows);results=[]
 readiness,rh=load_verified('evidence/matched_SG_G181/d/readiness.json',hashes);capture,ch=load_verified('evidence/matched_SG_G181/128K_failed_warm/2202/capture/capture.json',hashes)
 launch=datetime.datetime.fromisoformat(readiness['actual_launch_handle']['at'].replace('Z','+00:00'));end_capture=datetime.datetime.fromisoformat(capture['at'].replace('Z','+00:00'));need(launch.utcoffset() is not None and end_capture.utcoffset() is not None,'reference timezone unknown')
 time_hypotheses=[]
 for off in (0,8):
  t0=first_time.replace(tzinfo=datetime.timezone(datetime.timedelta(hours=off)));t1=last_time.replace(tzinfo=datetime.timezone(datetime.timedelta(hours=off)))
  time_hypotheses.append({'assumed_log_utc_offset_hours':off,'first_selected_decode_line_minus_launch_receipt_s':(t0-launch).total_seconds(),'capture_receipt_minus_last_selected_decode_line_s':(end_capture-t1).total_seconds()})
 for (size,phase),(start,end,src) in windows.items():
  rows=[];alt=0
  for naive,values in allrows:
   t=naive.replace(tzinfo=datetime.timezone.utc)
   if in_window(t,start,end):rows.append((t,values))
   alt_t=naive.replace(tzinfo=datetime.timezone(datetime.timedelta(hours=8))).astimezone(datetime.timezone.utc)
   if in_window(alt_t,start,end):alt+=1
  rows.sort(key=lambda z:z[0]);bins=[];suppressed=0;span=(end-start).total_seconds()
  for i in range(math.ceil(span/60)):
   batch=[v for t,v in rows if i*60<=(t-start).total_seconds()<(i+1)*60]
   if 0<len(batch)<3:suppressed+=len(batch)
   elif batch:bins.append({'offset_start_s':60*i,'offset_end_s':min(60*(i+1),span),'sample_count':len(batch),'full_usage_p50':quantile([v['full_usage'] for v in batch],.5),'prealloc_queue_p50':quantile([v['prealloc_queue'] for v in batch],.5),'prealloc_queue_max':max(v['prealloc_queue'] for v in batch),'same_line_joint_count':sum(v['full_usage']>=.90 and v['prealloc_queue']>0 for v in batch),'batch_running_p50':quantile([v['batch_running'] for v in batch],.5)})
  need(suppressed==0,'small-bin privacy: widen bin and rebuild all aggregates')
  values=[v for _,v in rows];gaps=[(b[0]-a[0]).total_seconds() for a,b in zip(rows,rows[1:])];both=sum(v['full_usage']>=.90 and v['prealloc_queue']>0 for v in values);ratio_consistent=sum(abs(v['full_tokens']/345856-v['full_usage'])<=.0051 for v in values)
  results.append({'input_tokens':size,'phase':phase,'assumed_log_timezone':'UTC+00:00','alternative_utc_plus_8_count':alt,'snapshot_sha256':src,'enclosing_span_s':span,'decode_log_sample_count':len(values),'sample_stats':metrics(values),'adjacent_log_gap_s':None if not gaps else {'min':min(gaps),'p50':quantile(gaps,.5),'max':max(gaps)},'displayed_full_ge_0_90_count':sum(v['full_usage']>=.90 for v in values),'prealloc_positive_count':sum(v['prealloc_queue']>0 for v in values),'same_line_full_ge_0_90_and_prealloc_positive':both,'same_line_full_eq_0_90_and_prealloc_positive':sum(v['full_usage']==.90 and v['prealloc_queue']>0 for v in values),'ratio_consistent_with_345856_at_2dp':ratio_consistent,'bin_width_s':60,'suppressed_small_bin_sample_count':suppressed,'bins_60s':bins})
 return {'schema':'glm53-sg-d-published-log-conditional-windows-v2','published_log_sha256':LOG_SHA,'source_log_published_differs_from_original':True,'source_log_decode_lines':len(allrows),'seconds_precision_line_count':seconds_precision,'reference_receipt_sha256':[rh,ch],'timezone_hypotheses':time_hypotheses,'results':results,'limitations':['Published text is redacted and not byte-identical to original; per-field redaction preservation is not independently proven','Log has no timezone; UTC and alignment with scrape controller clock are hypotheses, not calibrated or identified to individual requests','All observed selected line timestamps are integer seconds; quantization convention and cross-clock error are unknown','TP0 EP0 appear in prefix; PP0 or unique process identity is not thereby established','Rows are emitted on decode batches, nonuniform and not time-weighted; absent bins are missing observations, not zero','Usage is a two-decimal displayed field; p50 is a statistic over rounded values, not true hardware precision','One log line reports full usage and prealloc count together; neither proves an admission decision, allocated request identity, actual evictability or causal full-KV bottleneck','Input length is inferred only from assumed scrape window; includes possible old/probe/other events','No cross-framework paired window, network attribution or TTFT contribution']}


def checks(doc):
 need(len(doc['results'])==4 and doc['source_log_decode_lines']==1188 and doc['seconds_precision_line_count']==1188,'log coverage or precision')
 expected={(32768,'cold'):(66,45),(32768,'hot'):(66,47),(65536,'cold'):(157,134),(65536,'hot'):(156,133)}
 for x in doc['results']:
  key=(x['input_tokens'],x['phase']);need(key in expected and (x['decode_log_sample_count'],x['same_line_full_ge_0_90_and_prealloc_positive'])==expected[key],'conditional sample mismatch');need(sum(z['sample_count'] for z in x['bins_60s'])+x['suppressed_small_bin_sample_count']==x['decode_log_sample_count'],'bin coverage');need(all(z['sample_count']>=3 and 0<=z['same_line_joint_count']<=z['sample_count'] and z['offset_end_s']<=x['enclosing_span_s'] for z in x['bins_60s']),'bin bounds');need(x['same_line_full_ge_0_90_and_prealloc_positive']<=min(x['prealloc_positive_count'],x['displayed_full_ge_0_90_count'])<=x['decode_log_sample_count'],'joint bounds');need(x['alternative_utc_plus_8_count']==0,'alternative timezone check');need(x['ratio_consistent_with_345856_at_2dp']==x['decode_log_sample_count'],'usage basis check')


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--write',action='store_true');ap.add_argument('--check',action='store_true');args=ap.parse_args();d=summarize();checks(d);text=json.dumps(d,ensure_ascii=False,indent=2)+'\n'
 if args.write:
  OUT.parent.mkdir(parents=True,exist_ok=True)
  with OUT.open('x',encoding='utf8') as f:f.write(text)
 if args.check:need(OUT.read_text(encoding='utf8')==text,'output mismatch')
 print(json.dumps({'status':'pass','conditional_windows':len(d['results']),'sha256':hashlib.sha256(text.encode()).hexdigest(),'wrote':args.write,'checked':args.check}))


if __name__=='__main__':
 try:main()
 except (OSError,ValueError,TypeError,KeyError,UnicodeError,json.JSONDecodeError):raise SystemExit('FAILED_CLOSED: input, integrity, or parsing validation failed')
