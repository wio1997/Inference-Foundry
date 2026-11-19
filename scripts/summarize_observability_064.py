#!/usr/bin/env python3
"""Aggregate startup observability flags and exact candidate log marker counts.

Only fixed public, previously redacted archives are read. Server-info responses
and service-log bytes are touched in memory but never emitted. A missing literal
is not proof a logging feature never ran or that any private archive is absent.
"""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/observability_064/summary.json'
INFO='evidence/matched_SG_G181/p/functional_PD_g182/startup_{}_server_info.json'
LOG='evidence/matched_SG_G181/128K_failed_warm/{}/capture/service_server.log'
KEYS={'enable_metrics':bool,'enable_request_time_stats_logging':bool,'enable_trace':bool,'export_metrics_to_file':bool,'log_requests':bool,'log_requests_level':int}


def need(v,m):
 if not v:raise ValueError(m)


def values(doc):
 found={key:[] for key in KEYS}
 def visit(x):
  if isinstance(x,dict):
   for k,v in x.items():
    if k in KEYS:need(type(v) is KEYS[k],'invalid field type');found[k].append(v)
    visit(v)
  elif isinstance(x,list):
   for v in x:visit(v)
 visit(doc);need(all(v and len(set(v))==1 for v in found.values()),'missing or contradictory flag');return {k:v[0] for k,v in found.items()}


def summarize():
 checks={}
 for line in (ROOT/'SHA256SUMS').read_text().splitlines():
  digest,name=line.split('  ',1);need(name not in checks,'duplicate SHA');checks[name]=digest
 def source(path):
  p=Path(path);need(not p.is_absolute() and '..' not in p.parts and p.as_posix() in checks,'unsafe source');f=ROOT/p;need(f.is_file() and not f.is_symlink(),'invalid source');before=f.stat();need(before.st_size<10000000,'source too large');raw=f.read_bytes();after=f.stat();need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'source changed');h=hashlib.sha256(raw).hexdigest();need(h==checks[p.as_posix()],'SHA mismatch');return raw,h
 roles=[]
 for code,port,label in (('p','2201','prefill'),('d','2202','decode')):
  raw,sh=source(INFO.format(code));flags=values(json.loads(raw));raw,lh=source(LOG.format(port));marker=sum(b'ReqTimeStats(' in line for line in raw.splitlines());roles.append({'role':label,'startup_server_info_sha256':sh,'startup_flags':flags,'later_published_service_log_sha256':lh,'candidate_exact_marker_line_count':marker})
 need(all(r['startup_flags']=={'enable_metrics':True,'enable_request_time_stats_logging':False,'enable_trace':False,'export_metrics_to_file':False,'log_requests':False,'log_requests_level':2} and r['candidate_exact_marker_line_count']==0 for r in roles),'observed values changed')
 return {'schema':'g181-startup-observability-scope-v1','roles':roles,'limits':['Historical startup snapshot is a point-in-time response; it does not prove effective configuration throughout formal runs or rule out dynamic changes','Later published logs have undergone redaction; no per-field original preservation or complete coverage of all historical artifacts is established','Marker is a case-sensitive literal from candidate code; zero matches cannot rule out other logging formats, files, rotation or unobserved tracing','Candidate stage logging and trace may expose request identities and payload-related metadata; do not enable/export without separate privacy review','No request-level linkage, causal bottleneck or new measurement is established']}


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--write',action='store_true');ap.add_argument('--check',action='store_true');a=ap.parse_args();doc=summarize();text=json.dumps(doc,ensure_ascii=False,indent=2)+'\n'
 if a.write:
  OUT.parent.mkdir(parents=True,exist_ok=True)
  with OUT.open('x',encoding='utf8') as f:f.write(text)
 if a.check:need(OUT.read_text(encoding='utf8')==text,'output mismatch')
 print(json.dumps({'status':'pass','roles':len(doc['roles']),'sha256':hashlib.sha256(text.encode()).hexdigest(),'wrote':a.write,'checked':a.check}))


if __name__=='__main__':
 try:main()
 except (OSError,ValueError,TypeError,KeyError,json.JSONDecodeError):raise SystemExit('FAILED_CLOSED: source or whitelist validation failed')
