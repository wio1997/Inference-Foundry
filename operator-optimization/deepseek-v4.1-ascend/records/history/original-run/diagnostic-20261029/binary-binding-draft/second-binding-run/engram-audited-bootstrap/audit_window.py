"""Read-only bounded audit of one window; rank mapping supplied externally."""
import argparse,json,re
from pathlib import Path
MAX_LOG=1024*1024;MAX_STATUS=64;MAX_STATUS_BYTES=4096

def read_stable(p,limit):
 a=p.stat()
 if a.st_size>limit:raise ValueError('size_limit')
 data=p.read_bytes();b=p.stat()
 if (a.st_size,a.st_mtime_ns)!=(b.st_size,b.st_mtime_ns):raise ValueError('changed_during_read')
 return data

def audit(directory,manifest):
 directory=Path(directory);run=manifest.get('run_id');window=manifest.get('window_id');workers=manifest.get('workers')
 if not isinstance(run,str) or not re.fullmatch('[0-9a-f]{32}',run) or not isinstance(window,str) or not re.fullmatch('[0-9a-f]{32}',window) or not isinstance(workers,list) or not workers or len(workers)>MAX_STATUS:raise ValueError('bad_manifest')
 if any(type(x.get('pid')) is not int or type(x.get('rank')) is not int for x in workers):raise ValueError('bad_worker')
 if len({x['pid'] for x in workers})!=len(workers) or len({x['rank'] for x in workers})!=len(workers):raise ValueError('duplicate_mapping')
 files=[]
 with __import__('os').scandir(directory) as scan:
  for entry in scan:
   if re.fullmatch(r'observer-[0-9a-f]{24}\.json',entry.name):files.append(Path(entry.path))
   if len(files)>MAX_STATUS:raise ValueError('worker_status_cap')
 statuses=[]
 for f in files:
  v=json.loads(read_stable(f,MAX_STATUS_BYTES))
  if v.get('run_id')==run:statuses.append(v)
 log=directory/'engram-metadata.jsonl';problems=[];events=[]
 if log.exists():
  raw=read_stable(log,MAX_LOG)
  if raw and not raw.endswith(b'\n'):problems.append('truncated_log')
  for line in raw.splitlines():
   try:events.append(json.loads(line))
   except (ValueError,TypeError):problems.append('malformed_log');break
  if any(e.get('event')=='trace_cap' for e in events):problems.append('trace_cap')
 else:problems.append('event_log_missing')
 results=[]
 for w in workers:
  matches=[s for s in statuses if s.get('pid')==w['pid']]
  item={'pid':w['pid'],'rank_supplied_unverified':w['rank'],'status':'UNKNOWN','reasons':[]}
  if len(matches)!=1:item['reasons'].append('missing_or_ambiguous_process_instance');results.append(item);continue
  s=matches[0];item['process_instance']=s.get('process_instance')
  if s.get('installation',{}).get('status')!='verified_before_install':item['reasons'].append('installation_unverified')
  if s.get('closed_window')!=window or s.get('status') not in ('WINDOW_CLOSED','PROCESS_EXIT'):item['reasons'].append('window_close_ack_missing')
  if s.get('inflight')!=0:item['reasons'].append('inflight_or_unknown')
  if s.get('state_incomplete') is not False:item['reasons'].append('observer_tainted')
  ev=[e for e in events if e.get('process_instance')==s.get('process_instance') and e.get('window_id')==window]
  begin=[e.get('call') for e in ev if e.get('event')=='embed_begin'];end=[e.get('call') for e in ev if e.get('event')=='embed_end']
  if any(type(x) is not int for x in begin+end) or len(begin)!=len(set(begin)) or len(end)!=len(set(end)) or sorted(begin)!=sorted(end):item['reasons'].append('missing_or_duplicate_call_pair')
  if s.get('calls_seen')!=len(begin) or s.get('begin_written')!=len(begin) or s.get('end_written')!=len(end):item['reasons'].append('counter_mismatch')
  item['observed_begin']=len(begin);item['observed_end']=len(end)
  item['reasons']+=problems
  if not item['reasons']:item['status']='NO_LOSS_DETECTED_CONDITIONAL_SNAPSHOT'
  results.append(item)
 return {'run_id':run,'window_id':window,'rank_mapping':'externally supplied; identity and provenance NOT verified by this script','graph_coverage':'UNKNOWN','global_issues':problems,'workers':results,'limits':['Only wrapper-observed Python calls; graph replay and uninstalled processes remain UNKNOWN.','Checkpoint and file snapshot are observations, not proof that later records cannot appear.','Rank mapping, log completeness, backend semantics and clock relationship require independent validation.']}

def main():
 p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('manifest');args=p.parse_args();print(json.dumps(audit(args.directory,json.loads(Path(args.manifest).read_text())),indent=2))
if __name__=='__main__':main()
