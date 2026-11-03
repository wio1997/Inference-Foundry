#!/usr/bin/env python3
"""Only inventory saved EP8/EP15 profiler schemas and dump-like paths; no device access."""
import base64, hashlib, json, pathlib, shlex, subprocess
root=pathlib.Path(__file__).resolve().parent
src=root/'target-layer-mc2-16rank.json'
data=json.loads(src.read_text())
ranks=[r['rank'] for r in data['ranks'] if r.get('ep') in (8,15)]
if len(ranks)!=2: raise RuntimeError('Expected exactly two ranks')
remote=r'''import collections,csv,hashlib,json,os,pathlib,re,sqlite3,sys
root=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces')
rankdirs=json.loads(sys.argv[1]); pat=re.compile(r'expert|moe|rout|topk|active.?mask|cumsum|dump|tensor|snapshot',re.I)
result=[]
for rank in rankdirs:
 r=root/rank
 if not r.is_dir() or r.parent!=root: raise RuntimeError('rank path missing or invalid')
 ext=collections.Counter();hits=[];dbs=[];headers=[];files=0;errors=[]
 for base,dirs,names in os.walk(r,followlinks=False):
  dirs[:]=sorted(d for d in dirs if not pathlib.Path(base,d).is_symlink())
  for name in sorted(names):
   p=pathlib.Path(base,name)
   if p.is_symlink():continue
   rel=str(p.relative_to(r));files+=1;ext[p.suffix.lower() or '(none)']+=1
   if pat.search(rel):hits.append({'path':rel,'bytes':p.stat().st_size})
   if p.suffix.lower() in ('.db','.sqlite','.sqlite3'):
    try:
     conn=sqlite3.connect('file:'+str(p)+'?mode=ro&immutable=1',uri=True)
     tables=conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
     matched=[]
     for (table,) in tables:
      if not isinstance(table,str):continue
      cols=[x[1] for x in conn.execute('PRAGMA table_info('+ '"'+table.replace('"','""')+'"'+')')]
      matching=[c for c in cols if pat.search(c)]
      if pat.search(table) or matching:matched.append({'table':table,'matching_columns':matching})
     conn.close();dbs.append({'path':rel,'bytes':p.stat().st_size,'table_count':len(tables),'schema_matches':matched})
    except Exception as e:errors.append({'path':rel,'error_type':type(e).__name__})
   if name in ('kernel_details.csv','op_summary.csv'):
    try:
     with p.open(newline='',encoding='utf-8-sig') as f:row=next(csv.reader(f))
     headers.append({'path':rel,'columns':len(row),'schema_matches':[c for c in row if pat.search(c)],'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()})
    except Exception as e:errors.append({'path':rel,'error_type':type(e).__name__})
 result.append({'rank':rank,'file_count':files,'extensions':dict(sorted(ext.items())),'keyword_paths':hits,'csv_headers':headers,'databases':dbs,'errors':errors})
print(json.dumps(result,ensure_ascii=True))'''
encoded=base64.b64encode(remote.encode()).decode()
cmd='docker exec wio-vllm-v41-repro169-stage-20261028 python3 -c '+shlex.quote("import base64;exec(base64.b64decode('"+encoded+"'))")+' '+shlex.quote(json.dumps(ranks,separators=(',',':')))
p=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',cmd],capture_output=True,text=True,timeout=120,check=True)
out={'input_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'scope':'existing recapture EP8 and EP15 profile trees only, schemas and candidate paths, no tensor contents','ranks':json.loads(p.stdout)}
target=root/'recapture-expert-metadata-inventory.json';target.write_text(json.dumps(out,indent=2)+'\n')
for r in out['ranks']:print(r['rank'], 'files',r['file_count'],'hits',len(r['keyword_paths']),'db',len(r['databases']),'errors',len(r['errors']))
print('saved',target)
