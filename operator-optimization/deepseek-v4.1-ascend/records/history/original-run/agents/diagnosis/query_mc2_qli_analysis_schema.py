#!/usr/bin/env python3
"""Metadata-only schemas of two fixed historical profiler analysis.db files."""
import hashlib,json,re,sqlite3,stat
from pathlib import Path
BASE=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces').resolve(strict=True)
RANKS=[('ep0','dp0_pp0_tp0_dcp0_ep0_rank0_728_20261028020238584_ascend_pt',323584),('ep8','dp1_pp0_tp0_dcp0_ep8_rank0_727_20261028020238580_ascend_pt',290816)]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(65536),b''):h.update(b)
 return h.hexdigest()
def safe_name(s):
 return s if isinstance(s,str) and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,127}',s) else None
def category(s):
 return [t for t in ('expert','route','token','length','seq','metadata','tensor','memory','task','rank') if isinstance(s,str) and t in s.lower()]
all_results=[]
for label,rank,observed_size in RANKS:
 raw=BASE/rank/'ASCEND_PROFILER_OUTPUT'/'analysis.db'; p=raw.resolve(strict=True)
 if p!=raw or not p.is_relative_to(BASE):raise RuntimeError('unexpected path or symlink')
 st=p.stat()
 if not stat.S_ISREG(st.st_mode) or st.st_size>2_000_000:raise RuntimeError('not regular or too large')
 digest=sha(p);out={'rank':label,'size_bytes':st.st_size,'matches_indexed_size':st.st_size==observed_size,'sha256_observed_now':digest,'wal_present':Path(str(p)+'-wal').exists(),'shm_present':Path(str(p)+'-shm').exists(),'tables':[]}
 c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.execute('PRAGMA query_only=ON');c.execute('PRAGMA temp_store=MEMORY')
 ticks=[0]
 def budget():
  ticks[0]+=1
  return int(ticks[0]>1000)
 c.set_progress_handler(budget,10000)
 try:
  tables=c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name LIMIT 101").fetchall()
  out['table_list_truncated']=len(tables)>100
  for (name,) in tables[:100]:
   safe=safe_name(name); table={'name':safe,'unsafe_name_sha256':None if safe is not None else hashlib.sha256(str(name).encode()).hexdigest(),'name_categories':category(safe),'columns':[]}
   quoted='"'+name.replace('"','""')+'"'
   cols=c.execute('PRAGMA table_info('+quoted+')').fetchmany(101);table['columns_truncated']=len(cols)>100
   for col in cols[:100]:
    cn=safe_name(col[1]);table['columns'].append({'name':cn,'unsafe_name_sha256':None if cn is not None else hashlib.sha256(str(col[1]).encode()).hexdigest(),'categories':category(cn)})
   out['tables'].append(table)
  out['status']='complete'
 except sqlite3.OperationalError:out['status']='interrupted_or_sqlite_error'
 c.set_progress_handler(None,0);c.close();out['sha256_after_read']=sha(p);out['unchanged_during_read']=digest==out['sha256_after_read'];all_results.append(out)
print(json.dumps({'scope':'metadata-only tables and columns in fixed EP0 and EP8 analysis.db; no data rows','samples':all_results,'limits':['Observed SHA is only a current read fingerprint, not a proven historical frozen hash.','Names or absence of names do not establish runtime tensor values, content, replay bindings, or source completeness.','Potential WAL indicates same-file SHA alone does not identify complete logical DB state.','No table row contents, NPU access, user payloads, or target edits.']},ensure_ascii=False,indent=2))
