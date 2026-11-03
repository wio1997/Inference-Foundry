from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,socket,re
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72')
OUT=P/'analysis/native-PMU-complete-v18/current-capture-service-cache-log-v25.json';assert not OUT.exists()
p=P/'runtime/logs/service.log';assert p.exists();st=p.stat();assert st.st_size<50000000;b=p.read_bytes();text=b.decode(errors='replace');rows=[];params=[]
for i,line in enumerate(text.splitlines()):
 if len(line)>1600:
  vals={}
  for key,pattern in [('max_num_seqs',r'max_num_seqs[=:]\s*(\d+)'),('num_gpu_blocks_override',r'num_gpu_blocks_override[=:]\s*(None|\d+)'),('kv_cache_dtype',r'kv_cache_dtype[=:]\s*([\w\-]+)'),('kv_cache_layout',r'kv_cache_layout[=:]\s*([\w\-]+)'),('prefix_caching',r'enable_prefix_caching[=:]\s*(True|False)'),('kv_transfer_none',r'kv_transfer_config[=:]\s*(None)')]:
   m=re.search(pattern,line)
   if m:vals[key]=m.group(1)
  if vals:params.append(dict(line=i+1,whitelisted_parameters=vals,original_line_length=len(line)))
  continue
 if any(term.lower()in line.lower()for term in ['available kv cache memory','available memory for kv','kv cache size','GPU blocks','NPU blocks','num_blocks','block_stride','page_size','cache layout','layer tuples','layer tuple','maximum concurrency']):rows.append(dict(line=i+1,text=line))
assert(st.st_size,st.st_mtime_ns)==(p.stat().st_size,p.stat().st_mtime_ns)
result=dict(created_utc=datetime.now(timezone.utc).isoformat(),path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),original_stat_unchanged=True,technical_cache_lines=rows[:160],technical_cache_line_count=len(rows),whitelisted_startup_parameters=params[:10],
scope='Exact saved service.log path fromthecapturecontroller,not oldmigratedhome logs or newlivehealth.',no_new_source_model_import_or_device_or_service_work=True,overall_complete=False)
bb=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(bb)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(bb).hexdigest()),ensure_ascii=False))
