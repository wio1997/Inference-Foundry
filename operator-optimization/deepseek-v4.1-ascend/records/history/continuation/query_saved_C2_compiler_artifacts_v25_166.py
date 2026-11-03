from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72')
OUT=P/'analysis/native-PMU-complete-v18/saved-C2-compiler-and-cache-log-surfaces-v25.json';assert not OUT.exists()
p=P/'analysis/native-execution-identity-v14/saved-profile-surfaces-v14.json';b=p.read_bytes();assert hashlib.sha256(b).hexdigest()=='5e9b2972791b9d02814abea7256d07b72fbe043d715efab5f74cdcf518649792';inv=json.loads(b)
refs={r['relative_path']:r for r in inv['files']};binrefs=[r for r in inv['files']if Path(r['relative_path']).suffix=='.npubin'];assert len(binrefs)==73
metadata=[];selected=[];unresolved=[]
for r in binrefs:
 relative=Path(r['relative_path']);base=relative.with_suffix('');mr=str(base.with_suffix('.json'))
 if mr not in refs:unresolved.append(dict(binary=r['relative_path'],reason='NO_SAME_STEM_JSON_IN_SAVED_INDEX'));continue
 mp=P/'runtime'/mr;assert mp.stat().st_size==refs[mr]['bytes']and mp.stat().st_size<=262144
 bb=mp.read_bytes();d=json.loads(bb);name=str(d.get('name',''))
 metadata.append(dict(path=mr,name=name,sha256=hashlib.sha256(bb).hexdigest(),keys=list(d)))
 if '_pool_kernel'not in name and '_cache_update_kernel'not in name and not any(k in relative.name for k in ['_pool_kernel','_cache_update_kernel']):continue
 artifacts=[]
 for ext in ['.json','.source','.ttir','.npubin']:
  rp=str(base.with_suffix(ext))
  if rp not in refs:continue
  ap=P/'runtime'/rp;bb=ap.read_bytes();assert len(bb)==refs[rp]['bytes'];rec=dict(path=rp,bytes=len(bb),sha256=hashlib.sha256(bb).hexdigest())
  if ext in ['.source','.ttir']:
   ss=bb.decode(errors='replace').splitlines();terms=['TOKEN_ALIGNED','RATIO','CACHE_SIZE','func.func','tt.func','_pool_kernel','_cache_update_kernel','constexpr']
   lines=[dict(line=i+1,text=l[:500])for i,l in enumerate(ss)if any(t in l for t in terms)];rec['selected_signature_or_constant_lines']=lines[:60];rec['matching_line_count']=len(lines)
  artifacts.append(rec)
 selected.append(dict(metadata_path=mr,metadata=d,artifacts=artifacts))
logs=[]
for r in inv['files']:
 rp=r['relative_path'];name=Path(rp).name.lower()
 if Path(rp).suffix not in ['.log','.txt']or not any(t in name for t in ['server','serve','vllm','engine','stdout']):continue
 if len(logs)>=12 or r['bytes']>10000000:continue
 f=P/'runtime'/rp;bb=f.read_bytes();assert len(bb)==r['bytes'];terms=['KV cache','kv_cache_config','CircularBuffer','num_blocks','page_size','physical blocks','GPU blocks','block_size','max_num_seqs']
 hits=[dict(line=i+1,text=l[:700])for i,l in enumerate(bb.decode(errors='replace').splitlines())if any(t in l for t in terms)]
 logs.append(dict(path=rp,bytes=len(bb),sha256=hashlib.sha256(bb).hexdigest(),hits=hits[:120],matching_lines=len(hits)))
result=dict(created_utc=datetime.now(timezone.utc).isoformat(),saved_compiled_binary_entries=len(binrefs),same_stem_metadata_entries=len(metadata),unresolved_metadata=unresolved,C2_compiler_candidates=selected,selected_cache_logs=logs,
 limit='Saved compiler artifacts andnames only. Without an execution->binary hash/module-load link,candidates do not prove a specificbinary wasexecuted. Log absence isnot runtimeconfiguration absence.',
 no_new_target_source_DB_model_import_or_device_or_service_work=True,overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(summary={k:v for k,v in result.items()if k not in ['C2_compiler_candidates','selected_cache_logs']},C2_compiler_candidates=selected,selected_cache_logs=logs,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
