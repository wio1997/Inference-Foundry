from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
A=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030');P=A/'hostprof-20261030T212812Z-360e0c72'
OUT=P/'analysis/native-execution-identity-v14/saved-profile-hardware-and-engine-provenance-v17.json';assert not OUT.exists()
def pin(p,sha):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
caps=pin(P/'analysis/profile-capabilities-index-v12-v2.json','79ba556ef3855b9066392e438371a0a9d3b38ea7511269ce7ff2a6814529a095');old=pin(A/'profile-readonly-20261031/framework-speedscope-callsites-v8.json','b3c6e2a72bf5af17ab18769b8795b9a9ee8d759850c0d468abccc35dce74378d')
manifestpath=P/'analysis/offline-db-v12/input-copy-manifest.json';mb=manifestpath.read_bytes();manifest={r['relative_path']:r for r in json.loads(mb)}
rows=[];seen=set()
wanted={'DeviceInfo','platform','platform_version','chip_id','chip_type','soc_version','soc_name','aic_frequency','aiv_frequency','ai_core_num','ai_vector_num','ai_cpu_num','SysCntFreq','syscnt_freq','syscntFreq','CANN','cann_version','torch_npu_version'}
def select(v,path=''):
 out={}
 if isinstance(v,dict):
  for k,x in v.items():
   q=path+'.'+k if path else k
   if k=='DeviceInfo':
    devices=x if isinstance(x,list)else[x]
    out[q+'_field_names']=[list(z)for z in devices if isinstance(z,dict)]
    keep={'id','ai_core_num','aic_frequency','aiv_frequency','aiv_num','ai_cpu_num','ai_cpu_frequency','hwts_frequency','aicpu_num','aicore_num','chip_type','chip_id','dev_type','device_type','vector_core_num','memory_size','type'}
    out[q]=[{kk:vv for kk,vv in z.items()if kk in keep}for z in devices if isinstance(z,dict)]
   elif k in wanted:out[q]=x
   elif isinstance(x,(dict,list)):out.update(select(x,q))
 elif isinstance(v,list):
  for i,x in enumerate(v):out.update(select(x,path+'['+str(i)+']'))
 return out
for ref in caps['profiler_infos']:
 info=Path(ref['path']);root=info.parent
 for p in [info]+sorted(root.rglob('info*.json*')):
  if str(p)in seen:continue
  seen.add(str(p));assert p.stat().st_size<2*1024**2
  b=p.read_bytes();sha=hashlib.sha256(b).hexdigest();rel=str(p.relative_to(P/'runtime/official-traces'));copy=manifest.get(rel)
  if copy:assert sha==copy['sha256']
  if p==info:assert sha==ref['sha256']
  try:obj=json.loads(b)
  except (json.JSONDecodeError,UnicodeDecodeError)as exc:
   rows.append(dict(path=str(p),sha256=sha,bytes=len(b),original_copy_manifest_SHA_match=bool(copy),parse_status='NON_JSON_MATCHING_INFO_FILENAME',parse_error=type(exc).__name__,selected_profile_hardware_fields={}))
   continue
  rows.append(dict(path=str(p),sha256=sha,bytes=len(b),original_copy_manifest_SHA_match=bool(copy),parse_status='JSON',top_level_keys=list(obj)if isinstance(obj,dict)else[],selected_profile_hardware_fields=select(obj)))
assert len(rows)<300
source=dict(artifact_path=str(A/'profile-readonly-20261031/framework-speedscope-callsites-v8.json'),artifact_sha256='b3c6e2a72bf5af17ab18769b8795b9a9ee8d759850c0d468abccc35dce74378d',original_source_root=old['source_root'],source_SHA256=old['source_sha256'],interpretation='This EngineCore sampled evidence is from the archived source169-project run,not from the current166 hostprof process set.')
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),profile_info_records=len(rows),parse_status_counts=dict(Counter(r['parse_status']for r in rows)),TorchNPU_profiler_info_files=len(caps['profiler_infos']),engine_original_source_root=old['source_root'],no_model_source_device_or_inference_work=True,overall_complete=False,limits=['Recorded hardware metadata is a capture-time field,not independent driver/chip peak capability verification.','A frequency value still needs its primary unit/normalization path; no MFU or hardware-peak selection applied.','Archived EngineCore sample weights and processCPU remain separate from current166 worker/frontend captures.'])
payload=(json.dumps(dict(summary=summary,input_copy_manifest=dict(path=str(manifestpath),sha256=hashlib.sha256(mb).hexdigest()),profile_hardware=rows,EngineCore_sample_provenance=source),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
distinct={}
for r in rows:
 fields=r['selected_profile_hardware_fields'];key=json.dumps(fields,sort_keys=True,separators=(',',':'));distinct.setdefault(key,dict(records=0,fields=fields));distinct[key]['records']+=1
public=dict(summary=summary,engine_provenance=source,distinct_profile_hardware_fields=list(distinct.values()),full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest())
print(json.dumps(public,ensure_ascii=False))
