from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,socket,re,shlex
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72');OUT=P/'analysis/native-PMU-complete-v18/saved-overlap-configuration-v21.json';assert not OUT.exists()
match=re.compile(r'overlap|shared.{0,24}(expert|stream)|multi.?stream|enable_fused_mc2|mix_placement|use_mega_moe|cudagraph_mode|enforce_eager|graph_capture',re.I)
secret=re.compile(r'password|api.?key|access.?key|secret|credential|auth.?token',re.I)
def select(obj,path=''):
 out=[]
 if isinstance(obj,dict):
  for k,v in obj.items():
   name=path+'.'+str(k)if path else str(k)
   if secret.search(str(k)):continue
   if match.search(str(k))and isinstance(v,(str,bool,int,float,type(None))):out.append(dict(field=name,value=v))
   if isinstance(v,(dict,list)):out.extend(select(v,name))
 elif isinstance(obj,list):
  for i,v in enumerate(obj):
   if isinstance(v,(dict,list)):out.extend(select(v,path+'['+str(i)+']'))
 return out
records=[]
for rel in ['preflight.json','diagnostic-identity.json','docker-create-argv.json','controller-finished.json','runtime/scripts/serve.sh','runtime/scripts/serve-host-profile.sh','runtime/logs/service.log']:
 p=P/rel
 if not p.exists():records.append(dict(path=str(p),exists=False));continue
 assert p.stat().st_size<128*1024**2,(str(p),'bounded saved config/log read exceeded')
 b=p.read_bytes();text=b.decode(errors='replace');entry=dict(path=str(p),exists=True,bytes=len(b),sha256=hashlib.sha256(b).hexdigest());hits=[]
 if rel=='docker-create-argv.json':
  args=json.loads(text);env=[]
  for i,a in enumerate(args):
   if a=='--env'and i+1<len(args):
    name,sep,value=args[i+1].partition('=')
    if match.search(name)and not secret.search(name):env.append(dict(name=name,value=value))
  entry.update(relevant_explicit_environment=env,environment_names=[args[i+1].partition('=')[0]for i,a in enumerate(args)if a=='--env'and i+1<len(args)and match.search(args[i+1].partition('=')[0])],full_argv_not_emitted=True)
 elif rel.endswith('.json'):
  entry['selected_fields']=select(json.loads(text))
 elif rel.endswith('.sh'):
  for i,l in enumerate(text.splitlines(),1):
   if l.lstrip().startswith('export '):
    name,sep,value=l.lstrip()[7:].partition('=')
    if match.search(name)and not secret.search(name):hits.append(dict(line=i,name=name,value=value))
   if l.startswith('exec ')and'vllm serve 'in l:
    args=shlex.split(l)
    for j,a in enumerate(args):
     if a in ['--additional-config','--compilation-config']and j+1<len(args):
      try:hits.append(dict(line=i,flag=a,selected_fields=select(json.loads(args[j+1]))))
      except json.JSONDecodeError:hits.append(dict(line=i,flag=a,parse_error='not JSON;not interpreted'))
     elif a=='--enforce-eager':hits.append(dict(line=i,flag=a,value=True))
  entry['selected_script_settings']=hits
 else:
  # Preserve only bounded key/value evidence around relevant configuration names.
  spans=[];total=0
  for i,l in enumerate(text.splitlines(),1):
   found=list(match.finditer(l));total+=len(found)
   for m in found:
    if len(spans)>=45:break
    snippet=l[max(0,m.start()-50):min(len(l),m.end()+180)]
    if secret.search(snippet):snippet='[omitted surrounding text with authentication-related field]'
    spans.append(dict(line=i,snippet=snippet))
  entry.update(relevant_match_count=total,bounded_log_spans=spans,display_is_complete=total<=45)
 records.append(entry)
cfg=P/'runtime/config';cfgfiles=sorted(p for p in cfg.rglob('*')if p.is_file())if cfg.exists()else[];assert len(cfgfiles)<=50
for p in cfgfiles:
 b=p.read_bytes();assert len(b)<2*1024**2
 try:fields=select(json.loads(b))
 except (json.JSONDecodeError,UnicodeDecodeError):fields=[]
 records.append(dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),selected_json_fields=fields,content_not_emitted=True))
result=dict(created_utc=datetime.now(timezone.utc).isoformat(),capture_root=str(P),records=records,scope='Only saved capture-run configuration/argv/serve scripts/service log;no current environment query or source import.',no_new_source_DB_model_device_or_service_work=True,overall_complete=False,limits=['Absence of an explicit environment variable is not the default value or runtime activation.','Configuration values and source guard definitions must be bound before causal interpretation.','Captured log spans are bounded;full file hashes retained,not full logs or credentials.','No fresh health/ownership/idle claim is made.'])
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
