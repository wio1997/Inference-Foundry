"""Repair known diagnostic-only copied pin;restart owned startup once,retain receipt."""
import base64,hashlib,json,sys,zlib
from pathlib import Path
sys.path.insert(0,'/Users/wio/work/171-vllm-migration-20261102');from control import invoke
H=Path(__file__).resolve().parent;p=H/'hook/joint_runtime.py';old=p.read_text();bad='efe13f0e801b70d7fba7498ac44062a8eccfd78e5b6303cf658d33dd22217680';good='efe13f0e801b70d7fba7498ac44062a8eccfd78e5b6303cf658d33dd22217580';assert old.count(bad)==1
new=old.replace(bad,good);compile(new,str(p),'exec');(H/'joint_runtime.submitted-pin-error.py.txt').write_text(old)
code='PAYLOAD='+repr(base64.b64encode(zlib.compress(new.encode())).decode())+'\nEXPECTED_OLD='+repr(hashlib.sha256(old.encode()).hexdigest())+'\n'+r'''from pathlib import Path
import base64,hashlib,importlib.util,json,os,signal,subprocess,sys,time,zlib
p=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/joint176');h=json.loads((p/'SUBMITTED.json').read_text());v=Path('/proc/'+str(h['pid'])+'/stat').read_text().rsplit(')',1)[1].split();assert v[19]==h['start_ticks'] and v[0]!='Z';s=json.loads((p/'STATUS.json').read_text());assert s['state']=='WAITING_JOINT_MODEL_HEALTH' and s['case_attempts']==0
f=p/'hook/joint_runtime.py';assert hashlib.sha256(f.read_bytes()).hexdigest()==EXPECTED_OLD
C=Path('/data/tiankuan/wio/migration-171-20261102/builtin-cohort-v164-228ad25fcf9b');sys.path.insert(0,str(C));spec=importlib.util.spec_from_file_location('owned_tools',C/'controller.py');tools=importlib.util.module_from_spec(spec);spec.loader.exec_module(tools);busy,_=tools.busy();tools.owns_processes(busy,tools.owned(tools.MAIN))
os.kill(h['pid'],signal.SIGSTOP)
receipt={'reason':'Copied diagnostic hash contained decimal175 substring incorrectly replaced with176;fix exact pinned SHA before worker load. Target source unchanged.','controller_temporarily_stopped_for_owned_startup_restart':True,'controller_pid':h['pid'],'start_ticks':h['start_ticks'],'time':time.time(),'old_hook_sha256':EXPECTED_OLD,'extra_owned_diagnostic_reload':1}
try:
 tools.command(['docker','stop','--time','30',tools.MAIN],60)
 oldlog=p/'diagnostic-service.log'
 if oldlog.exists():(p/'diagnostic-service-before-pin-fix.log').write_bytes(oldlog.read_bytes())
 (p/'hook/joint_runtime.submitted-pin-error.py.txt').write_bytes(f.read_bytes());new=zlib.decompress(base64.b64decode(PAYLOAD));compile(new,str(f),'exec');q=f.with_suffix('.fix');q.write_bytes(new);q.replace(f);receipt['new_hook_sha256']=hashlib.sha256(new).hexdigest()
 tools.command(['docker','start',tools.MAIN],30);receipt['new_container_started_at']=tools.inspect(tools.MAIN)['State']['StartedAt']
finally:
 (p/'PIN-CORRECTION.json').write_text(json.dumps(receipt,indent=2)+'\n');os.kill(h['pid'],signal.SIGCONT)
print(json.dumps(receipt))'''
x=invoke('target',code,timeout=110);(H/'pin-correction.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');p.write_text(new);print(json.dumps(x))
