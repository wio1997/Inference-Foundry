"""Read archived launch/visibility metadata and specific device-identity log lines."""
from pathlib import Path
from datetime import datetime, timezone
import json, subprocess
from finalize_origin_coverage_v9 import N, O, read, write
from cache_saved_raw_evidence_v155 import ssh_args
from inspect_saved_raw_profile_v155 import BASE


def main():
    output=O/'rank-device-archived-provenance-v156.json';assert not output.exists()
    container=read(O/'raw-record-schema-source-index-v155.json')['container_id']
    code='''import os,socket,json,hashlib,re,subprocess
from pathlib import Path
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'and os.geteuid()==0
base=Path(%r);container=%r
allowed={'ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES','ASCEND_DEVICE_ID','DEVICE_ID','LOCAL_RANK','RANK','WORLD_SIZE','CANN_VERSION'}
def source(path):
 assert path.is_file()and not path.is_symlink()and path.resolve().is_relative_to(base.resolve());s=path.stat();b=path.read_bytes();t=path.stat();assert(s.st_size,s.st_mtime_ns)==(t.st_size,t.st_mtime_ns);return b,dict(relative=str(path.relative_to(base)),bytes=len(b),mtime_ns=s.st_mtime_ns,sha256=hashlib.sha256(b).hexdigest(),source_stat_unchanged=True)
b,launch_src=source(base/'docker-create-argv.json');args=json.loads(b);envs=[];devices=[]
for i,arg in enumerate(args[:-1]):
 if arg=='--env':
  name,_,value=args[i+1].partition('=')
  if name in allowed:envs.append(dict(name=name,value=value))
 if arg=='--device':devices.append(args[i+1])
p=subprocess.run(['docker','inspect',container],capture_output=True,text=True,check=True);obj=json.loads(p.stdout)[0];assert obj['Id']==container and not obj['State']['Running'];cfg=obj['Config'];inspect_env=[]
for arg in cfg.get('Env',[]):
 name,_,value=arg.partition('=')
 if name in allowed:inspect_env.append(dict(name=name,value=value))
inspect_filtered=dict(id=obj['Id'],image=obj['Image'],running=obj['State']['Running'],privileged=obj['HostConfig']['Privileged'],devices=obj['HostConfig'].get('Devices'),filtered_env=inspect_env)
scripts=[]
for name in['serve.sh','serve-host-profile.sh']:
 path=base/'runtime'/'scripts'/name
 if not path.exists():continue
 b,meta=source(path);selected=[]
 for i,line in enumerate(b.decode().splitlines()):
  if any(re.match(r'^\\s*(?:export\\s+)?'+re.escape(name)+r'=',line)for name in allowed):selected.append(dict(line=i+1,text=line))
 scripts.append(dict(source=meta,visibility_or_rank_lines=selected))
logs=[];logroot=base/'runtime'/'logs'
pattern=re.compile(r'(?:physical.{0,16}(?:device|chip)|(?:device|chip).{0,16}physical|logic(?:al)?.{0,16}device|device.{0,16}(?:logic|bus|pci)|(?:local_rank|local rank|device_id|device id|npu device|ASCEND_RT_VISIBLE_DEVICES)\\s*[:=])',re.I)
if logroot.exists():
 for path in sorted(logroot.iterdir()):
  if not path.is_file()or path.is_symlink():continue
  if path.stat().st_size>32*1024**2:logs.append(dict(relative=str(path.relative_to(base)),skipped='32MiB bounded log cap'));continue
  b,meta=source(path);hits=[];total=0
  for i,line in enumerate(b.decode(errors='replace').splitlines()):
   if pattern.search(line):
    total+=1
    if len(hits)<160:hits.append(dict(line=i+1,text=line[:1400],truncated=len(line)>1400))
  logs.append(dict(source=meta,matching_lines=total,selected=hits,selection_capped=total>160))
print(json.dumps(dict(hostname=socket.gethostname(),euid=os.geteuid(),launch_source=launch_src,archived_launch_filtered_env=envs,archived_launch_device_args=devices,stopped_container_filtered=inspect_filtered,scripts=scripts,logs=logs,root_entry_names=[p.name for p in sorted(base.iterdir())],read_only_archived_files_and_stopped_inspect=True,no_live_device_or_process_queries=True,new_device_work=False)))
'''%(BASE,container)
    p=subprocess.run(ssh_args(code),capture_output=True,timeout=60);assert p.returncode==0,p.stderr.decode(errors='replace')[-1000:]
    data=json.loads(p.stdout);data.update(observed_utc=datetime.now(timezone.utc).isoformat(),explicit_runtime_logic_to_physical_bridge_proved=False,overall_complete=False);write(output,data)
    print(json.dumps(dict(status='PASS',launch_filtered_env=data['archived_launch_filtered_env'],scripts=len(data['scripts']),logs=[dict(source=x.get('source',{}).get('relative'),matching_lines=x.get('matching_lines'),skipped=x.get('skipped'))for x in data['logs']],new_device_work=False)))


if __name__=='__main__':main()
