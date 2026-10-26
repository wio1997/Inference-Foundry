#!/usr/bin/env python3
"""Materialize a new benchmark revision; preserve already frozen measurements."""
import ast,hashlib,json,pathlib
R=pathlib.Path(__file__).resolve().parents[1]
old=R/'scripts/bench_tokenizer_image.py';s=old.read_text()
start=s.index('def cgroup_stat():');end=s.index('def read_optional',start)
replacement=r'''def discover_cgroup():
 membership=pathlib.Path('/proc/self/cgroup').read_text();mountinfo=pathlib.Path('/proc/self/mountinfo').read_text();groups={}
 for line in membership.splitlines():
  _,controllers,path=line.split(':',2)
  for controller in controllers.split(','):groups[controller]=path
 candidates=[]
 for line in mountinfo.splitlines():
  left,right=line.split(' - ',1);fields=left.split();fs,_,options=right.split()[:3];mount_root=pathlib.PurePosixPath(fields[3]);mount=pathlib.Path(fields[4])
  if fs=='cgroup2':version=2;group=groups.get('')
  elif fs=='cgroup' and 'cpu' in options.split(','):version=1;group=groups.get('cpu')
  else:continue
  if group is None:continue
  group_path=pathlib.PurePosixPath(group)
  if group_path.is_relative_to(mount_root):paths=[mount/str(group_path.relative_to(mount_root))]
  elif group=='/':paths=[mount]
  else:paths=[]
  for path in paths:
   if (path/'cpu.stat').is_file():candidates.append((version,path,group,str(mount_root)))
 assert len(candidates)==1,{'cgroup_candidates':[(v,str(p),g,r) for v,p,g,r in candidates],'membership':membership,'mountinfo':mountinfo}
 version,path,group,mount_root=candidates[0]
 if version==1:
  quota=int((path/'cpu.cfs_quota_us').read_text());period=int((path/'cpu.cfs_period_us').read_text());assert quota==period and quota>0
 else:
  quota,period=(path/'cpu.max').read_text().split();assert quota!='max' and int(quota)==int(period)
 return {'version':version,'cpu_path':str(path),'group':group,'mount_root':mount_root,'quota_us':int(quota),'period_us':int(period),'membership':membership}
cgroup=discover_cgroup()
def cgroup_stat():
 raw={line.split()[0]:int(line.split()[1]) for line in (pathlib.Path(cgroup['cpu_path'])/'cpu.stat').read_text().splitlines() if len(line.split())==2}
 assert {'nr_periods','nr_throttled'}<=raw.keys()
 if cgroup['version']==1:
  assert 'throttled_time' in raw;raw['throttled_usec']=raw['throttled_time']//1000
 else:assert 'throttled_usec' in raw
 return raw
'''
s=s[:start]+replacement+s[end:];needle="'cpu_max':read_optional('/sys/fs/cgroup/cpu.max'),";assert s.count(needle)==1;s=s.replace(needle,"'cgroup':cgroup," );ast.parse(s)
target=R/'scripts/bench_tokenizer_image_v2.py';target.open('x').write(s)
p=json.loads((R/'specs/L011-protocol.json').read_text());p['version']=2;p['measurement_revision']={'reason':'R001/R002 assumed cgroup v2 root cpu.stat; every sample reported unavailable. Preserve both runs; rerun same design with discovered v1/v2 membership and verified one-CPU quota.','prior_protocol_sha256':hashlib.sha256((R/'specs/L011-protocol.json').read_bytes()).hexdigest(),'changed':['cgroup discovery and counter sampling only'],'not_changed':['inputs','actor parameters','arrival process','warmup','seed','pairs','timers','CPU limits']};(R/'specs/L011-protocol-v2.json').open('x').write(json.dumps(p,indent=2)+'\n')
materializer=(R/'scripts/materialize_tokenizer_benchmark.py').read_text().replace('specs/L011-protocol.json','specs/L011-protocol-v2.json').replace('bench_tokenizer_image.py','bench_tokenizer_image_v2.py');ast.parse(materializer);(R/'scripts/materialize_tokenizer_benchmark_v2.py').open('x').write(materializer)
print(json.dumps({'probe_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((R/'specs/L011-protocol-v2.json').read_bytes()).hexdigest()}))
