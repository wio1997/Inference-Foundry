# SPDX-License-Identifier: Apache-2.0
"""CPU fault injection into the actual Decode host finally and stop fence."""
import argparse,ast,contextlib,copy,hashlib,io,json,pathlib,runpy,shutil,tarfile,tempfile,types

p=argparse.ArgumentParser()
p.add_argument('--run',type=pathlib.Path,required=True)
p.add_argument('--run-id',required=True)
a=p.parse_args();R0=a.run
source=(R0/'host_probe.py').read_text();tree=ast.parse(source)
block=next(n for n in tree.body if isinstance(n,ast.Try)).finalbody
compiled=compile(ast.fix_missing_locations(ast.Module(body=block,type_ignores=[])),'actual-host-finally-CPU-fixture','exec')
helper=R0/'flash_pd_writer_first_cleanup.py';m=runpy.run_path(str(helper));results=[]
good={'project_id':'glm53-flash','run_id':a.run_id,'node':46,'image':'fixture-image','container':'if-glm53-flash-'+a.run_id+'-pd46','cleanup_identity_verified':True,'post_inspect':{'rc':0,'stdout':'false 0\n'},'post_npu':{'rc':0,'stdout':'No running processes found\n'*8}}
for case in ('valid_permit','missing_permit','corrupt_permit','wrong_writer','wrong_identity'):
 with tempfile.TemporaryDirectory() as td:
  R=pathlib.Path(td);shutil.copyfile(helper,R/helper.name);(R/'destinations-exposed').write_text('CPU fixture only')
  permit=m['writer_stop_proof'](good,run_id=a.run_id,image='fixture-image',container=good['container'])
  if case=='corrupt_permit':permit['sha256']='0'*64
  if case=='wrong_writer':permit['writer_container']='another-project'
  if case!='missing_permit':(R/'writer-stop-permit.json').write_text(json.dumps(permit))
  clock=[0.];calls=[];saved=[];out={'project_id':'glm53-flash'}
  def run(args,t=30):
   calls.append(args)
   if args[:2]==['docker','inspect']:
    if '--format' in args:text='false 0\n'
    else:text=json.dumps([{'Image':'fixture-image','Config':{'Labels':{'project_id':'other' if case=='wrong_identity' else 'glm53-flash','run_id':a.run_id}}}])
   elif args[:2]==['docker','stop']:
    assert out['decode_stop_permitted'];text='fixture-own-container\n'
   elif args==['npu-smi','info']:text='No running processes found\n'*8
   else:raise AssertionError(args)
   return {'rc':0,'stdout':text,'stderr':''}
  def logs(args,**kw):
   assert args[:2]==['docker','logs'];calls.append(args);kw['stdout'].write(b'CPU_FIXTURE\n');return types.SimpleNamespace(returncode=0)
  def save():saved.append(copy.deepcopy(out))
  time=types.SimpleNamespace(monotonic=lambda:clock[0],sleep=lambda seconds:clock.__setitem__(0,clock[0]+seconds))
  env={'R':R,'node':47,'created':True,'name':'fixture-own-container','image':'fixture-image','out':out,'run':run,'save':save,'json':json,'time':time,'subprocess':types.SimpleNamespace(run=logs,STDOUT=-2),'hashlib':hashlib,'tarfile':tarfile}
  with contextlib.redirect_stdout(io.StringIO()):exec(compiled,env)
  stops=[c for c in calls if c[:2]==['docker','stop']]
  assert len(stops)==int(case=='valid_permit')
  if case=='valid_permit':assert out['post_inspect']['stdout'].strip()=='false 0' and saved[0]['decode_stop_permitted']
  else:assert out.get('cleanup_error') and 'post_inspect' not in out
  if case=='wrong_identity':assert not any(c[:2]==['docker','logs'] for c in calls)
  results.append({'case':case,'passed':True,'actual_resource_commands':0,'mocked_D_stop_calls':len(stops)})
# Controller must never publish a permit while P is live/occupied/wrong owner.
for case,change in [('normal',{}),('nonzero_exit',{'post_inspect':{'rc':0,'stdout':'false 137'}}),('running',{'post_inspect':{'rc':0,'stdout':'true 0'}}),('seven_idle',{'post_npu':{'rc':0,'stdout':'No running processes found\n'*7}}),('wrong_run',{'run_id':'other'}),('unverified_identity',{'cleanup_identity_verified':False})]:
 status=copy.deepcopy(good);status.update(change);clock=[0.];calls=[]
 result=m['cleanup'](lambda n,f,v:calls.append((n,f)),lambda:status,run_id=a.run_id,image='fixture-image',container=good['container'],bound=1.,monotonic=lambda:clock[0],sleep=lambda s:clock.__setitem__(0,clock[0]+s))
 expected=case in ('normal','nonzero_exit');assert result['writer_stop_proven']==result['decode_stop_permitted']==expected
 assert calls==([(46,'stop.json'),(47,'writer-stop-permit.json'),(47,'stop.json')] if expected else [(46,'stop.json')])
 results.append({'case':'controller_'+case,'passed':True,'actual_resource_commands':0,'decode_stop_permitted':expected})
report={'project_id':'glm53-flash','passed':True,'host_sha256':hashlib.sha256(source.encode()).hexdigest(),'helper_sha256':hashlib.sha256(helper.read_bytes()).hexdigest(),'cases':results,'actual_resource_commands':0,'scope':'CPU callback/actual-finally fault injection only; no runtime shutdown qualification.'}
(R0/'local-writer-first-host-fixture.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
