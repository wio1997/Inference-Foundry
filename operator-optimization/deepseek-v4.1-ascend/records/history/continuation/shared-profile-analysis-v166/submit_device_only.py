import base64,hashlib,json,shlex,sys,zlib
from pathlib import Path
sys.path.insert(0,'/Users/wio/work/171-vllm-migration-20261102')
from control import invoke
HERE=Path(__file__).resolve().parent
original=(HERE.parent/'profile-entrypoints-cache-v162/serve.sh').read_text()
assert hashlib.sha256(original.encode()).hexdigest()=='c297201185b5676254af4d7693279f89bfa60f9c76eb340058a78ecf730b7bba'
app='/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash/analysis/npu166'
cfg={'profiler':'torch','torch_profiler_dir':app+'/raw','torch_profiler_with_memory':False,
     'torch_profiler_with_stack':False,'torch_profiler_record_shapes':False,'ignore_frontend':True,'max_iterations':12}
lines=original.splitlines()
at=next(i for i,l in enumerate(lines) if l.startswith('exec '))
lines[at]+=' --profiler-config '+shlex.quote(json.dumps(cfg,separators=(',',':')))
lines[at:at]=['export WIO_DEVICE_ONLY_PROFILE=v166',
              'export WIO_DEVICE_PROFILE_ROOT="$PWD/analysis/npu166"',
              'export PYTHONPATH="$PWD/analysis/npu166/hook${PYTHONPATH:+:$PYTHONPATH}"']
enabled='\n'.join(lines)+'\n';(HERE/'serve-enabled.sh').write_text(enabled)
for name in ['sitecustomize.py','controller.py']:compile((HERE/name).read_text(),name,'exec')
bundle={'controller.py':(HERE/'controller.py').read_text(),'hook/sitecustomize.py':(HERE/'sitecustomize.py').read_text(),'serve-enabled.sh':enabled}
payload=base64.b64encode(zlib.compress(json.dumps(bundle).encode())).decode()
code='PAYLOAD='+repr(payload)+'\n'+r'''import base64,zlib,json,subprocess,hashlib
from pathlib import Path
root=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/npu166')
root.mkdir(parents=True,exist_ok=True)
if (root/'SUBMITTED.json').exists():print(json.dumps({'existing_handle':True,**json.loads((root/'SUBMITTED.json').read_text())}));raise SystemExit(0)
bundle=json.loads(zlib.decompress(base64.b64decode(PAYLOAD)))
for name,text in bundle.items():
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
plan={'same_full_case':'C8 32768 input/1024output/P90','NPU_activities_only':True,'profiler_level':'Level0','PMU':False,'CPU_stack_memory_shapes':False,
      'max_iterations_per_worker_window':12,'windows':2,'startup_script_restored_after_ready':True,
      'live_own_service_retains_inactive_profile_capability_until_normal_restart':True,
      'limits':{'acquisition_seconds':2700,'restore_seconds':2400,'client_seconds':600,'directory_stop_threshold':16*1024**3,'profile_worker_file_limit_after_load':256*1024**2},
      'authorization':'User explicitly authorizes probes and this container inference/framework process operations.',
      'sources':[{'path':name,'bytes':len(text.encode()),'sha256':hashlib.sha256(text.encode()).hexdigest()} for name,text in bundle.items()]}
(root/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
log=(root/'controller.log').open('ab');p=subprocess.Popen(['python3','-I','-S',str(root/'controller.py')],stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True)
handle={'pid':p.pid,'start_ticks':Path('/proc/'+str(p.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19],'root':str(root),'plan':plan}
(root/'SUBMITTED.json').write_text(json.dumps(handle,indent=2)+'\n');print(json.dumps(handle))
'''
result=invoke('target',code)
(HERE/'device-only-submission.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
