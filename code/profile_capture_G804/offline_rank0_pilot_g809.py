"""Bounded CPU-only SDK parsing on a private copy of one captured rank."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,re,shutil,socket,subprocess,time
host=socket.gethostname();assert host in ('<ROLE_HOSTNAME>','<ROLE_HOSTNAME>')
root=Path('/PRIVATE_TASK_ROOT');r=root/'vllm_pd_g727';o=r/'offline_rank0_g809';o.mkdir(exist_ok=False)
lock=(o/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);began=time.monotonic()
def save(name,value):
 p=o/name;q=p.with_suffix('.tmp');q.write_text(json.dumps(value,indent=2)+'\n');q.replace(p)
def state(phase,**more):
 d={'phase':phase,'pid':os.getpid(),'host':host,'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-began,**more};save('state.json',d);print(json.dumps(d),flush=True)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
try:
 s=json.loads((r/'state.json').read_text());assert s['generation']==727 and str(r/'node_controller.py').encode() in Path('/proc',str(s['pid']),'cmdline').read_bytes()
 c=json.loads(subprocess.check_output(['docker','inspect',s['container_id']]))[0];assert c['State']['Running'] and c['Image']=='sha256:d5492ecfdb3071ff2d4bae3f849245014c54b63035d9eefb35243869461ddae1' and c['HostConfig']['CpusetCpus']=='0-159'
 dirs=list((r/'profiles').glob('*_ascend_pt'));assert len(dirs)==8
 ranks={int(re.search(r'_rank(\d+)_',p.name).group(1)):p for p in dirs};assert set(ranks)==set(range(8));source=ranks[0]
 assert (source/'profiler_info_0.json').is_file() and (source/'FRAMEWORK/torch.op_range').stat().st_size>0
 total=sum(p.stat().st_size for p in source.rglob('*') if p.is_file());assert shutil.disk_usage(r).free>total*3+20*1024**3
 state('hashing_original_rank0_before_isolated_copy',source=str(source),raw_bytes=total,CPU_affinity='156-159',SDK_max_process_number=1,CPU_only_offline_no_new_model_requests=True)
 frozen={str(p.relative_to(source)):{'sha256':sha(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in source.rglob('*') if p.is_file()};save('original_raw_SHA256.json',frozen)
 copied=o/source.name;code=subprocess.run(['cp','-a','--reflink=auto',str(source),str(copied)],cwd=root,timeout=180).returncode;assert code==0
 for name,item in frozen.items():assert sha(copied/name)==item['sha256']
 save('copy_receipt.json',{'original':str(source),'copy':str(copied),'files':len(frozen),'bytes':total,'all_source_copy_SHA256_match':True,'method':'cp-a-reflink-auto; separateinodeandcopy-on-writeornormalcopy; nohardlinks; SDKwritesonlyderivedcopy'})
 tmp=o/'tmp';tmp.mkdir();inside='/task/vllm_pd_g727/offline_rank0_g809/'+source.name
 code='from torch_npu.profiler.profiler import analyse; analyse(profiler_path='+repr(inside)+', max_process_number=1)'
 cmd=['docker','exec','-e','TMPDIR=/task/vllm_pd_g727/offline_rank0_g809/tmp','-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1',c['Id'],'taskset','-c','156-159','timeout','--signal=TERM','--kill-after=30','600','python3','-u','-c',code]
 with (o/'sdk_parse.log').open('x') as stream:
  p=subprocess.Popen(cmd,cwd=root,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);save('actual_parse_handle.json',{'pid':p.pid,'argv':cmd,'container_id':c['Id'],'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'inner_bound_s':600,'scope':'CPU-onlyofflineSDKanalyse; nativecapturedrawcopiedandSHAchecked; existingmodelcontroller/guardretained; noNPUrequests'})
  state('SDK_rank0_parsing',docker_exec_pid=p.pid);returncode=p.wait(timeout=660)
 original_unchanged=all((source/name).is_file() and sha(source/name)==item['sha256'] and (source/name).stat().st_size==item['bytes'] and (source/name).stat().st_mtime_ns==item['mtime_ns'] for name,item in frozen.items())
 assert original_unchanged,'Originalrawchanged; preserveandinspect'
 output=copied/'ASCEND_PROFILER_OUTPUT';files=[{'relative':str(p.relative_to(output)),'bytes':p.stat().st_size} for p in output.rglob('*') if p.is_file()]
 summary={'returncode':returncode,'original_raw_SHA256_unchanged':original_unchanged,'output_files':files,'output':str(output),'trace_view_present':(output/'trace_view.json').is_file(),'op_statistic_present':(output/'op_statistic.csv').is_file(),'actual_device_and_logical_coverage_accepted':False,'scope':'SDKsuccess/outputfiles alone insufficient; parent mustinspect actualdeviceevents andds41.scheduler logicalannotations. Never recapturebecauseparserneedsreview.'};save('result.json',summary)
 assert returncode==0 and summary['trace_view_present'] and summary['op_statistic_present'],summary
 state('pilot_SDK_complete_pending_device_logical_review',**summary)
except BaseException as error:
 state('failed_requires_review',error=repr(error),scope='Preserveoriginalraw/copy/log/actualhandle. Inspectbeforeanyretry; noNPUrecapture.');raise
