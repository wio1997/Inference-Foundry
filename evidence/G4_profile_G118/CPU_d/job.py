from pathlib import Path
import subprocess,json,os,datetime
r=Path('<TASK_ROOT>/vllm_G4_CPU_profile_g130')
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[-2:]));os.nice(10)
for index,c in enumerate([['python3', '/task/vllm_G4_CPU_profile_g130/package/parallel_profile_archive.py', '--root', '/task/vllm_g4/profiles/d', '--identity', '/task/vllm_G4_CPU_profile_g130/identity.json', '--stability-receipt', '/task/vllm_G4_CPU_profile_g130/raw_stability_d.json', '--out', '/task/vllm_G4_CPU_profile_g130/analysis_manifest.json', '--work-root', '/task/vllm_G4_CPU_profile_g130/analysis_copies', '--analyse'], ['python3', '/task/vllm_G4_CPU_profile_g130/package/analyse_role.py', '--root', '/task/vllm_G4_CPU_profile_g130', '--role', 'd']]):
 result=subprocess.run(['docker','exec','-e','TORCH_DEVICE_BACKEND_AUTOLOAD=0','yxy-glm53-vllm-d-g4']+c,cwd=r)
 (r/('stage_'+str(index)+'_returncode.json')).write_text(json.dumps({'rc':result.returncode,'argv':c,'at':datetime.datetime.now(datetime.timezone.utc).isoformat()}))
 assert result.returncode==0
