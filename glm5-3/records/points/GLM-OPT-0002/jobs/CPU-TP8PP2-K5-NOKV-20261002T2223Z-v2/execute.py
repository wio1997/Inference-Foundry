from pathlib import Path
import json,sys,subprocess,shlex,hashlib
sys.path.insert(0,'/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime')
from phase_runner import atomic_json,utc
j=Path(__file__).parent;p=j.parents[1];old=p/'runs/GLM-RUN-0109';x=next(a for a in json.loads((old/'planned_launch.json').read_text())if a['node']=='167');argv=list(x['argv'])
k=argv.index('--speculative-config')+1;spec=json.loads(argv[k]);spec['num_speculative_tokens']=5;argv[k]=json.dumps(spec,separators=(',',':'))
k=argv.index('--compilation-config')+1;comp=json.loads(argv[k]);comp['cudagraph_capture_sizes']=[6,12,24,48];comp['max_cudagraph_capture_size']=48;argv[k]=json.dumps(comp,separators=(',',':'))
atomic_json(j/'candidate_argv.json',argv)
dest='/data/tiankuan/wio/glm52-pd/deploy/private/tp8pp2_k5_cpu_20261002T2223Z_v2';source=(j/'cpu_probe.py').read_text()
code="from pathlib import Path;import sys;p=Path(sys.argv[1]);p.mkdir();(p/'cpu_probe.py').write_text(sys.stdin.read())"
z=subprocess.run(['ssh','-o','BatchMode=yes','root@172.16.10.167',shlex.join(['python3','-c',code,dest])],input=source.encode(),capture_output=True,timeout=30);(j/'copy.stderr').write_bytes(z.stderr);z.check_returncode()
plug=Path(argv[1]).parent;env=dict(x['env']);env['VLLM_PP_LAYER_PARTITION']='42,36';shell='ulimit -c 0; export PD_LOCAL_IP=172.16.10.167 PD_NIC=business; source /data/tiankuan/wio/glm52-pd/deploy/scripts/pd_common_env.sh; export PYTHONPATH='+str(plug)+':$PYTHONPATH VLLM_HOST_IP=172.16.10.167 '+' '.join(k+'='+shlex.quote(v)for k,v in env.items())+'; python3 '+str(plug/'native_acl_lifecycle.py')+' script '+dest+'/cpu_probe.py '+shlex.quote(json.dumps(argv))
cmd=['ssh','-o','BatchMode=yes','root@172.16.10.167',shlex.join(['docker','exec','glm52-single','bash','-c',shell])];atomic_json(j/'command.json',dict(at=utc(),argv=cmd,GPU_workers_started=0))
z=subprocess.run(cmd,capture_output=True,timeout=240);(j/'cpu.stdout').write_bytes(z.stdout);(j/'cpu.stderr').write_bytes(z.stderr);z.check_returncode()
acks=[json.loads(l)for l in z.stdout.decode().splitlines()if l.startswith('{"event":')]
assert acks[0]['event']=='task_acl_init'and acks[-1]['event']=='task_acl_finalize'and acks[0]['returncode']==acks[-1]['returncode']==0
cfg=next(a for a in acks if a['event']=='full_native_CLI_API_config_valid');geometry=None;assert cfg['DCP']==8 and cfg['NPU_workers_started']==0 and cfg['DSA_CP_native_proof']['enable_expert_parallel'] is False
out=dict(at=utc(),native_CPU_config=cfg,CP_geometry=geometry,actual_SDK_init=acks[0],actual_SDK_finalize=acks[-1],candidate_argv=argv,models=0,requests=0,limitations=['Native EngineArgs/CLI and extracted original pureCPgeometry only; no allocations/workers/model/GPUrequest','3GiB PP2 candidate cache bytes do not prove actual KVcapacity/startupGraphfit/PDreshard correctness or E2Egain','Use16rankprofile conclusion to choose subsequent candidate; noCurrent/KEEP/capacity'])
atomic_json(j/'reduction.json',out);b=(j/'reduction.json').read_bytes();atomic_json(j/'result.json',dict(schema_version=1,job_id=j.name,status='completed',summary='NativeTP8PP2DCP8/EPdisabled_AllGather/KVconnectorNone/noSP/MTP5/Graph6,12,24,48/KV3GiB CPUCLI/SDK0 andnoKVconnector valid; no modeloperation',execution=dict(inner_exit_code=0,acceptance='passed',processes=[]),findings=[],evidence=[dict(id='reduction',path=str(j/'reduction.json'),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator='actualCLI/SDK0/sourcehash/CPgeometry')],unknowns=out['limitations'],decision_request=None,next_check_at=None));print(json.dumps(dict(DCP=8,geometry_rows=0,models=0,requests=0)))
