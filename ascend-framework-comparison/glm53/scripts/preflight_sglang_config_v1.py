#!/usr/bin/env python3
"""Exact B103 config/model-metadata preflight on idle34; no engine or model call."""
import hashlib,importlib.util,json,shlex
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('b',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs/20261017T0860-sglang-config-preflight-v1';IMAGE='sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263'
FROZEN={'34': '33c89937d11179d1ca0109dc1356e723de7e3ce46f63fc09c721d63c3fdaaf1f', '35': '9b769b5adf130ed673d49911deb6b2c769361b7aded13e777f55c6927b873476', '36': '54f01d92adde9862e5d0307b600d0acfe27750bab7db6da97bf59afd1cc1bc9d', '41': '85a2d31f0ad346c612ab666c6d4e8e17f32c37212eba43df3671c1b0c22946c2'}
for n,value in FROZEN.items():assert hashlib.sha256((b.ROOT/f'config/sglang-correctness-v1/node-{n}.json').read_bytes()).hexdigest()==value
CONFIGS={str(n):json.loads((b.ROOT/f'config/sglang-correctness-v1/node-{n}.json').read_text()) for n in (34,35,36,41)}
CODE=r'''
import pathlib,json,hashlib
root=pathlib.Path('/sgl-workspace/sglang/python/sglang/srt');sources=[]
paths=list((root/'layers/quantization').glob('*modelslim*'))+[root/'speculative/eagle_utils.py',root/'speculative/spec_registry.py',root/'arg_groups/fields/attention.py']
for p in paths:
 candidates=list(p.rglob('*.py')) if p.is_dir() else [p]
 for f in candidates:
  if f.is_file() and f.stat().st_size<650000:
   raw=f.read_bytes();sources.append({'path':str(f),'sha256':hashlib.sha256(raw).hexdigest(),'text':raw.decode()})
print('GLM53_SGLANG_EXTRA_SOURCE='+json.dumps(sources),flush=True)
import memcache_hybrid as mc
import memfabric_hybrid as mf
from sglang.srt.server_args import prepare_server_args
from sglang.srt.arg_groups.overrides import resolved_view
from sglang.srt.runtime_context import derive_attention_widths
from sglang.srt.configs.model_config import ModelConfig
summary=[]
for node,intended in CONFIGS.items():
 args=prepare_server_args(intended['launch_args']);cfg=resolved_view(args)
 fields=['model_path','served_model_name','device','dtype','quantization','kv_cache_dtype','tp_size','dp_size','ep_size','nnodes','node_rank','enable_dp_attention','attn_cp_size','dcp_size','pp_size','mem_fraction_static','max_running_requests','chunked_prefill_size','max_prefill_tokens','context_length','page_size','disaggregation_mode','disaggregation_transfer_backend','disable_cuda_graph','disable_overlap_schedule','speculative_algorithm','speculative_num_steps','speculative_eagle_topk','speculative_num_draft_tokens','enable_hierarchical_cache','hicache_size','hicache_mem_layout','hicache_io_backend','hicache_storage_backend','hicache_storage_prefetch_policy']
 actual={key:getattr(cfg,key) for key in fields};dp,tp=derive_attention_widths(tp_size=cfg.tp_size,attn_cp_size=cfg.attn_cp_size,dp_size=cfg.dp_size,enable_dp_attention=cfg.enable_dp_attention)
 assert cfg.tp_size==32 and cfg.nnodes==2 and cfg.ep_size==32 and dp==intended['parallel_intent']['attention_dp'] and tp==intended['parallel_intent']['attention_tp']
 assert cfg.device=='npu' and cfg.quantization=='modelslim' and cfg.disaggregation_transfer_backend=='ascend'
 assert cfg.speculative_num_steps==5 and cfg.speculative_num_draft_tokens==6 and cfg.speculative_eagle_topk==1
 assert cfg.disable_cuda_graph and cfg.disable_overlap_schedule
 model=ModelConfig.from_server_args(args)
 assert model.hf_config.architectures==['GlmMoeDsaForCausalLM']
 summary.append({'node':node,'actual':actual,'attention_tp':tp,'attention_dp':dp,'model_architectures':model.hf_config.architectures,'model_dtype':str(model.dtype),'is_mla':getattr(model,'is_deepseek_mla',None)})
print('GLM53_SGLANG_PREFLIGHT='+json.dumps({'configs':summary,'model_requests':0,'engine_created':False,'weight_payload_read':False,'sdk_paths':[mf.__file__,mc.__file__],'sdk_versions':[pathlib.Path(mf.__file__).with_name('VERSION').read_text(),pathlib.Path(mc.__file__).with_name('VERSION').read_text()]}),flush=True)
'''
def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 pod,host=b.NODES[34];m=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert m['rc']==0 and m['stdout']=='172.16.12.34'
 code='CONFIGS='+repr(CONFIGS)+'\n'+CODE
 args=['docker','run','--rm','--pull=never','--network=none','--shm-size=1g','--memory=8g','--cpus=4','--pids-limit=1024','--label','project_id=glm53','--name','glm53-sglang-config-preflight-v1-20261017-34','-v','/data/wio/models/GLM-5.3-w8a8:/data/wio/models/GLM-5.3-w8a8:ro','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','PYTHONDONTWRITEBYTECODE=1','-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','-e','HF_HOME=/tmp/hf','-e','XDG_CACHE_HOME=/tmp/cache']
 for dev in [f'/dev/davinci{i}' for i in range(16)]+['/dev/davinci_manager','/dev/devmm_svm','/dev/hisi_hdc']:args+=['--device',dev]
 args+=['--entrypoint','timeout',IMAGE,'180','python3','-c',code]
 payload='''import socket,subprocess,json,sys
assert socket.gethostname()==HOST
r=subprocess.run(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],capture_output=True,text=True,timeout=15);assert r.returncode==0 and r.stdout.strip()==IMAGE
before=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30);assert before.returncode==0 and before.stdout.count('No running processes')==8,'Node34 occupied; no preflight'
r=subprocess.run(ARGS,capture_output=True,text=True,timeout=200);print(r.stdout,end='');print(r.stderr,file=sys.stderr,end='')
after=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30);print('GLM53_POSTCHECK_JSON='+json.dumps({'rc':after.returncode,'npu_after':after.stdout}),flush=True)
sys.exit(r.returncode)
'''
 import re
 for k,v in [('HOST',host),('IMAGE',IMAGE),('ARGS',args)]:payload=re.sub(r'\b'+k+r'\b',lambda _:repr(v),payload)
 cmd=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
 r=b.invoke([cmd],payload,timeout=270);(RUN/'node-34.json').write_text(json.dumps(r,indent=2)+'\n');(RUN/'receipt.json').write_text(json.dumps({'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'config_sha256':FROZEN,'rc':r['rc'],'model_requests':0},indent=2)+'\n');print('sglang preflight',r['rc'],flush=True)
if __name__=='__main__':main()
