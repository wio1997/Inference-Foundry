#!/usr/bin/env python3
"""Read-only allowlisted process flags and source-symbol search; no model requests."""
import datetime,hashlib,json,pathlib,shlex,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'runs/L003-R001-model-boundary';OUT.mkdir(parents=True,exist_ok=False)
INNER=r'''
import pathlib,json,hashlib,re
allowed={'--model-path','--model','--tokenizer-path','--hf-config-path','--model-impl','--load-format','--quantization','--kv-cache-dtype','--speculative-algorithm','--speculative-num-steps','--speculative-eagle-topk','--speculative-num-draft-tokens','--tp','--tp-size','--ep','--ep-size','--dp','--dp-size','--pp','--pp-size','--nnodes','--node-rank','--disaggregation-mode','--disaggregation-transfer-backend','--attention-backend','--moe-a2a-backend','--port','--trust-remote-code','--disable-overlap-schedule','--disable-cuda-graph'}
v={'processes':[],'symbol_matches':[]}
for proc in pathlib.Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:args=(proc/'cmdline').read_bytes().decode().split('\0')
 except (OSError,UnicodeDecodeError):continue
 if 'sglang.launch_server' not in args:continue
 flags={}
 for i,a in enumerate(args):
  key=a.split('=',1)[0]
  if key in allowed:
   flags[key]=a.split('=',1)[1] if '=' in a else (args[i+1] if i+1<len(args) and not args[i+1].startswith('--') else True)
  if key=='--json-model-override-args':
   try:
    x=json.loads(a.split('=',1)[1] if '=' in a else args[i+1]);flags[key]={k:x[k] for k in ['architectures','model_type'] if k in x}
   except:flags[key]='present; selective parsing failed'
 try:env=dict(a.split('=',1) for a in (proc/'environ').read_bytes().decode().split('\0') if '=' in a)
 except:env={}
 v['processes'].append({'pid':int(proc.name),'module':'sglang.launch_server','flags':flags,'PYTHONPATH':env.get('PYTHONPATH')})
roots=[pathlib.Path('/sgl-workspace/sglang/python/sglang/srt')]
for r in roots:
 for p in r.rglob('*.py'):
  try:b=p.read_bytes();text=b.decode()
  except:continue
  if 'GlmMoeDsaForCausalLM' not in text and 'glm_moe_dsa' not in text and '_disable_npu_fused_split_qk_norm' not in text:continue
  matches=[{'line':i,'text':l.strip()[:240]} for i,l in enumerate(text.splitlines(),1) if any(t in l for t in ['GlmMoeDsaForCausalLM','glm_moe_dsa','_disable_npu_fused_split_qk_norm'])]
  v['symbol_matches'].append({'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'matches':matches[:30]})
print(json.dumps(v))
'''
JUMP=r'''
import subprocess,json,sys
j=json.load(sys.stdin)
outer='import subprocess,sys; p=subprocess.run(["docker","exec","sglang-learning37-glm53-20261017","python3","-c",'+repr(j['inner'])+'],capture_output=True,text=True,timeout=60);sys.stdout.write(p.stdout);sys.stderr.write(p.stderr);sys.exit(p.returncode)'
p=subprocess.run(['kubectl','exec','-i','-n','default','sglang-tuning-access-37-20261013','-c','hostctl','--','nsenter','-t','1','-m','--','python3','-c',outer],capture_output=True,text=True,timeout=90)
sys.stdout.write(p.stdout);sys.stderr.write(p.stderr);sys.exit(p.returncode)
'''
cmd=['ssh','-S','/tmp/glm53-active-20261018.sock','-o','BatchMode=yes','-o','ConnectTimeout=8','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])]
started=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=json.dumps({'inner':INNER}),capture_output=True,text=True,timeout=120)
(OUT/'observation.json').write_text(p.stdout);(OUT/'stderr.log').write_text(p.stderr)
receipt={'started_utc':started,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'command':cmd,'exit_code':p.returncode,'read_only':True,'model_requests':0,'artifacts':{n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ['observation.json','stderr.log']}}
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'exit_code':p.returncode}));sys.exit(p.returncode)
