import hashlib,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
root=Path(__file__).parent;deploy=Path("/data/tiankuan/wio/glm52-pd/deploy");out={"at":utc(),"nodes":{},"model_content":"small files hashed; full weight content unknown"}
code="import subprocess,hashlib,json,pathlib; r={}; r['vllm_head']=subprocess.check_output(['git','-C','/vllm-workspace/vllm','rev-parse','HEAD']).decode().strip();r['ascend_head']=subprocess.check_output(['git','-C','/vllm-workspace/vllm-ascend','rev-parse','HEAD']).decode().strip();r['ascend_diff']=subprocess.check_output(['git','-C','/vllm-workspace/vllm-ascend','diff']).decode();r['attention_sha256']=hashlib.sha256(pathlib.Path('/vllm-workspace/vllm-ascend/vllm_ascend/attention/sfa_v1.py').read_bytes()).hexdigest();print(json.dumps(r))"
for node in ['166','167']:
 def command(argv):
  if node=='167':argv=['ssh','-o','BatchMode=yes','root@172.16.10.167',__import__('shlex').join(argv)]
  p=subprocess.run(argv,capture_output=True,text=True);return {'argv':argv,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
 entries={'image':command(['docker','inspect','--format','{{.Config.Image}} {{.Image}}','glm52-single']),'source':command(['docker','exec','glm52-single','python3','-c',code]),'processes':command(['docker','exec','glm52-single','ps','-eo','pid,ppid,args'])}
 for k,v in entries.items():(root/(node+'_'+k+'.snapshot.json')).write_text(json.dumps(v,indent=2)+'\n')
 out['nodes'][node]={k:{'path':str(root/(node+'_'+k+'.snapshot.json')),'exit_code':v['exit_code']} for k,v in entries.items()}
 if any(v['exit_code']!=0 for v in entries.values()):raise RuntimeError('identity capture failed')
for n in ['config.json','tokenizer_config.json','quant_model_description.json']:
 p=Path('/data/tiankuan/wio/GLM-5.2-w8a8')/n
 if p.exists():out.setdefault('model_small_files',[]).append({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
atomic_json(root/'identity.json',out)
