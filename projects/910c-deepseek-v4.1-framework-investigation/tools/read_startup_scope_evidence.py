#!/usr/bin/env python3
"""Read current source references/config and bounded completed startup log lines."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess

REMOTE=r'''
from pathlib import Path
from hashlib import sha256
import json,subprocess,sys,shlex
name='wio-vllm-v41-repro171-20261102';item,=json.loads(subprocess.check_output(['docker','inspect',name]));assert item['State']['Running'] and item['Image']=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
inner=r"""from pathlib import Path
from hashlib import sha256
import json,re,shutil,subprocess
roots=[Path('/vllm-workspace/vllm/vllm'),Path('/vllm-workspace/vllm-ascend/vllm_ascend')]
pattern=r'current_workspace_manager|use_workspace_lane|init_workspace_manager|reset_workspace_manager|\.get_simultaneous\('
paths=set();rg=shutil.which('rg')
if rg:
 r=subprocess.run([rg,'-l','-g','*.py',pattern,*map(str,roots)],capture_output=True,text=True);assert r.returncode in [0,1];paths={Path(s) for s in r.stdout.splitlines()}
else:
 for root in roots:
  for p in root.rglob('*.py'):
   if re.search(pattern,p.read_text()):paths.add(p)
refs=[]
for p in sorted(paths):
 raw=p.read_bytes();refs.append(dict(runtime_path=str(p),sha256=sha256(raw).hexdigest(),matches=[dict(line=i,text=s.strip()) for i,s in enumerate(raw.decode().splitlines(),1) if re.search(pattern,s)]))
sources=[]
for p in [roots[0]/'models/deepseek_v41/common/mm_preprocess.py',roots[0]/'v1/worker/startup_plan.py',roots[0]/'transformers_utils/configs/deepseek_v41.py',roots[1]/'models/deepseek_v41/vl_model.py',roots[1]/'models/deepseek_v41/vision.py',roots[1]/'ascend_config.py']:
 b=p.read_bytes();sources.append(dict(runtime_path=str(p),sha256=sha256(b).hexdigest(),content=b.decode()))
p=Path('/data/tiankuan/wio/DeepSeek-V4.1-Flash-1017-w8a8/config.json');b=p.read_bytes();d=json.loads(b)
keys=['model_type','architectures','num_hidden_layers','vocab_size','index_topk','index_n_heads','index_head_dim','compress_ratios','rms_norm_eps','vision_n_layers','vision_width','vision_n_heads','vision_image_size','qk_nope_head_dim','qk_rope_head_dim','hidden_size','num_attention_heads']
cfg=dict(path=str(p),sha256=sha256(b).hexdigest(),bytes=len(b),selected={k:d.get(k) for k in keys},raw_config=d)
print(json.dumps(dict(workspace_references=refs,workspace_search=dict(roots=list(map(str,roots)),pattern=pattern,engine='rg' if rg else 'stdlib fallback',scope='Literal source references only; not dynamic call graph or active-use proof'),sources=sources,model_config=cfg)))
"""
r=subprocess.run(['docker','exec',name,'python3','-B','-I','-S','-c',inner],capture_output=True,text=True)
if r.returncode:print(r.stderr,file=sys.stderr);sys.exit(r.returncode)
x=json.loads(r.stdout)
serve=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/scripts/serve.sh');b=serve.read_bytes();ts=shlex.split(b.decode().replace('\\\n',' '),comments=True)
allow={'--language-model-only','--limit-mm-per-prompt','--skip-mm-profiling','--mm-processor-kwargs','--kv-cache-memory','--kv-cache-memory-bytes','--additional-config','--enable-mm-embeds'};opts={}
for i,t in enumerate(ts):
 if t.split('=')[0] in allow:opts[t.split('=')[0]]=t.split('=',1)[1] if '=' in t else (ts[i+1] if i+1<len(ts) and not ts[i+1].startswith('--') else True)
x['saved_launch']=dict(path=str(serve),sha256=sha256(b).hexdigest(),selected_flags=opts)
lp=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/framework-profile-20261109T192807Z/service.log')
if lp.is_file():
 st=lp.stat();assert st.st_size<128*1024*1024;digest=sha256();selected=[];n=0
 wanted=['Triton warmup','Triton kernel warmup','Early kernel warmup','Encoder cache will be initialized','Skipping encoder profiling','All non-text modalities','Workspace locked','Starting to load model','Model loading took','Graph capturing finished']
 with lp.open('rb') as f:
  for line in f:
   digest.update(line);n+=1;s=line.decode(errors='replace').strip()
   if any(w in s for w in wanted):selected.append(dict(line=n,text=s))
 assert (st.st_size,st.st_mtime_ns)==(lp.stat().st_size,lp.stat().st_mtime_ns)
 x['completed_startup_log']=dict(path=str(lp),sha256=digest.hexdigest(),bytes=st.st_size,mtime_ns=st.st_mtime_ns,total_lines=n,selected=selected[:512],matched_lines=len(selected),scope='Existing completed diagnostic clone startup, not current health or original startup latency')
x.update(container_id=item['Id'],image=item['Image'],model_weights_loads=0,CPU_tokenizer_loads=0,new_NPU_work=0,new_profile_collection=0,service_requests=0,remote_writes=[]);print(json.dumps(x))
'''


def main():
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE],capture_output=True,text=True,timeout=100)
    if r.returncode:raise RuntimeError(r.stderr[-3000:])
    x=json.loads(r.stdout);out=Path('audit/execution-search')
    for row in x['sources']:
        content=row.pop('content');p=out/'startup-warmup-runtime-source'/row['runtime_path'].removeprefix('/vllm-workspace/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content);assert sha256(p.read_bytes()).hexdigest()==row['sha256'];row['local_path']=str(p)
    x['reader_sha256']=sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'startup-scope-evidence.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(workspace_reference_files=len(x['workspace_references']),model_config=x['model_config'],saved_launch=x['saved_launch'],log_rows=x.get('completed_startup_log',{}).get('matched_lines'),new_NPU_work=0,remote_writes=[]),ensure_ascii=False))


if __name__=='__main__':main()
