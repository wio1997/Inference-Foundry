import pathlib,json,subprocess,re,time
r=pathlib.Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030');latest=json.loads((r/'moe-observer-latest.json').read_text());root=pathlib.Path(latest['run_root']);out={}
for n in ['controller-state.json','controller-error.json','top-level-error.json','npu-monitor-error.json','external-npu-detected.json','restore-result.json','controller-finished.json','release-before-restore.json']:
 p=root/n
 if p.exists():out[n]=json.loads(p.read_text())
for name in ['wio-vllm-v41-repro166-20261030','wio-vllm-v41-'+latest['run_id']]:
 q=subprocess.run(['docker','inspect',name],capture_output=True,text=True)
 if q.returncode==0:
  c=json.loads(q.stdout)[0];out[name]={'id':c['Id'],'state':c['State']}
p=root/'runtime/logs/service.log';lines=p.read_text(errors='replace').splitlines();hits=[i for i,l in enumerate(lines) if any(x in l for x in ['Traceback (most recent call last)','AssertionError','ValueError:','RuntimeError:','TypeError:'])];selected=[]
for i in hits[:12]:selected.append({'line':i+1,'context':lines[max(0,i-3):min(len(lines),i+16)]})
out['root_cause_contexts']=selected;out['log_tail']=lines[-10:]
base=pathlib.Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/logs/service.log')
with base.open('rb') as f:
 f.seek(max(0,base.stat().st_size-4000));out['base_log_tail']=f.read().decode(errors='replace').splitlines()[-8:]
print(json.dumps(out,ensure_ascii=False))
