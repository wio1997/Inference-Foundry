import json,subprocess,socket,pathlib,gzip,shutil,urllib.request,re
assert socket.gethostname()=='B4-33U-Atlas900RCKA3-32'
root=pathlib.Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/seed-live3-20261018/node32');model='4cb8a338da590f0d0fd624892b2af7a20884029c7418c05ada75f5dbf3bcb157';assert json.loads((root/'launch.json').read_text())['container_id']==model
for node,port in [(32,29481),(33,29482)]:
 with urllib.request.urlopen('http://172.16.12.%s:%s/metrics'%(node,port),timeout=5) as response:text=response.read().decode()
 for name in ['num_requests_running','num_requests_waiting','kv_cache_usage_perc']:
  values=re.findall(r'^vllm:'+name+r'(?:\{[^\n]*\})?\s+(\S+)',text,re.M);assert values and all(float(v)==0 for v in values),(node,name,values)
with urllib.request.urlopen('http://172.16.12.32:29882/healthcheck',timeout=5) as response:assert json.load(response)['request_num']==0
out=pathlib.Path(__file__).resolve().parent/'artifacts';out.mkdir(exist_ok=False);results=[]
for name,cid in [('official-proxy-v1','19dffcd43b09fe06e6c2e928d86341afd5f16332964d92fbd5a8b261808da4e9'),('first-token-proxy-v1','94e9c421d3cba926beb48ec927e01711d33cd08efa3ed10c4200a9afffd3bdf7'),('local-complete-proxy-v1','cfa1eade67ed5dda23ab25e599a5f6c86dfcc2f0b179770886f812ce7b1a7579')]:
 local=root/name;launch=json.loads((local/'launch.json').read_text());assert launch['CID']==cid;d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert d['Id']==cid and d['State']['Running'] and d['Image']=='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83' and not d['HostConfig'].get('Devices');expected=launch['argv'][launch['argv'].index('--name')+1];assert d['Name']=='/'+expected
 dest=out/name;dest.mkdir();shutil.copy2(local/'launch.json',dest/'launch.json')
 if (local/'proxy.py').exists():shutil.copy2(local/'proxy.py',dest/'proxy.py')
 stop=subprocess.run(['docker','stop','-t','20',cid],capture_output=True,text=True,timeout=30);after=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert stop.returncode==0 and not after['State']['Running']
 logs=subprocess.check_output(['docker','logs','--timestamps',cid],stderr=subprocess.STDOUT,timeout=20);assert len(logs)<64*1024**2
 with gzip.open(dest/'console.txt.gz','wb') as f:f.write(logs)
 results.append({'CID':cid,'name':d['Name'],'before':d['State'],'after':after['State'],'stop_returncode':stop.returncode,'console_bytes':len(logs)})
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'stopped':results,'P_model_kept_running':True,'D_model_kept_running':True,'scope':'Only three exact owned finite CPU proxies of current P; source/logs retained. No other workload or holder changed.'})+'\nGLM53_RESULT_JSON_END')
