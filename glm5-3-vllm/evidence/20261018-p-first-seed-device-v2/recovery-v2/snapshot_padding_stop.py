import gzip,hashlib,json,pathlib,re,shutil,socket,subprocess,urllib.request
assert socket.gethostname()=='B4-33U-Atlas900RCKA3-32'
root=pathlib.Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/prefill-padding-live1-20261018/node32');cid='28e687e363317e918ed636a4615f918dda3ea80839626e357e82d77ea87faf56';proxy='89264f730eebcae1aa26589aca139e9da449b776668ecac6adc802ed87187d30';image='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
plan=json.loads((root/'plan.json').read_text());launch=json.loads((root/'launch.json').read_text());assert launch['container_id']==cid
assert not (root/'prefill-padding-probe/enable').exists()
for n,port in [(32,29481),(33,29482)]:
 with urllib.request.urlopen('http://172.16.12.%d:%d/metrics'%(n,port),timeout=5) as f:s=f.read().decode()
 for name in ['num_requests_running','num_requests_waiting','kv_cache_usage_perc']:
  v=re.findall(r'^vllm:'+name+r'(?:\{[^\n]*\})?\s+(\S+)',s,re.M);assert v and all(float(x)==0 for x in v),(n,name,v)
with urllib.request.urlopen('http://172.16.12.32:29880/healthcheck',timeout=5) as f:assert json.load(f)['request_num']==0
pd=json.loads(subprocess.check_output(['docker','inspect',proxy]))[0];md=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
pl=json.loads((root/'official-proxy-v1/launch.json').read_text());assert pl['CID']==proxy
assert pd['State']['Running'] and pd['Image']==image and not pd['HostConfig'].get('Devices') and pd['Name']=='/'+pl['argv'][pl['argv'].index('--name')+1]
assert md['State']['Running'] and md['Image']==image and md['Name']=='/'+plan['container']['name']
assert json.loads((root/'context.json').read_text())['client']['lease']['uid']==plan['reservation']['expected_uid']=='1f257348-8296-4fa5-8bc9-1791e1553b29'
out=pathlib.Path(__file__).resolve().parent/'artifacts';out.mkdir(exist_ok=False)
files=[p for p in root.rglob('*') if p.is_file() and p.suffix in ['.json','.jsonl','.py'] and not p.is_symlink()];total=sum(p.stat().st_size for p in files);assert total<128*1024**2,total
for p in files:
 q=out/'P-root'/p.relative_to(root);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
for patch in plan['runtime_patches']:
 p=pathlib.Path(patch['source']);b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==patch['sha256'];q=out/'overlay-sources'/p.name;q.parent.mkdir(exist_ok=True);q.write_bytes(b)
results=[]
for name,ident,before in [('official-proxy',proxy,pd),('P-model',cid,md)]:
 q=subprocess.run(['docker','stop','-t','30',ident],capture_output=True,text=True,timeout=45);after=json.loads(subprocess.check_output(['docker','inspect',ident]))[0];assert q.returncode==0 and not after['State']['Running']
 b=subprocess.check_output(['docker','logs','--timestamps',ident],stderr=subprocess.STDOUT,timeout=20);assert len(b)<64*1024**2
 with gzip.open(out/(name+'-console.txt.gz'),'wb') as f:f.write(b)
 results.append({'name':name,'CID':ident,'before':before['State'],'after':after['State'],'console_bytes':len(b),'rc':q.returncode})
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'stopped':results,'snapshot_files':len(files),'snapshot_bytes':total,'D_kept_running':True,'holder_kept':True,'scope':'Only exact owned P+CPU proxy; finite candidate replacement, no other jobs stopped.'})+'\nGLM53_RESULT_JSON_END')
