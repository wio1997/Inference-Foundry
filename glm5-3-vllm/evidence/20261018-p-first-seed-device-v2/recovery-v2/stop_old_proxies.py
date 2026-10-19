from pathlib import Path
import gzip,hashlib,json,subprocess,tarfile,io,socket
assert socket.gethostname()=='B4-33U-Atlas900RCKA3-32'
base=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/seed-live1-20261018/node32');out=Path(__file__).resolve().parent/'artifacts';out.mkdir(exist_ok=False);rows=[]
for sub,source,sha in [('official-proxy-v1','/vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py','85b8982948530758be655d3448ff10fc3979e56622221feace560272a81f1469'),('first-token-proxy-v1','/proxy.py','249ce0c04adfb0c971d55a1247b38a2eb7cc1e1acc057417d66ce90233719c96')]:
 launch=json.loads((base/sub/'launch.json').read_text());cid=launch['CID'];before=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert before['Id']==cid and before['Image']=='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83' and not before['HostConfig']['Devices'];assert before['Name'] in ['/glm53-official-proxy-32-seed-live2-20261018','/glm53-first-token-proxy-32-live2-20261018']
 dest=out/sub;dest.mkdir();raw=subprocess.check_output(['docker','cp',cid+':'+source,'-'],timeout=20)
 with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
  files=[x for x in archive.getmembers() if x.isfile()];assert len(files)==1;data=archive.extractfile(files[0]).read()
 assert hashlib.sha256(data).hexdigest()==sha;(dest/'proxy.py').write_bytes(data)
 q=subprocess.run(['docker','stop','-t','15',cid],capture_output=True,text=True,timeout=25);after=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert q.returncode==0 and not after['State']['Running'];logs=subprocess.run(['docker','logs','--timestamps',cid],capture_output=True,timeout=20);assert logs.returncode==0 and len(logs.stdout)+len(logs.stderr)<32*1024**2
 with gzip.open(dest/'console.txt.gz','wb') as f:f.write(logs.stdout+b'\n'+logs.stderr)
 for name in ['launch.json','result.json']:
  if (base/sub/name).exists():(dest/name).write_bytes((base/sub/name).read_bytes())
 rows.append({'cid':cid,'name':before['Name'],'source_sha256':sha,'before_state':before['State'],'after_state':after['State'],'compute_devices':[],'stop_returncode':q.returncode})
(out/'finish.json').write_text(json.dumps(rows,indent=2)+'\n');print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'proxies':rows,'scope':'Own exact CPU proxy CIDs only, no backend/holder or other jobs changed'})+'\nGLM53_RESULT_JSON_END')
