"""One current-runtime32-call functional matrix; performance remains unmeasured."""
import ast,hashlib,importlib.util,json,re
from pathlib import Path
P=Path(__file__).resolve()
def load(name,file,sha):
 path=P.with_name(file);assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
base=load('pool','start_vllm_shared_pool_v4.py','33869c715fc71e7fdcd312e500f08733da7722a0e870a723726d19cd6bd6becf')
identity=load('identity','glm53_vllm_shared_model_identity_v3.py','24a93a46cfc19946e4e52cc2d255e92ea552c127b560d12ee1fea7108a18fb89')
RUN=base.b.ROOT/'runs/20261017T1772-vllm-shared-short-matrix-v1'
PAYLOAD=P.with_name('probe_glm53_pd_matrix_payload_v5.py').read_text();PAYLOAD_SHA='75ff3e09e7b91119502ca83c05758393df839d9803c4f50086f8aa6aa01920f7';assert hashlib.sha256(PAYLOAD.encode()).hexdigest()==PAYLOAD_SHA

def render(code,values):
 pattern=r'\b(?:'+'|'.join(re.escape(k) for k in values)+r')\b';text=re.sub(pattern,lambda m:repr(values[m[0]]),code);ast.parse(text);return text

def check(node,item):
 code=identity.HOST+r'''
import urllib.request
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(URL,timeout=10) as response:assert response.status==200;raw=response.read(8388609);assert len(raw)<=8388608
rows=[float(l.rsplit(' ',1)[1]) for l in raw.decode().splitlines() if l.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{'))];assert rows and all(x==0 for x in rows)
print(json.dumps({'node':NODE,'container_id':CID,'model_identity':model_identity,'idle_native_gauges':rows}))
'''
 return base.invoke(node,render(code,{'CID':item['container_id'],'NAME':item['name'],'IMAGE':base.IMAGE,'NODE':node,'URL':f'http://172.16.12.{node}:'+('30300' if node in [30,31] else '30400')+'/metrics'}),120)

def execute(item):
 code=identity.HOST+r'''
import sys
try:
 result=subprocess.run(['docker','exec','-i',CID,'env','PYTHONDONTWRITEBYTECODE=1','TOKENIZERS_PARALLELISM=false','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','timeout','600','python3','-'],input=PAYLOAD,capture_output=True,text=True,timeout=660)
 print(result.stdout,end='');print(result.stderr,file=sys.stderr,end='');rc=result.returncode
except subprocess.TimeoutExpired as error:
 for raw,stream in [(error.stdout,sys.stdout),(error.stderr,sys.stderr)]:
  if raw:print(raw.decode(errors='replace') if isinstance(raw,bytes) else raw,file=stream,end='')
 rc=124
print('GLM53_PROBE_PROCESS_RC='+str(rc),flush=True)
raise SystemExit(rc)
'''
 return base.invoke(30,render(code,{'CID':item['container_id'],'NAME':item['name'],'IMAGE':base.IMAGE,'NODE':30,'PAYLOAD':PAYLOAD}),720)

def main():
 evidence=base.b.ROOT/'results/vllm-shared-startup-v1.json';assert hashlib.sha256(evidence.read_bytes()).hexdigest()=='db64b2127c6ec3f84e6ec273cf0339ab62c0aec8e9ef9e8337779c19589ca953';qualified=json.loads(evidence.read_text());assert qualified['pass'] and qualified['total_model_SDK_SDMA_init']==64
 cap=json.loads((base.b.ROOT/'results/vllm-shared-pool-capacity-v4.json').read_text())
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'payload_sha256':PAYLOAD_SHA,'precheck':{},'rc':None,'matrix_result':None,'formal_requests':0,'phase':'FUNCTIONAL_ONLY_REQUIRES_NATIVE_POST_AUDIT'}
 try:
  for n in [30,31,43,45]:
   r=check(n,cap['providers'][str(n)]);(RUN/f'precheck-{n}.json').write_text(json.dumps(r,indent=2)+'\n');receipt['precheck'][str(n)]=r['rc'];assert r['rc']==0
  (RUN/'dispatch.json').write_text(json.dumps(receipt,indent=2)+'\n')
  r=execute(cap['providers']['30']);(RUN/'client-30.json').write_text(json.dumps(r,indent=2)+'\n');receipt['rc']=r['rc']
  rows=[l.split('=',1)[1] for l in r['stdout'].splitlines() if l.startswith('GLM53_MATRIX_RESULT=')]
  if len(rows)==1:receipt['matrix_result']=json.loads(rows[0])
 finally:(RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print('short matrix',receipt['rc'],flush=True)
if __name__=='__main__':main()
