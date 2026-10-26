"""One installed-source question: actual vLLM D draft graph capability, no profiling/model calls."""
import ast,base64,gzip,hashlib,importlib.util,json,os,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RID='20261019T-L12B-vllm-D-spec-graph-installed-source-v2'
PROBE="import pathlib,json,hashlib,gzip,base64,sys,subprocess\nroots=['/vllm-workspace/vllm','/vllm-workspace/vllm-ascend'];paths=[]\nfiles={roots[0]:['vllm/config/speculative.py','vllm/v1/spec_decode/eagle.py','vllm/v1/spec_decode/mtp.py','vllm/v1/spec_decode/draft_model.py'],roots[1]:['vllm_ascend/spec_decode/eagle_proposer.py','vllm_ascend/spec_decode/mtp_proposer.py','vllm_ascend/spec_decode/draft_model_runner.py','vllm_ascend/worker/model_runner_v1.py']}\nmissing=[]\nfor root,names in files.items():\n for name in names:\n  p=pathlib.Path(root)/name\n  if p.is_file():paths.append(p)\n  else:missing.append(str(p))\nassert 1<=len(paths)<=8\nout={'project_id':'glm53','framework':'vllm','question':'Whether actual installed MTP draft model path can execute Graph under frozen D FULL_DECODE_ONLY config; no claim from targetGraph alone','files':[],'SDK_init_calls':0,'model_requests':0};total=0\nfor p in sorted(set(paths)):\n assert not p.is_symlink() and any(p.resolve().is_relative_to(pathlib.Path(root).resolve()) for root in roots)\n raw=p.read_bytes();total+=len(raw);assert len(raw)<=2*1024**2 and total<=12*1024**2\n out['files'].append({'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'source':raw.decode()})\nassert out['files'] and not any(k.startswith(('torch','vllm','sglang','memfabric_hybrid','memcache_hybrid')) for k in sys.modules)\nout['missing_whitelisted_files']=missing;out['source_bytes']=total;raw=json.dumps(out).encode();print(json.dumps({'sha256':hashlib.sha256(raw).hexdigest(),'decoded_bytes':len(raw),'gzip_base64':base64.b64encode(gzip.compress(raw,mtime=0)).decode()}))\n"

def plan():
 p=ROOT/'scripts/inspect_glm53_C32_indexer_partition_installed_sources_v1.py';s=importlib.util.spec_from_file_location('prior',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 fw,n,item,image,shared,_,_=m.plans()[0];assert fw=='vllm' and n==43
 lib=ROOT/'scripts/inventory_readonly.py';assert hashlib.sha256(lib.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777';s=importlib.util.spec_from_file_location('inventory',lib);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
 code=m.render(m.HOST,{'HOSTNAME':b.NODES[43][1],'CID':item['container_id'],'NAME':item['name'],'IMAGE':image,'SHARED_RID':shared,'PROBE':PROBE,'NODE':43});ast.parse(code);return b,code

def main():
 # During restored-P observation this adds one read-only SSH; startup signals must have completed.
 pending=json.loads((ROOT/'runs/20261019T-L12B-restored-L10-P-models-and-observe-v1/receipt.json').read_text());assert len(pending['starts'])==2 and all(v==0 for v in pending['starts'].values())
 b,code=plan();review=json.loads((ROOT/'results/C32-vllm-D-spec-graph-source-actual-review-v2.json').read_text());assert review['host_code_sha256']==hashlib.sha256(code.encode()).hexdigest()
 run=ROOT/'runs'/RID;run.mkdir(exist_ok=False);(run/'execution-once.lock').write_text(str(os.getpid()));receipt={'project_id':'glm53','max_additional_SSH':1,'model_requests':0,'SDK_init_calls':0,'remote_mutations':False}
 try:
  pod=b.NODES[43][0];mapped=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert mapped['rc']==0 and mapped['stdout']=='172.16.12.43'
  command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-']);wire=b.invoke([command],code,timeout=180);(run/'node-43.json').write_text(json.dumps(wire,indent=2)+'\n');receipt['node_rc']=wire['rc'];assert wire['rc']==0
  obj=json.loads(wire['stdout']);probe=obj['source_probe'];assert probe['rc']==0,probe['stderr'];packed=json.loads(probe['stdout']);raw=gzip.decompress(base64.b64decode(packed['gzip_base64'],validate=True));assert len(raw)==packed['decoded_bytes'] and hashlib.sha256(raw).hexdigest()==packed['sha256'];(run/'source-index.json').write_bytes(raw)
  out=json.loads(raw);dest=run/'source';dest.mkdir()
  for item in out['files']:
   raw=item['source'].encode();assert hashlib.sha256(raw).hexdigest()==item['sha256'];(dest/item['path'].strip('/').replace('/','__')).write_bytes(raw)
  receipt['source_files']=len(out['files']);receipt['source_bytes']=out['source_bytes'];receipt['pass']=True
 finally:(run/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt))
if __name__=='__main__':main()
