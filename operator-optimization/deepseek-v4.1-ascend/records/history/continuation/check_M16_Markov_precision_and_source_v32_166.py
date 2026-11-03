from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
M=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18');OUT=M/'M16-current-Markov-BF16-add-ID-ABI-and-v9-source-reuse-check-v32.json';assert not OUT.exists()
p=M/'M16-root-index-head-layout-closure-and-current-Markov-step-census-v32-v2.json';b=p.read_bytes();assert hashlib.sha256(b).hexdigest()=='ed78d1485dc234d8c67317febddfabdf8eb1235c1f83b91984493a2135c102e1';d=json.loads(b)
counts=Counter()
for r in d['Markov_post_native_TASKs']:
 c=r['definition']
 if not c:continue
 shape={key:c.get(key)for key in['opType_text','inputShapes_text','outputShapes_text','inputDataTypes_text','outputDataTypes_text']};counts[(r['context'],json.dumps(shape,sort_keys=True))]+=1
code="from pathlib import Path;import hashlib,json;p=Path('/vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py');print(json.dumps(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())))"
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=20);assert r.returncode==0,r.stderr;source=json.loads(r.stdout);source['prior_v9_sha256']='458fb50e6839f3a353ddc7181ace4fdae02e35093d8ccd1c4327dc24ae5fff31';source['matches_prior_v9']=source['sha256']==source['prior_v9_sha256']
result=dict(created_utc=datetime.now(timezone.utc).isoformat(),proposer_source_reuse=source,typed_step_shapes=[dict(context=k[0],signature=json.loads(k[1]),TASKs=v)for k,v in counts.items()],scope='SavednativeMarkovpostAdd/ArgMax/Cast/Slice/ViewCopytypedshapesandSHAmatchofalreadyreviewedexactv9proposerfile;noothersourcebody.',no_model_import_device_service_work=True,overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
