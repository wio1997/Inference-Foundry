#!/usr/bin/env python3
"""Read a narrow fixed-commit public source subset with Git blob verification."""
import base64,datetime,hashlib,json,pathlib,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1]
spec=json.loads((R/sys.argv[1]).read_text());O=R/spec['output'];assert O.resolve().is_relative_to(R/'runs');O.mkdir(parents=True,exist_ok=False)
files={}
for name in spec['paths']:
    assert (name.startswith(('python/sglang/srt/', 'python/sglang/kernels/', 'sgl-model-gateway/')) or name in ('python/sglang/utils.py', 'python/pyproject.toml', 'AGENTS.md')) and '..' not in name
    resource='repos/sgl-project/sglang/contents/'+name+'?ref='+spec['commit']
    p=subprocess.run(['gh','api',resource],capture_output=True,text=True,timeout=40);assert p.returncode==0,p.stderr
    j=json.loads(p.stdout);data=base64.b64decode(j['content']);blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest();assert blob==j['sha']
    q=R/'sources/installed'/name;preexisted=q.exists()
    if preexisted:assert q.read_bytes()==data,'Fixed commit source drift: '+name
    else:q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(data)
    files[name]={'git_blob_sha1':blob,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'public_API_resource':resource,'local_snapshot_preexisted':preexisted,'provenance':'PUBLIC_FIXED_COMMIT_GIT_BLOB_VERIFIED'}
result={'status':'PASS','commit':spec['commit'],'files':files,'remote_installed_tree_revalidated':False,'native_executed':False,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(O/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'PASS','files':len(files)}))
