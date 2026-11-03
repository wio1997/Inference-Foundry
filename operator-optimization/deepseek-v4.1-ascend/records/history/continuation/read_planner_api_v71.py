"""Two exact native QLI API files to bind injected platform attrs/output publication."""
from pathlib import Path
import base64,hashlib,json
from finalize_origin_coverage_v9 import N,O,read,write
import mirror_saved_profile_core_v53 as tr
from read_planner_sources_v71 import CONTAINER
def main():
    receipt=N/'planner-api166-v71.json';tr.CONTROL=N/'ssh166-v71-xinyufeng.sock';pub=read(O/'planner-public-api-locators-v71.json')['files'];paths=['/vllm-workspace/vllm-ascend/'+r['repo_path']for r in pub]
    code='''import os,socket,subprocess,tarfile,io,base64,hashlib,json
assert os.geteuid()==0 and socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
rows=[]
for path in %s:
 p=subprocess.run(['docker','cp',%s+':'+path,'-'],capture_output=True,timeout=15)
 assert p.returncode==0,p.stderr.decode(errors='replace')[-600:]
 with tarfile.open(fileobj=io.BytesIO(p.stdout),mode='r:*')as tar:
  fs=[m for m in tar.getmembers()if m.isfile()];assert len(fs)==1 and fs[0].size<100000
  b=tar.extractfile(fs[0]).read()
 rows.append(dict(path=path,sha256=hashlib.sha256(b).hexdigest(),source_b64=base64.b64encode(b).decode()))
print(json.dumps(dict(files=rows,container_started=False,model_Torch_NPU_capture_health_process_work=False)))
'''%(repr(paths),repr(CONTAINER))
    reused=receipt.exists()
    if reused:
        saved=read(receipt);assert saved['returncode']==0;data=json.loads(saved['stdout'])
    else:
        r=tr.run_readonly_python(code,timeout=40);write(receipt,dict(returncode=r.returncode,stdout=r.stdout.decode(errors='replace'),stderr=r.stderr.decode(errors='replace')));assert r.returncode==0
        data=json.loads(r.stdout)
    out=[]
    for row,pr in zip(data['files'],pub):
        b=base64.b64decode(row['source_b64']);assert hashlib.sha256(b).hexdigest()==row['sha256'];public_path=Path(pr['path']);public=public_path.read_bytes();blob=lambda x:hashlib.sha1(b'blob '+str(len(x)).encode()+bytes([0])+x).hexdigest()
        if blob(public)!=pr['blob_sha']and public.endswith(bytes([10]))and blob(public[:-1])==pr['blob_sha']:
            public_path.write_bytes(public[:-1]);public=public[:-1]
        assert blob(public)==pr['blob_sha']
        p=N/'planner-operator-source-v71'/Path(row['path']).relative_to('/');p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():assert p.read_bytes()==b
        else:p.write_bytes(b)
        out.append(dict(path=row['path'],local_path=str(p),sha256=row['sha256'],public_source_bytes_match=b==public,public_blob_SHA=pr['blob_sha'],public_immutable_blob_url=pr['git_url']))
    write(O/'planner-API-source-binding-v71.json',dict(files=out,all2public_sources_match=all(x['public_source_bytes_match']for x in out),immutable_remote_receipt_reused=reused,scope='Originalcontainer2exactQLIAPIfilecopies;platformcorecounts/SoCattrinjectionanddirectmetadataoutABI. No loadedbinaryattestation.',overall_complete=False));print(dict(files=2,all_public_match=all(x['public_source_bytes_match']for x in out),container_started=False,receipt_reused=reused))
if __name__=='__main__':main()
