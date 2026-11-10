#!/usr/bin/env python3
"""Upload immutable reviewable blobs; ref mutation is a separate connector lease."""
from pathlib import Path
from hashlib import sha1,sha256
import base64,concurrent.futures,json,subprocess
P=Path('/tmp/embedding-static-publication-payload.json');ROOT=Path('audit/execution-search')
def main():
    d=json.loads(P.read_text());assert len(d['files'])==85
    def upload(f):
        b=f['content'].encode();assert len(b)==f['bytes'] and sha256(b).hexdigest()==f['sha256'] and sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==f['git_blob_sha1']
        body=json.dumps(dict(content=base64.b64encode(b).decode(),encoding='base64')).encode()
        r=subprocess.run(['gh','api','repos/wio1997/Inference-Foundry/git/blobs','-X','POST','--input','-'],input=body,capture_output=True,timeout=90)
        if r.returncode:raise RuntimeError('Blob upload failed for '+f['path']+'; branch not changed')
        sha=json.loads(r.stdout)['sha'];assert sha==f['git_blob_sha1'];return dict(path=f['path'],mode='100644',type='blob',sha=sha)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:entries=list(pool.map(upload,d['files']))
    plan=dict(base_commit=d['base_commit'],base_tree_sha=d['base_tree_sha'],entries=entries,payload_SHA256=sha256(P.read_bytes()).hexdigest(),qualification='All immutable blobs uploaded and exactSHA checked; no ref/commit mutation by this helper')
    dest=ROOT/'embedding-static-staged-blobs.json';dest.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(files=len(entries),base=d['base_commit'],all_blob_SHAs_match=True,branch_mutation=False)))
if __name__=='__main__':main()
