#!/usr/bin/env python3
"""Stage immutable blobs only; connector performs exact ref lease separately."""
from pathlib import Path
from hashlib import sha1,sha256
import base64,concurrent.futures,json,subprocess
ROOT=Path('audit/execution-search')
def main():
    p=Path('/tmp/input-resource-publication-payload.json');d=json.loads(p.read_text());assert 50<=len(d['files'])<=70
    def upload(f):
        b=f['content'].encode();assert len(b)==f['bytes'] and sha256(b).hexdigest()==f['sha256'] and sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==f['git_blob_sha1']
        r=subprocess.run(['gh','api','repos/wio1997/Inference-Foundry/git/blobs','-X','POST','--input','-'],input=json.dumps(dict(content=base64.b64encode(b).decode(),encoding='base64')).encode(),capture_output=True,timeout=90)
        if r.returncode:raise RuntimeError('Blob staging failed; branch not changed: '+f['path'])
        h=json.loads(r.stdout)['sha'];assert h==f['git_blob_sha1'];return dict(path=f['path'],mode='100644',type='blob',sha=h)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:entries=list(pool.map(upload,d['files']))
    out=dict(base_commit=d['base_commit'],base_tree_sha=d['base_tree_sha'],entries=entries,payload_SHA256=sha256(p.read_bytes()).hexdigest(),branch_mutation=False);(ROOT/'input-resource-staged-blobs.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(files=len(entries),all_blob_SHAs_match=True,branch_mutation=False,base=d['base_commit'])))
if __name__=='__main__':main()
