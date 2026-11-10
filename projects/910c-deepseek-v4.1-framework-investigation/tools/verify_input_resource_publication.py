#!/usr/bin/env python3
"""Verify all published bytes and decode the full original native record blob."""
from pathlib import Path
from hashlib import sha1,sha256
import base64,concurrent.futures,json,urllib.parse,urllib.request,zlib
ROOT=Path('audit/execution-search')
def main():
    expected=json.loads(Path('/tmp/input-resource-publication-payload.json').read_text());c=json.loads(Path('/tmp/input-resource-published-commit.json').read_text());assert c['base']==expected['base_commit'];native=json.loads((ROOT/'input-resource-publication-manifest.json').read_text())['full_native_records']
    def verify(f):
        url='https://raw.githubusercontent.com/wio1997/Inference-Foundry/'+c['sha']+'/'+urllib.parse.quote(f['path'])
        with urllib.request.urlopen(url,timeout=50) as r:b=r.read()
        assert b==f['content'].encode() and len(b)==f['bytes'] and sha256(b).hexdigest()==f['sha256'] and sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==f['git_blob_sha1'],f['path']
        if f['path']==native['path']:
            compressed=base64.b64decode(b);decoded=zlib.decompress(compressed);assert sha256(compressed).hexdigest()==native['compressed_SHA256'] and sha256(decoded).hexdigest()==native['decompressed_SHA256'];d=json.loads(decoded);assert len(d)==16 and sum(len(x['windows']) for x in d)==3696
        return dict(path=f['path'],bytes=len(b),sha256=f['sha256'],git_blob_sha1=f['git_blob_sha1'],all_content_equal=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(verify,expected['files']))
    out=dict(commit=c['sha'],parent=c['base'],tree=c['tree'],file_count=len(rows),files=rows,paths_match_compared_commit=True,all_content_equal=True,sha256_and_git_blob_sha1_match=True,large_HTML_full_bytes_verified=True,full_native_compressed_and_decompressed_bytes_verified=True,qualification='Full pinned rawHTTPS readback; connector per-file SHA verification recorded separately after completion.',new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[]);(ROOT/'input-resource-publication.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='files'}))
if __name__=='__main__':main()
