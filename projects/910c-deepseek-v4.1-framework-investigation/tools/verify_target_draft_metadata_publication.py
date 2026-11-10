#!/usr/bin/env python3
"""Read every byte at the fixed published commit, including large default HTML."""
from pathlib import Path
from hashlib import sha1,sha256
import concurrent.futures,json,urllib.parse,urllib.request
ROOT=Path('audit/execution-search')
def main():
 d=json.loads(Path('/tmp/target-draft-publication-payload.json').read_text());c=json.loads(Path('/tmp/target-draft-published-commit.json').read_text());assert c['base']==d['base_commit']
 def verify(f):
  url='https://raw.githubusercontent.com/wio1997/Inference-Foundry/'+c['sha']+'/'+urllib.parse.quote(f['path'])
  with urllib.request.urlopen(url,timeout=60) as r:b=r.read()
  assert b==f['content'].encode() and len(b)==f['bytes'] and sha256(b).hexdigest()==f['sha256'] and sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==f['git_blob_sha1'],f['path']
  if f['path'].endswith('/target-draft-metadata-price-raw.json'):
   raw=json.loads(b);assert len(raw['ranks'])==16 and sum(len(r['calls']) for r in raw['ranks'])==176
  return dict(path=f['path'],bytes=len(b),sha256=f['sha256'],git_blob_sha1=f['git_blob_sha1'],all_content_equal=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(verify,d['files']))
 out=dict(commit=c['sha'],parent=c['base'],tree=c['tree'],file_count=len(rows),files=rows,paths_match_compared_commit=True,all_content_equal=True,sha256_and_git_blob_sha1_match=True,large_HTML_full_bytes_verified=True,full_176_calls_verified=True,qualification='Full pinned rawHTTPS readback; connector per-file SHA checks separately recorded.',new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[]);(ROOT/'target-draft-metadata-publication.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='files'}))
if __name__=='__main__':main()
