"""恢复原始数据；输入引用只是无损去重，不改变任何原始测量字节。"""
from pathlib import Path
import argparse,gzip,hashlib,json,os
ROOT=Path(__file__).resolve().parents[1]
def sha(b):return hashlib.sha256(b).hexdigest()
def put(path,b,expected,verify_only):
    assert sha(b)==expected,str(path)
    if verify_only:return
    if path.exists():
        assert sha(path.read_bytes())==expected,'Refusing to overwrite changed file: '+str(path)
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_bytes(b);os.replace(tmp,path)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--with-results',action='store_true')
    p.add_argument('--verify-only',action='store_true')
    a=p.parse_args();manifest=json.loads((ROOT/'evidence/data-archive-manifest.json').read_text());datasets={}
    for d in manifest['datasets']:
        b=gzip.decompress((ROOT/d['archive']).read_bytes());put(ROOT/d['path'],b,d['sha256'],a.verify_only)
        datasets[d['path']]=[json.loads(line)['question'] for line in b.splitlines() if line]
    n=0
    if a.with_results:
        for r in manifest['results']:
            if r['encoding']=='identity':
                assert sha((ROOT/r['path']).read_bytes())==r['sha256'];continue
            rows=[]
            for line in gzip.decompress((ROOT/r['archive']).read_bytes()).splitlines():
                x=json.loads(line);question=datasets[x['dataset']][x['row']]
                rows.append(x['json_prefix']+json.dumps(question,ensure_ascii=False)+x['json_suffix'])
            b=''.join(rows).encode();put(ROOT/r['path'],b,r['sha256'],a.verify_only);n+=len(rows)
    print(json.dumps({'datasets':len(datasets),'raw_request_rows_reconstructed_and_SHA256_verified':n,'verify_only':a.verify_only}))
if __name__=='__main__':main()
