#!/usr/bin/env python3
import argparse,gzip,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');args=p.parse_args()
root=Path(__file__).resolve().parents[1];count=0;bad=[]
for line in (root/'inventory/manifest.jsonl').open():
 e=json.loads(line)
 if not args.full and not e['repository_path'].startswith('records/latest/'):continue
 f=root/e['repository_path'];count+=1
 if not f.is_file():bad.append({'path':e['repository_path'],'reason':'missing'});continue
 h=hashlib.sha256()
 with f.open('rb') as x:
  for block in iter(lambda:x.read(1024*1024),b''):h.update(block)
 if h.hexdigest()!=e['published_sha256']:bad.append({'path':e['repository_path'],'reason':'published_sha_mismatch'})
 if args.full:
  h=hashlib.sha256();op=gzip.open if e['storage_encoding']=='gzip' else open
  with op(f,'rb') as x:
   for block in iter(lambda:x.read(1024*1024),b''):h.update(block)
  if h.hexdigest()!=e['decoded_published_sha256']:bad.append({'path':e['repository_path'],'reason':'decoded_sha_mismatch'})
  if e['credential_redactions']==0 and h.hexdigest()!=e['source_sha256']:bad.append({'path':e['repository_path'],'reason':'source_sha_mismatch'})
print(json.dumps({'status':'PASS' if not bad else 'FAIL','scope':'full' if args.full else 'latest','files_checked':count,'failures':bad,
 'scope_note':'仅校验本归档字节/脱敏来源关系，不重做原实验正确性、性能或融合合法性证明。'},ensure_ascii=False,indent=2))
raise SystemExit(1 if bad else 0)
