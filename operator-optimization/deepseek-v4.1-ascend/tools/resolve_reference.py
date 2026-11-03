#!/usr/bin/env python3
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('reference');args=p.parse_args()
root=Path(__file__).resolve().parents[1];roots=json.loads((root/'inventory/source-roots.json').read_text())['roots']
query=args.reference.split('#',1)[0];matches=[]
for line in (root/'inventory/manifest.jsonl').open():
 e=json.loads(line);original=str(Path(roots[e['source_root']])/e['source_relative'])
 if query==original or query==e['source_relative'] or query==e['repository_path']:
  matches.append({'file':str(root/e['repository_path']),'encoding':e['storage_encoding'],'source_sha256':e['source_sha256'],'redactions':e['credential_redactions']})
print(json.dumps({'reference':args.reference,'matches':matches,
 'if_absent':'可能是未收录的原始profile/执行日志/连接材料，或其他案例外部引用；查看inventory/excluded-materials.jsonl，不假定不存在该效应。'},ensure_ascii=False,indent=2))
