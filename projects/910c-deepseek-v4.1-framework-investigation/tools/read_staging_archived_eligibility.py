#!/usr/bin/env python3
"""Hash-check two archived reply-worker logs; inspect fields, not live service."""
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, '/Users/wio/work/branch-fusion-audit-20261104/tools')
from price_vocab_selection_fusion import SCHEDULE, ungzip

REMOTE = r'''
import pathlib,json,sys,hashlib,collections
out=[]
for w in json.loads(sys.argv[1]):
 p=pathlib.Path(w['events'][0]);assert p.is_relative_to('/data/tiankuan/wio/')
 raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==w['events'][1]
 rows=[json.loads(x) for x in raw.splitlines() if x]
 schemas=collections.Counter(tuple(sorted(r)) for r in rows)
 out.append(dict(rank=w['rank'],source=str(p),sha256=w['events'][1],records=len(rows),schemas=[dict(keys=k,count=v) for k,v in schemas.items()],first_records=[{k:v for k,v in r.items() if isinstance(v,(int,float,bool,str)) and len(str(v))<180} for r in rows[:8]]))
print(json.dumps(out))
'''

def main():
    schedule = ungzip(SCHEDULE)
    specs = [w for w in schedule['source_spec']['workers'] if w['rank'] in (0, 8)]
    r = subprocess.run(['/Users/wio/.local/bin/baai-ssh', '171', '--',
                        'python3', '-I', '-S', '-c', REMOTE, json.dumps(specs)],
                       capture_output=True, text=True, timeout=50)
    assert r.returncode == 0, r.stderr
    rows = json.loads(r.stdout)
    assert [r['rank'] for r in rows] == [0, 8]
    out = Path('audit/execution-search/staging-archived-eligibility-schema.json')
    out.write_text(r.stdout)
    for row in rows:
        keys = {k for s in row['schemas'] for k in s['keys']}
        print(json.dumps(dict(rank=row['rank'], records=row['records'],
                              discarded_count_recorded='num_discarded_requests' in keys,
                              all_greedy_recorded='all_greedy' in keys)))

if __name__ == '__main__':
    main()
