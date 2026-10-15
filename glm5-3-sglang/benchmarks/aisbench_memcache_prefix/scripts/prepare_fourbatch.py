#!/usr/bin/env python3
"""Offline immutable slicing: four concurrency waves, with <=64-prefix cohorts."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--source',default='/work/glm53-prefix-data')
p.add_argument('--root',default='/work/aisbench-glm53-run')
a=p.parse_args();src=Path(a.source);root=Path(a.root)
root.mkdir(exist_ok=True)
assert not(root/'dataset-audit.json').exists(),'Partial dataset needs explicit audit'
sha=lambda b:hashlib.sha256(b).hexdigest()
original=json.loads((src/'dataset-audit.json').read_text())
assert original['formal_requests']==768 and original['per_cc']==256
raw_manifest=(src/'dataset-manifest.jsonl').read_bytes()
assert sha(raw_manifest)==original['manifest_sha256']
records=[json.loads(x)for x in raw_manifest.splitlines()]
counts={16:64,32:128,64:256}
selected=[dict(x,source_row=x['row']) for x in records
          if not isinstance(x['cc'],int) or x['row']<counts[x['cc']]]
assert len(selected)==450 and len({x['prompt_sha256']for x in selected})==450
assert len({x['first128_tokens_sha256']for x in selected})==450
for row in selected:
    b=Path(row['prefix_ids_file']).read_bytes()
    assert len(b)==74400*4 and sha(b)==row['prefix_ids_sha256']
    assert row['chat_tokens']==80000 and row['warm_tokens']==74400
for folder in ['logs','warmups','configs','runs','reports']:(root/folder).mkdir(exist_ok=True)
cohorts=[]
for cc,total in counts.items():
    old=src/f'datasets/cc{cc}/test.jsonl'
    assert sha(old.read_bytes())==original['datasets'][str(cc)]['file_sha256']
    lines=old.read_text().splitlines()
    for start in range(0,total,64):
        cohort_id=f'cc{cc}-g{start//64}';leaf=root/'cohorts'/cohort_id
        assert not leaf.exists(),'Existing cohort requires audit'
        leaf.mkdir(parents=True)
        for folder in ['logs','warmups','configs','runs','reports']:(leaf/folder).mkdir()
        shutil.copytree(root/'scripts',leaf/'scripts',ignore=shutil.ignore_patterns('__pycache__'))
        subset=[dict(x,row=x['source_row']-start,cohort_id=cohort_id)
                for x in selected if x['cc']==cc and start<=x['source_row']<start+64]
        assert len(subset)==64 and {x['row']for x in subset}==set(range(64))
        data=lines[start:start+64]
        for row,line in zip(subset,data):
            assert sha(json.loads(line)['question'].encode())==row['prompt_sha256']
        out=leaf/f'datasets/cc{cc}/test.jsonl';out.parent.mkdir(parents=True)
        out.write_text('\n'.join(data)+'\n')
        manifest=leaf/'dataset-manifest.jsonl'
        manifest.write_text(''.join(json.dumps(x)+'\n'for x in subset))
        audit=dict(formal_requests=64,unique_prompts=64,per_cc=64,cohort_id=cohort_id,
                   cc=cc,source_start_row=start,source_end_row=start+64,
                   manifest_sha256=sha(manifest.read_bytes()),
                   source_manifest_sha256=original['manifest_sha256'],
                   datasets={str(cc):dict(count=64,file_sha256=sha(out.read_bytes()))})
        (leaf/'dataset-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        cohorts.append(dict(cohort_id=cohort_id,cc=cc,source_start_row=start,
                            requests=64,root=str(leaf),manifest_sha256=audit['manifest_sha256']))
for cc in ['smoke600','smoke_correctness']:
    old=src/f'datasets/cc{cc}/test.jsonl'
    assert sha(old.read_bytes())==original['datasets'][cc]['file_sha256']
    out=root/f'datasets/cc{cc}/test.jsonl';out.parent.mkdir(parents=True)
    out.write_bytes(old.read_bytes())
needle=json.loads((src/'needle-manifest.jsonl').read_text())
assert needle['cc']=='needle' and needle['chat_tokens']==80000
b=Path(needle['prefix_ids_file']).read_bytes()
assert len(b)==74400*4 and sha(b)==needle['prefix_ids_sha256']
for rel in ['needle-manifest.jsonl','datasets/ccneedle/test.jsonl']:
    out=root/rel;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes((src/rel).read_bytes())
(root/'dataset-manifest.jsonl').write_text(''.join(json.dumps(x)+'\n'for x in selected))
audit=dict(formal_requests=448,unique_formal_prompts=448,total=450,
           requests_per_concurrency={str(k):v for k,v in counts.items()},
           waves_per_concurrency=4,cohort_size=64,cohorts=cohorts,
           manifest_sha256=sha((root/'dataset-manifest.jsonl').read_bytes()),
           source_manifest_sha256=original['manifest_sha256'],
           prewarm_time_included_in_throughput=False,
           prewarm_tokens_included_in_throughput=False,
           note='Seven independent64-request cohorts: cc16 one,cc32 two,cc64 four. Sum formal durations only; preserve per-cohort AIS steady or N/A; never stitch prewarm gaps into continuous steady.')
(root/'dataset-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit))
