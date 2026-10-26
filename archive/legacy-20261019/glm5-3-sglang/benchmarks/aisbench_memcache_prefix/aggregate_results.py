#!/usr/bin/env python3
"""Re-audit 448 formal requests and aggregate formal time only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',required=True)
a=p.parse_args();root=Path(a.root)
data=json.loads((root/'dataset-audit.json').read_text())
assert data['formal_requests']==448
seen=set();groups=[];totals={cc:dict(concurrency=cc,requests=0,output_tokens=0,formal_duration_s=0.,prewarm_s=0.)for cc in [16,32,64]}
for group in data['cohorts']:
    leaf=Path(group['root']);cc=group['cc']
    report=json.loads((leaf/f'reports/cc{cc}/result.json').read_text())
    assert report['valid'] and report['requests']==64
    expected={x['row']:x for x in map(json.loads,(leaf/'dataset-manifest.jsonl').read_text().splitlines())}
    raw=Path(report['request_audit']['raw_file'])
    assert hashlib.sha256(raw.read_bytes()).hexdigest()==report['request_audit']['raw_sha256']
    rows=[json.loads(x)for x in raw.read_text().splitlines()]
    assert len(rows)==64 and {x['id']for x in rows}==set(range(64))
    for row in rows:
        assert row['success'] and (row['input_tokens'],row['output_tokens'])==(80000,600)
        value=row['input']
        text=value if isinstance(value,str)else value[0].get('content',value[0].get('prompt'))
        digest=hashlib.sha256(text.encode()).hexdigest()
        assert digest==expected[row['id']]['prompt_sha256'] and digest not in seen
        seen.add(digest)
        extra=row['extra_details_data'];usage=extra['native_usage'];cache=extra['cached_tokens_details']
        assert (usage['prompt_tokens'],usage['completion_tokens'])==(80000,600)
        assert usage['prompt_tokens_details']['cached_tokens']==74368
        assert cache['storage']==74368 and cache['device']==0 and cache['host']==0
    times=json.loads((leaf/f'reports/cc{cc}/request-timing.json').read_text())
    assert len(times)==64 and {x['id']for x in times}==set(range(64))
    seconds=max(x['end_time']for x in times)-min(x['start_time']for x in times)
    assert seconds>0
    warm=json.loads((leaf/f'warmups/cc{cc}.complete.json').read_text())
    total=totals[cc];total['requests']+=64;total['output_tokens']+=38400
    total['formal_duration_s']+=seconds;total['prewarm_s']+=warm['elapsed_s']
    groups.append(dict(cohort=group['cohort_id'],formal_duration_s=seconds,
                       native_ais_steady=report['steady'],report=str(leaf/f'reports/cc{cc}/result.json')))
assert len(seen)==448
for cc,n in [(16,64),(32,128),(64,256)]:
    total=totals[cc];assert total['requests']==n
    total['output_tokens_per_second']=total['output_tokens']/total['formal_duration_s']
    total['requests_per_second']=n/total['formal_duration_s']
result=dict(valid=True,independently_audited_requests=448,unique_prompts=448,
            prewarm_time_included=False,prewarm_tokens_included=False,
            aggregates=list(totals.values()),groups=groups,
            steady_definition='Native AIS steady is per group; gaps are never stitched into one continuous steady window.',
            note='Aggregate throughput=sum formal output tokens / sum formal group durations. See native per-group CSV/JSON for TTFT and TPOT.')
(root/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
with(root/'SUMMARY.csv').open('w',newline='')as stream:
    writer=csv.DictWriter(stream,fieldnames=list(totals[16]));writer.writeheader();writer.writerows(totals.values())
print(json.dumps(dict(valid=True,requests=448,summary=str(root/'SUMMARY.csv'))))
