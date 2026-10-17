#!/usr/bin/env python3
"""Read completed post-D integer controls only; never submit requests or NPU work."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control',required=True)
    parser.add_argument('--test-run', choices=('source-ready-abba-v1', 'known-failure-key-replay-v1', 'post-pool-state-control-v1'), default='source-ready-abba-v1')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    ssh = ['ssh','-S',args.control,'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-p','60011','root@61.241.77.34']
    def read(node, code):
        pod = 'sglang-node-access-32' if node == 32 else 'sglang-tuning-access-33-20261013'
        p = subprocess.run(ssh+['kubectl exec -i -n default '+pod+' -- nsenter -t1 -m -- python3 -'],input=code,text=True,capture_output=True,timeout=40)
        assert p.returncode == 0,p.stderr
        return json.loads(p.stdout)
    code = '''import json,pathlib
base=pathlib.Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline')
p=base/'tests/source-ready-abba-v1/progress.json'
f=base/'tests/source-ready-abba-v1/result.json'
print((f if f.exists() else p).read_text())
'''
    code = code.replace('tests/source-ready-abba-v1', 'tests/'+args.test_run)
    run = read(32,code)
    def controls(node):
        request_ids = [w['P_native_id' if node == 32 else 'D_local_id'] for w in run['waves']]
        code = '''import collections,json,pathlib
root=pathlib.Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline/pd-nodeNODE-vREV/kv-control-audit')
ids=IDS
selected=collections.defaultdict(lambda:collections.defaultdict(list))
for path in sorted(root.glob('buffers-*.jsonl')):
 raw=path.read_text()
 for line in raw.splitlines():
  try:r=json.loads(line)
  except json.JSONDecodeError:continue
  for request_id in ids:
   if r['phase']=='native_prefix' and r['request_id'].startswith(request_id+'-'):
    zero=[i for i,c in enumerate(r['checksums']) if c[0]==0]
    selected[request_id][r['key']].append({'rank':r['tp_rank'],'control_sha256':r['control_sha256'],'zero_entries':zero,'zero_bytes':sum(r['sizes'][i] for i in zero)})
print(json.dumps(selected))
'''.replace('NODE',str(node)).replace('REV','14' if node == 32 else '13').replace('IDS',repr(request_ids))
        return read(node,code)
    p,d = controls(32),controls(33)
    waves=[]
    for w in run['waves']:
        source=p[w['P_native_id']];target=d[w['D_local_id']]
        assert set(source)==set(target) and len(source)==31
        differences=[];source_divergent=[]
        for key in source:
            a,b=source[key],target[key]
            assert {x['rank'] for x in a}=={x['rank'] for x in b}==set(range(16))
            source_hashes={x['control_sha256'] for x in a}
            if len(source_hashes)!=1:source_divergent.append(key)
            bad=[x for x in b if x['control_sha256'] not in source_hashes]
            if bad:differences.append({'key':key,'target_not_matching_any_source':bad})
        waves.append({'wave':w['wave'],'fenced':w['source_fence_selected'],'output_matches_P_native':w['pool_matches_P_native'],'source_divergent_keys':source_divergent,'content_controls_match':not differences,'differences':differences})
    report={'test_run':args.test_run,'client_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':run['completed'],'failure':run.get('failure'),'waves':waves,'scope':'CPU read of already-generated controls for completed waves. No new request, tensor read, synchronization, patch or NPU work. Final immutable archives and full entry analysis are authoritative; four moments are not cryptographic identity.'}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'waves':[dict(wave=w['wave'],fenced=w['fenced'],output=w['output_matches_P_native'],content=w['content_controls_match'],bad_keys=len(w['differences'])) for w in waves],'completed':report['completed']}))


if __name__ == '__main__':
    main()
