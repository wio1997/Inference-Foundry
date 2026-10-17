#!/usr/bin/env python3
"""CPU-only compact four-view summaries for already completed post-D observations."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--control',required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--test-run',choices=('post-pool-state-control-v1','exact-key-source-fence-v1'),default='post-pool-state-control-v1');args=parser.parse_args();assert not args.output.exists()
    ssh=['ssh','-S',args.control,'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-p','60011','root@61.241.77.34']
    def read(node,code):
        pod='sglang-node-access-32' if node==32 else 'sglang-tuning-access-33-20261013'
        p=subprocess.run(ssh+['kubectl exec -i -n default '+pod+' -- nsenter -t1 -m -- python3 -'],input=code,text=True,capture_output=True,timeout=40);assert p.returncode==0,p.stderr;return json.loads(p.stdout)
    run_code="import json,pathlib\nb=pathlib.Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline/tests/post-pool-state-control-v1');p=b/'result.json';p=p if p.exists() else b/'progress.json';print(p.read_text())"
    run=read(32,run_code.replace('tests/post-pool-state-control-v1','tests/'+args.test_run))
    def controls(node):
        ids=[w['P_native_id' if node==32 else 'D_local_id'] for w in run['waves']]
        code='''import collections,json,pathlib
root=pathlib.Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline/pd-nodeNODE-vREV/kv-control-audit')
ids=IDS
selected=collections.defaultdict(lambda:collections.defaultdict(lambda:collections.defaultdict(list)))
for path in sorted(root.glob('buffers-*.jsonl')):
 for line in path.read_text().splitlines():
  try:r=json.loads(line)
  except json.JSONDecodeError:continue
  for rid in ids:
   if r['request_id'].startswith(rid+'-'):
    zeros=[i for i,c in enumerate(r['checksums']) if c[0]==0]
    sentinel=[i for i,(c,n) in enumerate(zip(r['checksums'],r['sizes'])) if c[0]==128*n and c[3]==16384*n]
    selected[rid][r['phase']][r['key']].append({'rank':r['tp_rank'],'control_sha256':r['control_sha256'],'zero_entries':zeros,'zero_bytes':sum(r['sizes'][i] for i in zeros),'constant_0x80_entries':sentinel})
print(json.dumps(selected))
'''.replace('NODE',str(node)).replace('REV','15' if node==32 else '14').replace('IDS',repr(ids))
        return read(node,code)
    p,d=controls(32),controls(33);waves=[]
    for w in run['waves']:
        pv=p[w['P_native_id']];dv=d[w['D_local_id']]
        a=pv['native_prefix'];b=dv['native_prefix'];c=pv['pool_readback'];f=dv['pool_readback']
        assert set(a)==set(b)==set(c)==set(f) and len(a)==31
        bad=[]
        for block,key in enumerate(a):
            assert len(a[key])==len(b[key])==16 and len(c[key])==len(f[key])==1
            ah={r['control_sha256'] for r in a[key]};bh={r['control_sha256'] for r in b[key]};ch=c[key][0]['control_sha256'];fh=f[key][0]['control_sha256']
            if len(ah)==len(bh)==1 and ah==bh=={ch}=={fh}:continue
            bad.append({'block':block,'key':key,'P_native':a[key],'real_D':b[key],'P_private_pool':c[key][0],'D_private_pool':f[key][0],
                        'P_private_matches_native':ch in ah,'D_private_matches_native':fh in ah,'P_and_D_private_match':ch==fh,'real_D_matches_P_native':ah==bh})
        waves.append({'wave':w['wave'],'output_matches_P_native':w['pool_matches_P_native'],'all_views_match':not bad,'bad':bad})
    report={'client_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':run['completed'],'waves':waves,'scope':'CPU read of already-produced moments only. No model/NPU action. Later pool views allow time/visibility changes; sentinel residue is an observed unchanged destination, not alone a physical write or hardware root proof. Immutable archives/full-entry analysis still required.'}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'completed':run['completed'],'waves':[{'wave':w['wave'],'output':w['output_matches_P_native'],'all_views_match':w['all_views_match'],'bad':[{'block':r['block'],'P_private_matches_native':r['P_private_matches_native'],'D_private_matches_native':r['D_private_matches_native'],'private_match':r['P_and_D_private_match'],'D0_zeros':r['real_D'][0]['zero_entries'],'P_pool_zeros':r['P_private_pool']['zero_entries'],'D_pool_zeros':r['D_private_pool']['zero_entries'],'P_pool_sentinel':r['P_private_pool']['constant_0x80_entries'],'D_pool_sentinel':r['D_private_pool']['constant_0x80_entries']} for r in w['bad']]} for w in waves]},indent=2))


if __name__=='__main__':main()
