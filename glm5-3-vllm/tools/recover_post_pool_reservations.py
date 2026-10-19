#!/usr/bin/env python3
"""Recover the recorded32-deleted/33-owned state without touching another claim."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--control',required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args()
    assert not args.receipt.exists()
    ssh=['ssh','-S',args.control,'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-p','60011','root@61.241.77.34']
    expected={32:'8058e02a-abe3-4c77-93ca-539946844948',33:'8e163546-e858-4c61-955a-cba01b6229c4'}
    pods={32:'sglang-node-access-32',33:'sglang-tuning-access-33-20261013'}
    def command(text):return subprocess.check_output(ssh+[text],timeout=40)
    def get(node):
        raw=command(f'kubectl get pod -n default glm53-execution-reservation-{node}-20261017 --ignore-not-found -o json')
        return json.loads(raw) if raw.strip() else None
    def no_other_claims(allow_old33):
        for node in [32,33]:
            doc=json.loads(command(f'kubectl get pods -A --field-selector spec.nodeName=a3-5u-atlas900rcka3-{node} -o json'))
            for pod in doc['items']:
                n=sum(int(c.get('resources',{}).get('requests',{}).get('huawei.com/Ascend910',0)) for c in pod['spec'].get('containers',[]))
                if n and pod['status']['phase'] not in ('Failed','Succeeded'):
                    assert allow_old33 and node==33 and pod['metadata']['uid']==expected[33], 'Another live NPU claim; do not preempt'
    report={'client_started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'recovery_scope':'32 old holder already absent,33 exact old owner remains. No model/NPU processes on either. Only old33 deleted normally, then finite3600s holders created if no other claim.'}
    try:
        assert get(32) is None
        old=get(33);assert old['metadata']['uid']==expected[33]
        assert old['metadata']['labels']['owner']=='codex-execution-20261017' and not old['metadata'].get('ownerReferences')
        assert old['spec']['containers'][0]['command']==['/bin/sleep','3600']
        no_other_claims(True)
        for node in [32,33]:
            raw=command(f'kubectl exec -i -n default {pods[node]} -- nsenter -t1 -m -- npu-smi info').decode()
            assert raw.count('No running processes found in NPU')==8
        args.receipt.with_suffix('.recovery-intent.json').write_text(json.dumps(report,indent=2)+'\n')
        command('kubectl delete pod -n default glm53-execution-reservation-33-20261017 --wait=false')
        deadline=time.monotonic()+70
        while True:
            old=get(33)
            if old is None:break
            assert old['metadata']['uid']==expected[33]
            assert time.monotonic()<deadline,'Own holder deletion did not finish normally'
            time.sleep(3)
        no_other_claims(False)
        documents=[]
        for node in [32,33]:
            doc=json.loads((Path(__file__).parent/f'node{node}-reservation.json').read_text())
            doc['spec']['activeDeadlineSeconds']=3600;doc['spec']['containers'][0]['command']=['/bin/sleep','3600'];documents.append(doc)
        subprocess.run(ssh+['kubectl create -f -'],input=json.dumps({'apiVersion':'v1','kind':'List','items':documents}).encode(),check=True,capture_output=True,timeout=40)
        deadline=time.monotonic()+120
        while True:
            after=[]
            for node in [32,33]:
                pod=get(node)
                after.append({'node':node,'name':pod['metadata']['name'],'uid':pod['metadata']['uid'],'phase':pod['status']['phase'],'creation_timestamp':pod['metadata']['creationTimestamp'],'start_time':pod['status'].get('startTime'),'ascend_real':pod['metadata'].get('annotations',{}).get('huawei.com/AscendReal'),'active_deadline_seconds':pod['spec']['activeDeadlineSeconds']})
            report['after']=after
            if all(r['phase']=='Running' and r['ascend_real'] for r in after):break
            assert time.monotonic()<deadline,'Finite reservation readiness exhausted; do not preempt'
            time.sleep(3)
        report['completed']=True
    except Exception as exc:
        report['failure']=type(exc).__name__+': '+str(exc);report['completed']=False
    args.receipt.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
    assert report['completed'],report.get('failure')


if __name__=='__main__':main()
