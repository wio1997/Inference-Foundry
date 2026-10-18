#!/usr/bin/env python3
"""Restore the pinned unmodified official entry point for current live backends."""
import datetime
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time
import urllib.request


def main():
    assert socket.gethostname()=='B4-33U-Atlas900RCKA3-32'
    base=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/seed-live1-20261018/node32');plan=json.loads((base/'plan.json').read_text());cid=json.loads((base/'launch.json').read_text())['container_id'];ins=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
    assert ins['Id']=='c12b4c9b92d44137da9fc07f5155737f9ff83635553c6e8b4208abdb914c54ac' and ins['State']['Running'] and ins['Name']=='/'+plan['container']['name']
    started=datetime.datetime.fromisoformat(ins['State']['StartedAt'][:26]+'+00:00');elapsed=(datetime.datetime.now(datetime.timezone.utc)-started).total_seconds();remaining=int(plan['container']['lifetime_seconds']-elapsed-180);assert 900<remaining<7200
    image=plan['image']['id'];source='/vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py';sha=subprocess.check_output(['docker','exec',cid,'sha256sum',source],text=True,timeout=15).split()[0];assert sha=='85b8982948530758be655d3448ff10fc3979e56622221feace560272a81f1469'
    root=base/'official-proxy-v1';root.mkdir(exist_ok=False);name='glm53-official-proxy-32-seed-live2-20261018';assert subprocess.run(['docker','inspect',name],capture_output=True).returncode!=0
    with socket.socket() as s:s.bind(('172.16.12.32',29880))
    argv=['docker','run','-d','--pull','never','--name',name,'--network','host','--read-only','--tmpfs','/tmp:rw,size=128m','--cpus','4','--memory','4g','--pids-limit','256','--cap-drop','ALL','--security-opt','no-new-privileges','-e','PYTHONDONTWRITEBYTECODE=1','-e','HF_HUB_OFFLINE=1','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','--entrypoint','/usr/bin/timeout',image,'--signal=TERM','--kill-after=15',str(remaining),'python3',source,'--host','172.16.12.32','--port','29880','--prefiller-hosts','172.16.12.32','--prefiller-ports','29481','--decoder-hosts','172.16.12.33','--decoder-ports','29482']
    proxy_cid=subprocess.check_output(argv,text=True,timeout=30).strip();report={'CID':proxy_cid,'argv':argv,'source_sha256':sha,'unmodified_official_proxy':True,'ready':False,'compute_devices':[],'max_lifetime_seconds':remaining,'parent_P_CID':cid,'default':'native control; experimental seed requires direct per-request opt-in and is not activated by this stock proxy'}
    (root/'launch.json').write_text(json.dumps(report,indent=2)+'\n')
    for unused in range(45):
        try:
            with urllib.request.urlopen('http://172.16.12.32:29880/healthcheck',timeout=2) as f:report['health']=json.load(f)
            report['ready']=True;break
        except Exception:time.sleep(1)
    (root/'result.json').write_text(json.dumps(report,indent=2)+'\n');print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END');assert report['ready']


if __name__=='__main__':main()
