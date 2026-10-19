#!/usr/bin/env python3
"""Stop only redundant CPU Torch summaries after all required CANN paths pass."""
import json
from pathlib import Path
import socket
import subprocess


def main():
    node={'B4-33U-Atlas900RCKA3-32':32,'B5-5U-Atlas900RCKA3-33':33}[socket.gethostname()]
    base=Path('/data/wio/glm5-3-vllm-execution/20261017-official-pd')
    proof=json.loads((base/'consumer-audits'/f'node{node}-profile-exported-v1/result.json').read_text())
    assert proof['completed'] and len(proof['traces'])==len(proof['matrices'])==16
    name=f'glm53-official-profile-export-{node}-v1-20261017'
    p=subprocess.run(['docker','inspect',name],capture_output=True,text=True)
    report={'node':node,'reason':'Required CANN trace/matrix data fully validated and independently copied; omit remaining CPU Torch aggregation. No inference model, proxy or holder stopped.','already_removed':p.returncode!=0}
    if p.returncode==0:
        doc=json.loads(p.stdout)[0]
        assert doc['Name']=='/'+name and doc['Image']=='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
        assert {x['PathOnHost'] for x in doc['HostConfig']['Devices']}=={'/dev/davinci_manager'}
        if doc['State']['Running']:
            subprocess.run(['docker','stop','-t','15',doc['Id']],capture_output=True,check=True,timeout=25)
        report['cpu_container_id']=doc['Id']
    out=base/'consumer-audits'/f'node{node}-profile-export-stop-v1';out.mkdir(parents=True,exist_ok=False)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
