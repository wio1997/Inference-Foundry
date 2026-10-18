#!/usr/bin/env python3
"""Freeze correctness-selected sum cases and disable only the owned D host trace."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess


def main():
    host=socket.gethostname();assert host in ('B4-33U-Atlas900RCKA3-32','B5-5U-Atlas900RCKA3-33')
    if host.endswith('-32'):
        base=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/tail-c1-live1-20261018/node32')
        cases=json.loads((base/'adopt-cases-v1/cases.json').read_text())['cases'];pairs=json.loads((base/'client-adopt-quality24-collect-v2/pairs.json').read_text());assert pairs['completed']
        refs={(r['prompt_sha256'],r['mode']):r for r in pairs['rows']};selected=[]
        for n in (256,257):
            for case in cases:
                if case['input_tokens']!=n or not case['case_id'].startswith('sum-three-'):continue
                control,adopt=(refs[(case['prompt_sha256'],m)] for m in ('control','adopt'))
                if not(control['numeric_pass'] and adopt['numeric_pass'] and control['token_ids']==adopt['token_ids']):continue
                selected.append({**case,'expected_token_ids':control['token_ids']});break
        assert len(selected)==2 and {c['case_id'] for c in selected}=={'sum-three-3','sum-three-4'}
        folder=base/'adopt-performance-cases-v1';folder.mkdir(exist_ok=False)
        data=json.dumps({'cases':selected,'selection':'Earliest exact64-matching sum-three fixture for each length, chosen on correctness before timing'},indent=2).encode();(folder/'cases.json').write_bytes(data)
        result={'host_cases_file':str(folder/'cases.json'),'cases':selected,'sha256':hashlib.sha256(data).hexdigest(),'new_device_ops':False}
    else:
        base=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/adopt-live1-20261018/node33');plan=json.loads((base/'plan.json').read_text());launch=json.loads((base/'launch.json').read_text())
        d=json.loads(subprocess.check_output(['docker','inspect',launch['container_id']]))[0]
        assert d['Id']=='5e457720adb117e100a8cdca6e1f5bbee1936a3d46f044b50ae3a62dcbc22723' and d['Name']=='/'+plan['container']['name'] and d['State']['Running']
        sentinel=base/'adopt-trace/disabled';assert not sentinel.exists();sentinel.touch()
        result={'trace_file_writes_disabled':True,'sentinel':str(sentinel),'container_id':d['Id'],'models_stopped':False,'new_device_ops':False}
    print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(result,indent=2)+'\nGLM53_RESULT_JSON_END')


if __name__=='__main__':main()
