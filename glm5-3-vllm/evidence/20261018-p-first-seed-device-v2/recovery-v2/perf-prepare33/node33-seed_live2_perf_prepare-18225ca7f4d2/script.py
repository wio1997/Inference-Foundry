#!/usr/bin/env python3
"""Freeze prior correctness-selected fixtures or disable only owned D trace."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess


def main():
    node={'B4-33U-Atlas900RCKA3-32':32,'B5-5U-Atlas900RCKA3-33':33}[socket.gethostname()]
    if node==32:
        old=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/tail-c1-live1-20261018/node32')
        root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/seed-live1-20261018/node32');cases=json.loads((old/'adopt-performance-cases-v1/cases.json').read_text())['cases'];quality=json.loads((root/'recovery-v2/quality24/pairs.json').read_text());assert quality['completed']
        for case in cases:
            rows=[r for r in quality['rows'] if r['prompt_sha256']==case['prompt_sha256']];assert len(rows)==3 and {r['mode'] for r in rows}=={'control','adopt','seed'}
            assert all(r['numeric_pass'] and r['token_ids']==case['expected_token_ids'] for r in rows),'Prior fixed fixture is not exact in new instance; no reselection by timing'
        folder=root/'performance-cases-v1';folder.mkdir(exist_ok=False);data=json.dumps({'cases':cases,'selection':'Same two correctness-selected sum fixtures fixed before v1 timing; revalidated all3 exact in new quality matrix, no latency selection.'},indent=2).encode();(folder/'cases.json').write_bytes(data);result={'cases_file':str(folder/'cases.json'),'sha256':hashlib.sha256(data).hexdigest(),'case_ids':[c['case_id'] for c in cases],'all3_exact_before_timing':True}
    else:
        root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/seed-live2-20261018/node33');plan=json.loads((root/'plan.json').read_text());cid=json.loads((root/'launch.json').read_text())['container_id'];actual=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert actual['Id']=='cfa70a8cc3af142a57402d7e51a21677ff2f0a2fcae4915aedac691849c51189' and actual['State']['Running'] and actual['Name']=='/'+plan['container']['name']
        sentinel=root/'adopt-trace/disabled';assert not sentinel.exists();sentinel.touch();result={'D_trace_writes_disabled':True,'CID':cid,'P_export_copy_and_log_overhead_retained_in_measured_prototype':True,'model_kept_loaded':True}
    print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(result)+'\nGLM53_RESULT_JSON_END')


if __name__=='__main__':main()
