#!/usr/bin/env python3
"""Finish Meta-LAST after independently audited stopped providers; no stop replay."""
import ast,hashlib,importlib.util,json
from pathlib import Path
P=Path(__file__).resolve();dep=P.parent/'stop_sglang_empty_pool_v1.py'
assert hashlib.sha256(dep.read_bytes()).hexdigest()=='cab27807a7ee441ec7c5ae2c3462d7b268fb8b4a2e85312eef3e57147f04b4ed'
s=importlib.util.spec_from_file_location('stop_methods',dep);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
RUN=m.m.b.ROOT/'runs/20261017T1708-finish-sglang-meta-stop-v1';REMOTE='/data/wio/inference-foundry/glm53/runs/20261017T1708-finish-sglang-meta-stop-v1'
def main():
    audit=m.m.b.ROOT/'runs/20261017T1707-sglang-pool-shutdown-audit-v1'
    for n in [34,35,36,41]:
        outer=json.loads((audit/f'node-{n}.json').read_text());assert outer['rc']==0;d=json.loads(outer['stdout'])
        assert not d['state']['Running'] and d['state']['ExitCode']==137 and d['npu']['stdout'].count('No running processes')==8
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
    # Change only this imported helper's output-path argument, never historical files.
    m.REMOTE=REMOTE
    code="import pathlib;p=pathlib.Path("+repr(REMOTE)+");assert not p.exists();p.mkdir(parents=True);print(str(p))"
    r=m.m.invoke(34,code);(RUN/'prepare.json').write_text(json.dumps(r,indent=2)+'\n');assert r['rc']==0
    r=m.meta_stop();(RUN/'meta-stop.json').write_text(json.dumps(r,indent=2)+'\n')
    (RUN/'receipt.json').write_text(json.dumps({'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'meta_stop_rc':r['rc'],'prior_providers_stopped':True,'prior_provider_exit_codes':[137]*4,'SDK_application_close_reached':False,'each_node_empty_NPU_cards':8,'model_requests':0},indent=2)+'\n')
    assert r['rc']==0
if __name__=='__main__':main()
