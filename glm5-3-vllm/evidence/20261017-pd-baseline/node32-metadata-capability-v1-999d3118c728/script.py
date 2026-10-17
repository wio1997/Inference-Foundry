#!/usr/bin/env python3
"""Verify pinned metadata-only client lookup/close without NPU mounts or object mutation."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess


def main():
    assert socket.gethostname()=='B4-33U-Atlas900RCKA3-32'
    base=Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline')
    root=base/'consumer-audits/metadata-capability-v1';root.mkdir(parents=True,exist_ok=False)
    code='''import json
from memcache_hybrid import DistributedObjectStore
store=DistributedObjectStore()
init=store.init(0,init_bm=False)
assert init==0,init
value=store.batch_is_exist(['glm53-metadata-capability-no-object-v1'])
methods={'remove_batch':hasattr(store,'remove_batch'),'remove':hasattr(store,'remove')}
close=store.close()
print('GLM53_RESULT_JSON_BEGIN'+json.dumps({'init':init,'missing_key_lookup':list(value),'methods':methods,'close':close,'npu_mounted':False,'object_mutation':False})+'GLM53_RESULT_JSON_END',flush=True)
assert list(value)==[0] and close==0
'''
    (root/'probe.py').write_text(code)
    image='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
    fix=Path('/data/wio/glm5-3-vllm-execution/20261017-memcache/loader-fix-v3')
    assert hashlib.sha256((fix/'aliases.json').read_bytes()).hexdigest()=='2a10023d36c2480332a24dadd2e24142a68d07a04a9cf358ced1f16adf167d03'
    env=json.loads(subprocess.check_output(['docker','image','inspect',image,'--format','{{json .Config.Env}}']))
    ld=next((x.split('=',1)[1] for x in env if x.startswith('LD_LIBRARY_PATH=')), '')
    py=next((x.split('=',1)[1] for x in env if x.startswith('PYTHONPATH=')), '')
    config='/data/wio/glm5-3-vllm-execution/20261017-memcache/pd-config-node32-fixed-v3'
    argv=['docker','run','--rm','--pull','never','--network','host','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--cpus','2','--memory','2g','--pids-limit','512','-v',str(root)+':/out','-v',str(fix/'site-packages')+':/opt/memcache-runtime:ro','-v',config+':/memcache-config:ro','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','-e','PYTHONPATH=/opt/memcache-runtime'+(':'+py if py else ''),'-e','LD_LIBRARY_PATH=/opt/memcache-runtime/memcache_hybrid/lib:/opt/memcache-runtime/memfabric_hybrid/lib:'+ld,'-e','MMC_LOCAL_CONFIG_PATH=/memcache-config/mmc-local.conf','-e','OMP_NUM_THREADS=1','-e','PYTHONDONTWRITEBYTECODE=1','--entrypoint','/usr/bin/timeout',image,'--signal=TERM','--kill-after=5','40','python3','/out/probe.py']
    p=subprocess.run(argv,capture_output=True,text=True,timeout=50)
    (root/'stdout.txt').write_text(p.stdout);(root/'stderr.txt').write_text(p.stderr)
    result={'returncode':p.returncode,'argv':argv,'scope':'No NPU device mounts, no payload transfer, no object mutation. Metadata-only lookup and explicit close capability for a possible exact-key repair contrast.'}
    if 'GLM53_RESULT_JSON_BEGIN' in p.stdout:
        result['probe']=json.loads(p.stdout.split('GLM53_RESULT_JSON_BEGIN',1)[1].split('GLM53_RESULT_JSON_END',1)[0])
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    assert p.returncode==0 and result['probe']['close']==0


if __name__=='__main__':main()
