"""Local source/config preparation; no resource deployment or compilation."""
import ast,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'config/overlays/vllm-sdma-v3';OUT=ROOT/'config/overlays/vllm-sdma-v4';CONFIG=ROOT/'config/vllm-correctness-v4'
assert not OUT.exists() and not CONFIG.exists();shutil.copytree(OLD,OUT);shutil.copytree(ROOT/'config/vllm-correctness-v3',CONFIG)
manifest=json.loads((OUT/'manifest.json').read_text());manifest['parent_manifest_sha256']=hashlib.sha256((OLD/'manifest.json').read_bytes()).hexdigest()
helper='''"""Append project audit events outside rotating Docker/SDK logs."""
import json,os,pathlib,threading,time
_lock=threading.RLock()
def persist(kind,fields):
    row={'kind':kind,'pid':os.getpid(),'thread_id':threading.get_ident(),'audit_monotonic_ns':time.monotonic_ns(),**fields}
    data=(json.dumps(row,sort_keys=True)+'\\n').encode()
    root=pathlib.Path('/run/glm53/audit');root.mkdir(exist_ok=True)
    with _lock:
        fd=os.open(root/f'events-{os.getpid()}.jsonl',os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
        try:
            if os.write(fd,data)!=len(data):raise RuntimeError('Incomplete GLM53 audit write')
        finally:os.close(fd)
'''
(OUT/'python/glm53_persistent_audit.py').write_text(helper);ast.parse(helper)
for name in ['glm53_sdma_transport.py','glm53_memcache_batch.py']:
 p=OUT/'python'/name;s=p.read_text();s='from glm53_persistent_audit import persist\n'+s if not s.startswith('"""') else s
 # Keep future import before helper import.
 if name=='glm53_sdma_transport.py':
  s=s.replace('from __future__ import annotations','from __future__ import annotations\nfrom glm53_persistent_audit import persist')
  s=s.replace("    log.info('GLM53_PD_SDMA=%s', json.dumps({", "    persist('PD_SDMA', {'event':event, **fields})\n    log.info('GLM53_PD_SDMA=%s', json.dumps({")
 else:
  s=s.replace('import hashlib','from glm53_persistent_audit import persist\nimport hashlib')
  s=s.replace("        logger.info('GLM53_MC_GET='+json.dumps(event,sort_keys=True))", "        persist('MC_GET', event)\n        logger.info('GLM53_MC_GET='+json.dumps(event,sort_keys=True))")
 ast.parse(s);p.write_text(s)
for item in manifest['helpers']:item['sha256']=hashlib.sha256((OUT/item['local']).read_bytes()).hexdigest()
manifest['helpers'].append({'local':'python/glm53_persistent_audit.py','sha256':hashlib.sha256((OUT/'python/glm53_persistent_audit.py').read_bytes()).hexdigest()})
manifest['changes_from_v3']=['Persistent perPID SDMA/MC audit, native transfers/batch32 unchanged. SDK log level controlled by versioned config.'];manifest['status']='PREPARED_NOT_DEPLOYED';(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for node in (30,31,43,45):
 p=CONFIG/f'node-{node}.json';cfg=json.loads(p.read_text());args=cfg['launch_args'];cfg['phase']='TARGET_DECODE_GRAPH_QUALIFICATION_NOT_PERFORMANCE_BASELINE';cfg['source_overlay']='config/overlays/vllm-sdma-v4'
 for connector in cfg['kv_transfer']['kv_connector_extra_config']['connectors']:
  if connector['kv_connector']=='MooncakePullConnector':connector['kv_connector_extra_config']['glm53_sdma']['sdk_log_level']=1
 args[args.index('--kv-transfer-config')+1]=json.dumps(cfg['kv_transfer'])
 if node in (43,45):
  args.remove('--enforce-eager');args+=['--compilation-config',json.dumps({'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':[6,12,18,24],'max_cudagraph_capture_size':24})]
  cfg['target_graph_intent']={'mode':'FULL_DECODE_ONLY','capture_tokens':[6,12,18,24],'max_sequences':4,'MTP_verify_width':6,'native_graph_execution_pending':True}
  spec={'method':'mtp','num_speculative_tokens':5,'enforce_eager':True};args[args.index('--speculative-config')+1]=json.dumps(spec)
 else:cfg['target_graph_intent']={'mode':'NONE','reason':'P is prefill role; native current config eager, graph change confined to D target.'}
 p.write_text(json.dumps(cfg,indent=2)+'\n')
print('manifest',hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest())
