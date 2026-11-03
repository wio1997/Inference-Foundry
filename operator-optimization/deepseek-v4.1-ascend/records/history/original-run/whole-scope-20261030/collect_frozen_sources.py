#!/usr/bin/env python3
"""Read explicitly selected frozen source trees from the owned stage container.

No target imports, NPU calls, service requests or remote writes are made.
"""
import base64
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import zipfile

REMOTE = r'''
import hashlib,io,json,pathlib,subprocess,sys,zipfile
roots={'vllm':pathlib.Path('/vllm-workspace/vllm'),'vllm-ascend':pathlib.Path('/vllm-workspace/vllm-ascend')}
expected={'vllm':'ced6857afa0ea7b2e3f0846a62e1394e90f15607','vllm-ascend':'8fde6d4e335fe300e2b949aab0e31659cce0811d'}
selection={
'vllm-ascend':[
 'vllm_ascend/models/deepseek_v41','vllm_ascend/models/common/ops',
 'vllm_ascend/attention/dsa_v41.py','vllm_ascend/attention/dsa_v1.py',
 'vllm_ascend/attention/context_parallel/dsa_v41_cp.py','vllm_ascend/attention/context_parallel/dsa_cp.py',
 'vllm_ascend/spec_decode','vllm_ascend/worker/model_runner_v1.py',
 'vllm_ascend/worker/worker.py','vllm_ascend/compilation/acl_graph.py',
 'vllm_ascend/ops/fused_moe','vllm_ascend/quantization/methods',
 'vllm_ascend/ops/dsa.py','vllm_ascend/ops/rope_dsv4.py','vllm_ascend/ops/triton/spec_decode',
 'vllm_ascend/ascend_config.py','vllm_ascend/utils.py',
 'vllm_ascend/ascend_forward_context.py','vllm_ascend/core/kv_cache_interface.py',
 'vllm_ascend/distributed/parallel_state.py','vllm_ascend/device',
 'csrc/torch_binding.cpp','csrc/mc2/dispatch_ffn_combine'],
'vllm':[
 'vllm/models/deepseek_v41/common','vllm/models/qwen3_dspark',
 'vllm/v1/sample','vllm/v1/spec_decode','vllm/v1/core','vllm/v1/engine',
 'vllm/v1/worker/gpu_model_runner.py','vllm/v1/worker/gpu_input_batch.py',
 'vllm/v1/worker/utils.py','vllm/v1/utils.py',
 'vllm/model_executor/layers/logits_processor.py',
 'vllm/model_executor/layers/vocab_parallel_embedding.py',
 'vllm/model_executor/layers/linear.py','vllm/model_executor/layers/layernorm.py',
 'vllm/model_executor/layers/activation.py','vllm/model_executor/layers/fused_moe',
 'vllm/model_executor/layers/quantization/utils/quant_utils.py',
 'vllm/distributed/parallel_state.py','vllm/distributed/communication_op.py',
 'vllm/entrypoints/openai/completion/serving.py','vllm/entrypoints/openai/completion/protocol.py',
 'vllm/config/speculative.py','vllm/config/scheduler.py','vllm/config/parallel.py']}
manifest={'scope':'explicit source-tree selection; not a complete transitive dependency closure','roots':{},'selection':selection,'missing':[],'files':[]}
buffer=io.BytesIO()
with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
 for name,root in roots.items():
  head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
  status=subprocess.check_output(['git','-C',str(root),'status','--short','--untracked-files=no'],text=True)
  assert head==expected[name] and not status,(name,head,status)
  manifest['roots'][name]={'head':head,'tracked_status':status}
  seen=set()
  for relative in selection[name]:
   p=root/relative
   if not p.exists():manifest['missing'].append(name+'/'+relative);continue
   files=[p] if p.is_file() else sorted(x for x in p.rglob('*') if x.is_file() and x.suffix in ('.py','.cpp','.cc','.h','.hpp'))
   for f in files:
    if f in seen:continue
    seen.add(f)
    assert root in f.resolve().parents and not f.is_symlink(),str(f)
    data=f.read_bytes();dest=name+'/'+str(f.relative_to(root))
    manifest['files'].append({'path':dest,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    archive.writestr(dest,data)
 archive.writestr('SOURCE-MANIFEST.json',json.dumps(manifest,indent=2))
sys.stdout.buffer.write(buffer.getvalue())
'''

def main():
    here=Path(__file__).resolve().parent
    destination=here/'frozen-source-snapshot'
    if destination.exists():raise RuntimeError('Destination exists; do not overwrite a frozen snapshot')
    payload=base64.b64encode(REMOTE.encode()).decode()
    code="import base64;exec(compile(base64.b64decode('"+payload+"'),'<readonly_source_snapshot>','exec'))"
    command=shlex.join(['docker','exec','wio-vllm-v41-repro169-stage-20261028','python3','-c',code])
    archive=here/'frozen-sources.zip'
    with archive.open('xb') as out,(here/'source-collection.stderr').open('w') as err:
        result=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',command],stdout=out,stderr=err,timeout=180,check=False)
    if result.returncode:raise RuntimeError('Source read failed; returncode='+str(result.returncode))
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('SOURCE-MANIFEST.json'))
        for x in manifest['files']:
            assert not Path(x['path']).is_absolute() and '..' not in Path(x['path']).parts
            data=z.read(x['path']);assert len(data)==x['bytes'] and hashlib.sha256(data).hexdigest()==x['sha256']
        destination.mkdir()
        for name in z.namelist():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            p=destination/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(name))
    receipt={'status':'SOURCE_SNAPSHOT_READ_AND_HASH_VERIFIED','file_count':len(manifest['files']),
             'missing_explicit_paths':manifest['missing'],'archive_bytes':archive.stat().st_size,
             'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'no_device_or_service_work':True}
    (here/'source-collection-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
