#!/usr/bin/env python3
"""Prepare source-only ordered MemCache get batching over frozen v2 SDMA overlay."""
import ast,difflib,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'config/overlays/vllm-sdma-v2';OUT=ROOT/'config/overlays/vllm-sdma-v3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert not OUT.exists(),'Do not overwrite a prepared overlay'
 manifest=json.loads((OLD/'manifest.json').read_text());OUT.mkdir(parents=True)
 for item in manifest['files']:
  p=OLD/item['local'];assert sha(p)==item['patched_sha256'];q=OUT/item['local'];q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
  p=OLD/item['patch'];q=OUT/item['patch'];q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
 for item in manifest['helpers']:
  p=OLD/item['local'];assert sha(p)==item['sha256'];q=OUT/item['local'];q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
 relative='distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py'
 capture=ROOT/'runs/20261017T0300-framework-source/code-30'
 entry=next(i for i in json.loads((capture/'index.json').read_text()) if i['path']=='/vllm-workspace/vllm-ascend/vllm_ascend/'+relative)
 source=capture/entry['local'];assert sha(source)==entry['sha256'];original=source.read_text();modified=original
 old='import torch\n';new='import torch\nfrom glm53_memcache_batch import batched_get_into_layers\n';assert modified.count(old)==1;modified=modified.replace(old,new,1)
 old='        _inject_device_ub_qos(extra_config)\n';new='''        self._glm53_get_batch_blocks = int((extra_config or {}).get("glm53_get_batch_blocks", 32))
        if not 1 <= self._glm53_get_batch_blocks <= 128:
            raise ValueError("glm53_get_batch_blocks must be in 1..128")
        _inject_device_ub_qos(extra_config)
''';assert modified.count(old)==1;modified=modified.replace(old,new,1)
 old='            res = self.store.batch_get_into_layers(key, addr, size, MmcDirect.COPY_G2L.value)'
 new='''            res = batched_get_into_layers(
                self.store, key, addr, size, MmcDirect.COPY_G2L.value, self._glm53_get_batch_blocks
            )''';assert modified.count(old)==1;modified=modified.replace(old,new,1);ast.parse(modified)
 q=OUT/relative;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(modified)
 patch=OUT/'patches'/(relative.replace('/','__')+'.patch');patch.write_text(''.join(difflib.unified_diff(original.splitlines(True),modified.splitlines(True),fromfile=relative,tofile=relative)))
 manifest['files'].append({'container_path':entry['path'],'local':relative,'original_sha256':entry['sha256'],'patched_sha256':sha(q),'patch':str(patch.relative_to(OUT))})
 helper=ROOT/'scripts/glm53_memcache_batch.py';ast.parse(helper.read_text());q=OUT/'python'/helper.name;shutil.copyfile(helper,q);manifest['helpers'].append({'local':str(q.relative_to(OUT)),'sha256':sha(q)})
 manifest['status']='PREPARED_NOT_DEPLOYED';manifest['changes_from_v2']=['Native MemCache get split into ordered32-key batches with fresh native per-batch read leases; all nonzero/shape errors fail without retry/fallback.'];manifest['lease_failure_evidence']='runs/20261017T0770-long-cache-failure-state-v1';manifest['parent_manifest_sha256']=sha(OLD/'manifest.json')
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(OUT,sha(OUT/'manifest.json'))
if __name__=='__main__':main()
