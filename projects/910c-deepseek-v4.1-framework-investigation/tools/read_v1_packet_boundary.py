#!/usr/bin/env python3
"""Pin metadata serializers, packet structs, adapters and output initialization."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_swa_query_producer_entry import REMOTE
ROOT=Path('audit/execution-search')
def main():
    prefix='/vllm-workspace/vllm-ascend/csrc/attention/'
    pairs=[
        ('sparse_flash_mla_metadata/op_kernel_aicpu/sparse_flash_mla_metadata_aicpu.cpp','smla-runtime-source/metadata-repo/sparse_flash_mla_metadata_aicpu.cpp'),
        ('sparse_flash_mla_metadata/op_kernel_aicpu/sparse_flash_mla_metadata_aicpu.h','smla-runtime-source/metadata-repo/sparse_flash_mla_metadata_aicpu.h'),
        ('sparse_flash_mla/op_kernel/sparse_flash_mla_kernel_metadata.h','smla-runtime-source/repo/op_kernel/sparse_flash_mla_kernel_metadata.h'),
        ('sparse_flash_mla/sparse_flash_mla_torch_adpt.h','smla-runtime-source/repo/sparse_flash_mla_torch_adpt.h'),
        ('sparse_flash_mla_metadata/op_host/op_api/aclnn_sparse_flash_mla_metadata.cpp',None),
        ('sparse_flash_mla_metadata/op_host/op_api/sparse_flash_mla_metadata.cpp',None)]
    for n in ['quant_lightning_indexer_v2/op_kernel/quant_lightning_indexer_v2_metadata.h','quant_lightning_indexer_v2/quant_lightning_indexer_v2_torch_adpt.h','quant_lightning_indexer_v2_metadata/op_kernel_aicpu/quant_lightning_indexer_v2_metadata_aicpu.cpp','quant_lightning_indexer_v2_metadata/op_api/aclnn_quant_lightning_indexer_v2_metadata.cpp','quant_lightning_indexer_v2_metadata/op_api/quant_lightning_indexer_v2_metadata.cpp']:
        pairs.append((n,'indexer-native-runtime-source/vllm-ascend/csrc/attention/'+n))
    spec=[]
    for n,local in pairs:
        p=ROOT/local if local else None
        spec.append(dict(runtime_path=prefix+n,local_path=str(p) if p and p.exists() else None,local_sha256=sha256(p.read_bytes()).hexdigest() if p and p.exists() else None))
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,json.dumps(spec)],capture_output=True,text=True,timeout=115)
    if r.returncode:raise RuntimeError(r.stderr)
    d=json.loads(r.stdout)
    for s in d['sources']:
        body=s.pop('content',None)
        if body is not None:
            p=ROOT/'v1-packet-runtime-source'/s['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body);assert sha256(p.read_bytes()).hexdigest()==s['runtime_sha256'];s['local_path']=str(p)
    d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),whole_chain_complete=False,qualification='Current source bytes only. Inspect serializer/output allocation before inferring pointer independence or entire native packet bytes.')
    (ROOT/'v1-packet-dependencies.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=len(d['sources']),existing_matches=sum(s['matches_local'] for s in d['sources']),new_bodies=sum(s['local_sha256'] is None for s in d['sources']),new_NPU_work=0,remote_writes=[])))
if __name__=='__main__':main()
