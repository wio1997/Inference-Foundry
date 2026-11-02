# SPDX-License-Identifier: Apache-2.0
"""CPU cache preparation: reused reference functions, actual config/small payloads.

No projections, sparse attention, model forward, allocation of full-model caches
or NPU access. Raw compression inputs are synthetic, actual APE is promoted from
checkpoint BF16 to the reference runtime FP32 parameter boundary.
"""
import argparse
import __future__
import ast
from dataclasses import asdict,replace
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from types import SimpleNamespace

import numpy as np
import torch
from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
from deepinfer.hugging_face.modelslim910c import ModelSlimCheckpoint
from deepinfer.model_executor.models.glm5_next.cache import (
    DSACacheLayout,compressed_slot_mapping,expand_pool_indices,
    state_from_aligned_prefix,append_dsa,dsa_checkpoint_manifest,
)

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--reference-dir',type=Path,required=True)
p.add_argument('--kernel-reference',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
assert not list(Path('/dev').glob('davinci[0-9]*'))
assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0'
torch.set_num_threads(2);torch.manual_seed(5317)
reader=ModelSlimCheckpoint(a.checkpoint,max_read_bytes=64*1024)
config=Glm5NextConfig.from_checkpoint(a.checkpoint);layout=DSACacheLayout.from_config(config,logical_block_size=16)
src=Path(__file__).resolve().parents[2]
report={'status':'in_progress','native_round':17,'NPU_execution':False,'attention_executed':False,
        'architecture':platform.machine(),'torch':torch.__version__,'layout':asdict(layout),'checks':{},'layers':{},
        'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'payload_bytes':0,
        'config_sha256':hashlib.sha256((a.checkpoint/'config.json').read_bytes()).hexdigest(),
        'source_sha256':{n:hashlib.sha256((src/n).read_bytes()).hexdigest() for n in [
            'deepinfer/model_executor/models/glm5_next/cache.py','deepinfer/model_executor/models/glm5_next/request_state.py',
            'tests/ascend/test_glm5_next_cache.py','tests/ascend/probe_glm5_next_cache_cpu.py']},'reference_sha256':{}}


def check(name,x,y,*,atol=0.,rtol=0.):
    torch.testing.assert_close(x,y,atol=atol,rtol=rtol)
    delta=(x.float()-y.float()).abs()
    report['checks'][name]={'passed':True,'shape':list(x.shape),'max_abs':float(delta.max()) if delta.numel() else 0.,'atol':atol,'rtol':rtol}


def pure_function(file,name):
    content=file.read_bytes();report['reference_sha256'][file.name]=hashlib.sha256(content).hexdigest()
    tree=ast.parse(content);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    module=ast.Module(body=[node],type_ignores=[]);namespace={'torch':torch}
    exec(compile(module,str(file),'exec',flags=__future__.annotations.compiler_flag),namespace)
    return namespace[name]


def independent_bytes(name,size):
    path,start,desc=reader.descriptor(name);begin,end=desc['data_offsets'];assert size<=end-begin and size<=reader.max_read_bytes
    with path.open('rb') as f:f.seek(start+begin);raw=f.read(size)
    assert len(raw)==size;return raw


try:
    assert (layout.latent_dim,layout.index_dim,layout.pool_size,layout.topk_tokens)==(512,128,4,2048)
    assert (config.mla_dimensions['qk_nope_head_dim'],config.mla_dimensions['v_head_dim'],config.mla_dimensions['qk_rope_head_dim'])==(256,256,0)
    expected_dsa=list(range(3,44,4));assert [s.layer_id for s in config.layers if s.attention=='deepseek_sparse_attention']==expected_dsa
    for layer_id in expected_dsa:
        manifest=dsa_checkpoint_manifest(reader,config,layer_id,layout)
        assert manifest.qk_scale==.0625 and len(manifest.tensors)==14
        report['layers'][str(layer_id)]=asdict(manifest)
        check(f'layer{layer_id}_manifest',torch.tensor(len(manifest.tensors)),torch.tensor(14))
    for mutate in ({'qk_rope_head_dim':64},):
        bad=replace(config,mla_dimensions={**config.mla_dimensions,**mutate})
        try:DSACacheLayout.from_config(bad,logical_block_size=16)
        except ValueError:pass
        else:raise AssertionError('RoPE contract was silently accepted')
    for bad_id in (0,-1,45):
        try:dsa_checkpoint_manifest(reader,config,bad_id,layout)
        except ValueError:pass
        else:raise AssertionError('non-DSA layer accepted')
    slots_ref=pure_function(a.reference_dir/'kv_cache.py','format_indexer_kpool_slot_mapping')
    tail_ref=pure_function(a.reference_dir/'sparse_attn_indexer_kpool.py','append_causal_tail')
    ring_ref=pure_function(a.reference_dir/'kv_cache.py','get_kpool_tail_ring_capacity')
    for lookahead in (0,1,3):
        cap=ring_ref(SimpleNamespace(speculative_config=SimpleNamespace(num_speculative_tokens=lookahead)),4)
        check(f'ring_capacity_{lookahead}',torch.tensor(replace(layout,lookahead=lookahead).tail_capacity),torch.tensor(cap))
    for block in (4,16,128):
        local=replace(layout,logical_block_size=block);positions=torch.arange(333)
        slots=((positions//block*7+19)%101)*block+positions%block;slots[::11]=-1
        check(f'paged_slots_block{block}',compressed_slot_mapping(slots,positions,local),slots_ref(slots,positions,block,4))
    for topk in (8,2048):
        local=replace(layout,topk_tokens=topk);positions=torch.tensor([0,1,2,3,4,6,7,8,15,16,2046,2047,2048,2050,4098])
        ids=torch.full((len(positions),topk//4),-1,dtype=torch.int32)
        base=torch.full((len(positions),topk+3),-1,dtype=torch.int32)
        for row,pos in enumerate(positions.tolist()):
            complete=(pos+1)//4;chosen=list(reversed(range(complete)))[:topk//4];ids[row,:len(chosen)]=torch.tensor(chosen,dtype=torch.int32)
            tokens=[token for pool in chosen for token in range(pool*4,pool*4+4)]
            base[row,:len(tokens)]=torch.tensor(tokens,dtype=torch.int32)
        tail_ref(base,positions,topk,4)
        actual=expand_pool_indices(ids,positions,local)
        check(f'packed_causal_tail_topk{topk}',actual,base)
        assert all(bool((row[row>=0]<=pos).all()) for row,pos in zip(actual,positions))
    kernel_doc=json.loads(a.kernel_reference.read_text())
    f=next(f for f in kernel_doc['files'] if 'glm5_next_kpool_tail_compress.py' in f['path'])
    assert hashlib.sha256(f['content'].encode()).hexdigest()==f['sha256']
    report['reference_sha256'][f['path']]=f['sha256']
    for layer in (3,43):
        root=f'model.language_model.layers.{layer}.self_attn.indexer.'
        ape,storage=reader.tensor(root+'index_kpool_compress_ape');assert storage=='BF16'
        raw=independent_bytes(root+'index_kpool_compress_ape',4*128*2)
        assert ape.contiguous().view(torch.uint8).numpy().tobytes()==raw
        gate,gate_storage=reader.row_slice(root+'index_kpool_compress_gate',0,2);assert gate_storage=='BF16'
        gate_raw=independent_bytes(root+'index_kpool_compress_gate',2*4096*2)
        assert gate.contiguous().view(torch.uint8).numpy().tobytes()==gate_raw
        report['payload_bytes']+=len(raw)+len(gate_raw)
        check(f'layer{layer}_gate_shape',torch.tensor(gate.shape),torch.tensor([2,4096]))
        ape=ape.float();n=23;latent=torch.randn(n,512).bfloat16();keys=torch.randn(n,128).bfloat16();gates=torch.randn(n,128)*7
        # Use actual two output channels of the compression gate, without
        # implying that the other126 channels or key projection were executed.
        hidden=torch.randn(n,4096).bfloat16();gates[:,:2]=hidden.float()@gate.float().T
        initial=state_from_aligned_prefix(latent[:0],torch.empty(0,128,dtype=torch.bfloat16),layout)
        full=append_dsa(latent,keys,gates,ape,initial,layout,start_position=0)
        expected=[]
        for start in range(0,20,4):
            scores=gates[start:start+4].double().numpy()+ape.double().numpy();weights=np.exp(scores-scores.max(0));weights/=weights.sum(0)
            expected.append((weights*keys[start:start+4].float().numpy()).sum(0))
        check(f'layer{layer}_compression_numpy',full.compressed,torch.tensor(np.asarray(expected)).bfloat16())
        state=initial;start=0
        for end in (1,3,4,7,12,22,23):
            state=append_dsa(latent[start:end],keys[start:end],gates[start:end],ape,state,layout,start_position=start);start=end
        for name in ('latent','compressed','tail','tail_positions'):check(f'layer{layer}_segmented_{name}',getattr(state,name),getattr(full,name))
        # Selected raw tail values must preserve channel FP32, not BF16 storage.
        for pos in (20,21,22):check(f'layer{layer}_tail_gate_{pos}',full.tail[1,pos%4],gates[pos])
        report['layers'][str(layer)]['small_payloads']={'ape_storage':storage,'ape_runtime':'FP32','ape_sha256':hashlib.sha256(raw).hexdigest(),
            'gate_rows':[0,2],'gate_storage':gate_storage,'gate_slice_sha256':hashlib.sha256(gate_raw).hexdigest(),
            'compression_tokens':n,'key_source':'synthetic BF16','gate_source':'synthetic FP32 with two actual checkpoint projection channels'}
    assert 'torch_npu' not in sys.modules
    report['status']='passed'
except Exception as exc:
    report['status']='failed';report['failure']=repr(exc);raise
finally:
    report['ended_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (a.output_dir/'cpu-cache-result.json').write_text(json.dumps(report,indent=2)+'\n')
