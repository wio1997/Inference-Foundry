"""Bounded reads and real Engram layer-1 projection/gate, without resident tables.

The 24 explicit row probes cover each head's table partition. Hash/normalization
is checked separately; this probe does not claim tokenizer or full-model output.
"""
import argparse
import ast
import json
import os
from pathlib import Path

import torch
import torch_npu

from deepinfer import PLATFORM
from deepinfer.hugging_face.config import load_hf_model_config
from deepinfer.npu_ops.checkpoint910c import ByteCheckpoint
from deepinfer.npu_ops.engram910c import gate
from ascend910c_moe_integration import cpu_matrix


def main():
    p=argparse.ArgumentParser();p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.npu.set_device(0);torch.set_num_threads(4)
    if not PLATFORM.is_910c_reference:raise RuntimeError('explicit 910C reference profile required')
    c=load_hf_model_config(args.model_dir/'config.json').gpt
    # Recover only the dependency-free layout class from the unchanged source.
    import dataclasses
    from sympy import nextprime
    source=Path('deepinfer/model_executor/models/v4_modules/engram_metadata.py')
    tree=ast.parse(source.read_text());tree.body=[n for n in tree.body if isinstance(n,ast.ImportFrom) and n.module=='__future__' or isinstance(n,ast.ClassDef) and n.name=='EngramLayout']
    namespace={'dataclass':dataclasses.dataclass,'AttrDict':type(c),'nextprime':nextprime};exec(compile(tree,str(source),'exec'),namespace)
    layout=namespace['EngramLayout'].from_cfg(c.engram);sizes=[s for part in layout.layer_vocab_sizes[1] for s in part]
    offset=0;rows=[]
    for i,size in enumerate(sizes):rows.append(offset+(13*i+7)%size);offset+=size
    indices=torch.tensor([rows],dtype=torch.int64)
    reader=ByteCheckpoint(args.model_dir,max_read_bytes=192*1024**2)
    name='layers.1.engram.embed.weight';embedded=reader.table_rows(name,indices,'npu:0')
    raw,dtype=reader.raw_rows(name,indices);scales,sfdtype=reader.raw_rows(name.removesuffix('.weight')+'.scale',indices)
    if dtype!='F8_E4M3' or sfdtype!='F8_E8M0':raise ValueError('unexpected table formats')
    cpu_embed=(raw.view(torch.float8_e4m3fn).float()*torch.pow(2.,scales.float()-127.).repeat_interleave(32,-1)).bfloat16()
    torch.testing.assert_close(embedded.cpu(),cpu_embed,rtol=0,atol=0)
    weight_name='layers.1.engram.wkv.weight';w=reader.matrix(weight_name,'npu:0')
    kv=(embedded.flatten(1).float()@w.float().T).bfloat16().reshape(1,c.hc_mult+1,c.hidden_size);del w
    cpu_w=cpu_matrix(reader,weight_name);cpu_kv=(cpu_embed.flatten(1).float()@cpu_w.float().T).bfloat16().reshape_as(kv.cpu());del cpu_w
    x=reader.rows('embed.weight',torch.tensor([1])).unsqueeze(1).expand(1,c.hc_mult,c.hidden_size).contiguous()
    wh=reader.tensor('layers.1.engram.q_weight')[0];we=reader.tensor('layers.1.engram.k_weight')[0]
    actual=gate(x.npu(),kv,wh.npu(),we.npu(),eps=c.rms_norm_eps).cpu()
    refsource=Path(os.environ['TILE_KERNELS_REFERENCE_SOURCE'])/'tile_kernels/torch/engram.py';tree=ast.parse(refsource.read_text());tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef)];ref={'torch':torch};exec(compile(tree,str(refsource),'exec'),ref)
    expected=ref['engram_gate_ref'](x,cpu_kv,wh,we,1e-6,c.rms_norm_eps)
    torch.testing.assert_close(actual,expected,rtol=.02,atol=.005)
    result={'status':'pass','layer':1,'rows_read':len(rows),'table_bytes_read':raw.numel()+scales.numel(),'output_shape':list(actual.shape),'relative_l2':float((actual.float()-expected.float()).norm()/expected.float().norm()),'max_abs_error':float((actual.float()-expected.float()).abs().max()),'boundary':'Actual selected table rows, FP8 wkv projection and Engram gate with BF16 activations; no resident 100GB tables, tokenizer integration or full-model inference.'}
    args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
