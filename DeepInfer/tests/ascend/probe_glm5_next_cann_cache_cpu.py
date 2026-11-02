# SPDX-License-Identifier: Apache-2.0
"""CPU-only cache ABI preparation with extracted installed Python references.

No native metadata/Triton/CANN operator, model projection, device allocation or
serving allocator is executed. Dense main/index pages only; padded tails are
supported explicitly. Reuses previous configuration/source evidence, no weights.
"""
import argparse
import ast
import __future__
from dataclasses import asdict
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from types import SimpleNamespace

import torch
from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
from deepinfer.model_executor.models.glm5_next.cache import DSACacheLayout,compressed_slot_mapping
from deepinfer.model_executor.models.glm5_next.cann_cache import (
    CANNCacheGeometry,view_main_index,view_tail,pack_tail_planes,unpack_tail_rows,
    expand_kernel_block_table,plan_tail,prepare_tail_state,
)

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--cache-views-reference',type=Path,required=True)
p.add_argument('--indexer-reference',type=Path,required=True);p.add_argument('--cann-reference',type=Path,required=True)
p.add_argument('--core-reference',type=Path,required=True);p.add_argument('--kernel-reference',type=Path,required=True)
p.add_argument('--prior-evidence',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0' and not list(Path('/dev').glob('davinci[0-9]*'))
torch.set_num_threads(2);torch.manual_seed(5319)
config=Glm5NextConfig.from_checkpoint(a.checkpoint);previous=json.loads(a.prior_evidence.read_text())
assert previous['status']=='passed' and previous['NPU_execution'] is False
assert hashlib.sha256((a.checkpoint/'config.json').read_bytes()).hexdigest()==previous['config_sha256']
src=Path(__file__).resolve().parents[2]
assert hashlib.sha256((src/'deepinfer/model_executor/models/glm5_next/projections.py').read_bytes()).hexdigest()==previous['source_sha256']['deepinfer/model_executor/models/glm5_next/projections.py']
report={'status':'in_progress','native_round':19,'NPU_execution':False,'CANN_Triton_executed':False,'weight_payload_bytes_read':0,
        'architecture':platform.machine(),'torch':torch.__version__,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'config_sha256':previous['config_sha256'],'reused_evidence_sha256':hashlib.sha256(a.prior_evidence.read_bytes()).hexdigest(),
        'checks':{},'cases':[],'reference_sha256':{},'source_sha256':{n:hashlib.sha256((src/n).read_bytes()).hexdigest() for n in [
            'deepinfer/model_executor/models/glm5_next/cann_cache.py','tests/ascend/test_glm5_next_cann_cache.py','tests/ascend/probe_glm5_next_cann_cache_cpu.py']}}


class BaseSpec:
    pass
class Spec(BaseSpec):
    pass
class Uniform:
    pass
class Backend:
    @staticmethod
    def get_kv_cache_shape(n,rows,heads,dim):return (n,rows,heads,dim)

namespace={'torch':torch,'get_dtype_size':lambda dtype:torch.empty(0,dtype=dtype).element_size(),
           'AscendMLAAttentionSpec':Spec,'MLAAttentionSpec':BaseSpec,'UniformTypeKVCacheSpecs':Uniform}


def functions(path,names,*,class_name=None):
    raw=path.read_bytes();report['reference_sha256'][path.name]=hashlib.sha256(raw).hexdigest();tree=ast.parse(raw)
    nodes=tree.body
    if class_name:nodes=next(n for n in nodes if isinstance(n,ast.ClassDef) and n.name==class_name).body
    chosen=[n for n in nodes if isinstance(n,ast.FunctionDef) and n.name in names];assert {n.name for n in chosen}==set(names)
    exec(compile(ast.Module(body=chosen,type_ignores=[]),str(path),'exec',flags=__future__.annotations.compiler_flag),namespace)


def check(name,x,y):
    torch.testing.assert_close(x,y,rtol=0,atol=0)
    report['checks'][name]={'passed':True,'shape':list(x.shape),'dtype':str(x.dtype),'exact':True}


def spec(block,head,ratio,page_bytes,dtype=torch.bfloat16):
    s=Spec();s.block_size=block;s.head_size=head;s.tokens_per_state=ratio;s.page_size_bytes=page_bytes;s.dtype=dtype;s.num_kv_heads=1
    s.unpadded_page_size_bytes=block//ratio*head*torch.empty(0,dtype=dtype).element_size()
    return s


try:
    functions(a.core_reference,['get_kv_cache_compression_ratio','get_storage_block_size'])
    functions(a.cache_views_reference,['_row_major_strides','view_kpool_tail_cache','view_compressed_indexer_cache','_view_nope_main_mla_cache'])
    functions(a.cann_reference,['_cache_rows','prepare_keypool_state'])
    functions(a.indexer_reference,['_get_state_table'],class_name='AscendIndexerKPoolMetadataBuilder')
    kernels=json.loads(a.kernel_reference.read_text());f=next(f for f in kernels['files'] if f['path'].endswith('glm5_next_cann_keypool.py'))
    assert hashlib.sha256(f['content'].encode()).hexdigest()==f['sha256'];report['reference_sha256'][f['path']]=f['sha256']
    for logical in (128,256,512):
        for lookahead in (0,3):
            layout=DSACacheLayout.from_config(config,logical_block_size=logical,lookahead=lookahead)
            # Canonical small page equals unpadded index page for these cases;
            # tail capacity4/7 is smaller than8/16/32 physical page rows.
            small_bytes=layout.compressed_block_size*layout.index_dim*2
            g=CANNCacheGeometry(layout,small_bytes);pages=8;tag=f'b{logical}_look{lookahead}'
            main_raw=torch.zeros(pages*g.main_page_bytes,dtype=torch.uint8);index_raw=torch.zeros(pages*g.index_page_bytes,dtype=torch.uint8)
            main,index,rope=view_main_index(main_raw,index_raw,pages,g)
            ms=spec(logical,512,1,g.main_page_bytes);ks=spec(logical,128,4,g.index_page_bytes)
            ref_main,ref_rope=namespace['_view_nope_main_mla_cache'](ms,main_raw,Backend,128)
            ref_index=namespace['view_compressed_indexer_cache']('index',ks,index_raw,Backend,128)[0]
            assert (main.shape,main.stride(),rope.shape,rope.stride())==(ref_main.shape,ref_main.stride(),ref_rope.shape,ref_rope.stride())
            assert (index.shape,index.stride())==(ref_index.shape,ref_index.stride())
            # Byte-level independent address oracle; writes via our views are
            # observed via installed views and the raw UINT8 backing.
            for page,offset,channel in ((7,logical-1,511),(2,127,0),(1,0,13)):
                k=page*(logical//128)+offset//128;row=offset%128
                main[k,row,0,channel]=page+.25
                check(f'{tag}_main_byte_page{page}',main_raw.view(torch.bfloat16)[(page*logical+offset)*512+channel],torch.tensor(page+.25).bfloat16())
            check(f'{tag}_main_installed',main,ref_main)
            for page,pool_row,channel in ((7,logical//4-1,127),(1,0,3)):
                k=page*(logical//128)+pool_row//32;row=pool_row%32
                index[k,row,0,channel]=page+.5
                check(f'{tag}_index_byte_page{page}',index_raw.view(torch.bfloat16)[(page*(logical//4)+pool_row)*128+channel],torch.tensor(page+.5).bfloat16())
            check(f'{tag}_index_installed',index,ref_index)
            raw=torch.full((pages*g.tail_page_bytes,),0x55,dtype=torch.uint8);tail=view_tail(raw,pages,g)
            ts=SimpleNamespace(dtype=torch.float32,unpadded_page_size_bytes=layout.tail_capacity*256*4,block_size=layout.tail_capacity,head_size=128)
            ref_tail=namespace['view_kpool_tail_cache']('tail',ts,raw,pages)[0]
            assert tail.shape==ref_tail.shape and tail.stride()==ref_tail.stride()
            planes=torch.randn(2,layout.tail_capacity,128);tail[7]=pack_tail_planes(planes)
            check(f'{tag}_tail_plane_roundtrip',unpack_tail_rows(ref_tail[7]),planes)
            unused=raw[7*g.tail_page_bytes+layout.tail_capacity*256*4:]
            check(f'{tag}_tail_padding_unchanged',unused,torch.full_like(unused,0x55))
            # Fill every raw row with deterministic data; padding remains sentinel.
            for page in range(pages):
                for row in range(layout.tail_capacity):tail[page,row].fill_(page*100+row)
            starts=torch.tensor([3,128+1,512+3,0]);lengths=torch.tensor([1,7,0,0]);tail_ids=torch.tensor([7,2,5,-1])
            plan=plan_tail(starts,lengths,tail_ids,g,num_tail_pages=pages)
            ref_meta=SimpleNamespace(cann_inputs=SimpleNamespace(page_rows=g.tail_page_rows,history_indices=plan.history_indices))
            reference_state=namespace['prepare_keypool_state'](tail,starts,ref_meta,4)
            ours=prepare_tail_state(tail,plan,4)
            check(f'{tag}_history_installed',ours[1:5],reference_state[1:5])
            wanted_history=[];wanted_writes=[];wanted_state=[]
            for request,(start,length,page) in enumerate(zip(starts.tolist(),lengths.tolist(),tail_ids.tolist())):
                wanted_history.append([max(page,0)*g.tail_page_rows+(start//4*4+c)%layout.tail_capacity for c in range(4)])
                final=start+length;remaining=final%4
                wanted_writes.append([page*g.tail_page_rows+(final-remaining+c)%layout.tail_capacity if length and c<remaining and page>=0 else -1 for c in range(4)])
                base=(request+1+(4 if start%4+length>=4 else 0))*4
                wanted_state.append([base+c for c in range(4)])
            check(f'{tag}_history_addresses',plan.history_indices,torch.tensor(wanted_history))
            check(f'{tag}_write_addresses',plan.write_indices,torch.tensor(wanted_writes))
            check(f'{tag}_state_addresses',plan.state_indices,torch.tensor(wanted_state))
            # Installed state table builder executes on CPU, without vLLM import.
            dummy=SimpleNamespace(compress_ratio=4,device=torch.device('cpu'),_captured_state_tables={},_state_table_scratch={},_state_table_shapes={})
            common=SimpleNamespace(slot_mapping=torch.zeros(int(lengths.sum()),dtype=torch.int64),max_query_len=int(lengths.max()),num_input_tokens=int(lengths.sum()),num_reqs=4)
            ref_table=namespace['_get_state_table'](dummy,common,False)
            check(f'{tag}_state_table_installed',plan.state_table,ref_table)
            saved=ours.clone();tail.zero_();check(f'{tag}_history_snapshot_private',ours,saved)
            ids=torch.tensor([[7,2,-1],[1,5,0]],dtype=torch.int32)
            table=expand_kernel_block_table(ids,g,num_logical_pages=pages)
            wanted=[]
            for row in ids.tolist():wanted.append([i*(logical//128)+j if i>=0 else -1 for i in row for j in range(logical//128)])
            check(f'{tag}_kernel_table_independent',table,torch.tensor(wanted,dtype=torch.int32))
            positions=torch.arange(logical+3)
            slots=torch.tensor([(7 if p<logical else 2)*logical+p%logical for p in positions.tolist()]);slots[::13]=-1
            compressed=compressed_slot_mapping(slots,positions,layout)
            # Native cache writer interprets compressed slots in blocks of32,
            # with cache.stride(0) bytes/elements carried separately.
            for token,slot in enumerate(compressed.tolist()):
                if slot>=0:
                    index[slot//32,slot%32,0,0]=token
                    check(f'{tag}_pooled_write_token{token}',index_raw.view(torch.bfloat16)[slot*128],torch.tensor(token).bfloat16())
            report['cases'].append({'logical_block':logical,'kernel_block':128,'kernel_pages_per_logical':g.kernel_blocks_per_logical,
                'main_page_bytes':g.main_page_bytes,'index_page_bytes':g.index_page_bytes,'tail_page_bytes':g.tail_page_bytes,
                'tail_capacity':layout.tail_capacity,'physical_tail_page_rows':g.tail_page_rows,'requests':4,'main_index_padding':'rejected'})
    try:CANNCacheGeometry(DSACacheLayout.from_config(config,logical_block_size=16),8192)
    except ValueError:report['checks']['logical16_not_CANN_kernel_compatible']={'passed':True}
    else:raise AssertionError('logical16 silently bound to kernel128')
    assert 'torch_npu' not in sys.modules
    report['status']='passed'
except Exception as exc:
    report['status']='failed';report['failure']=repr(exc);raise
finally:
    report['ended_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (a.output_dir/'cpu-cann-cache-result.json').write_text(json.dumps(report,indent=2)+'\n')
