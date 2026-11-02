# SPDX-License-Identifier: Apache-2.0
"""Actual-config CANN input/token packing references; native ABI unverified."""
import argparse,ast,datetime,hashlib,json,os,platform,sys
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
from deepinfer.model_executor.models.glm5_next.cann_indexer import CANNIndexerContract,prepare_nonempty_pki_cpu,pack_cann_token_indices_cpu,validate_indexer_source_calls

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--indexer-reference',type=Path,required=True)
p.add_argument('--adapter-reference',type=Path,required=True);p.add_argument('--sentinel-reference',type=Path,required=True);p.add_argument('--kernel-reference',type=Path,required=True)
p.add_argument('--prior-cache',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0' and not list(Path('/dev').glob('davinci[0-9]*'))
torch.set_num_threads(2);src=Path(__file__).resolve().parents[2];old=json.loads(a.prior_cache.read_bytes());config_sha=hashlib.sha256((a.checkpoint/'config.json').read_bytes()).hexdigest();assert old['status']=='passed' and not old['NPU_execution'] and old['config_sha256']==config_sha
refs=json.loads(a.kernel_reference.read_bytes());kernel=next(f for f in refs['files'] if f['path'].endswith('glm5_next_cann_keypool.py'));assert hashlib.sha256(kernel['content'].encode()).hexdigest()==kernel['sha256']
assert hashlib.sha256(a.indexer_reference.read_bytes()).hexdigest()==old['reference_sha256']['indexer_kpool.py']
assert hashlib.sha256(a.adapter_reference.read_bytes()).hexdigest()==old['reference_sha256']['cann_keypool.py']
assert kernel['sha256']==old['reference_sha256'][kernel['path']]
files=['deepinfer/model_executor/models/glm5_next/cann_indexer.py','tests/ascend/test_glm5_next_cann_indexer.py','tests/ascend/probe_glm5_next_cann_indexer_cpu.py']
report={'status':'in_progress','native_round':29,'NPU_execution':False,'Triton_kernel_executed':False,'model_generation':False,'weights_read_bytes':0,'native_ABI_validated':False,
        'architecture':platform.machine(),'torch':torch.__version__,'config_sha256':config_sha,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'reused_evidence_sha256':hashlib.sha256(a.prior_cache.read_bytes()).hexdigest(),'reference_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (a.indexer_reference,a.adapter_reference,a.sentinel_reference)},
        'source_sha256':{n:hashlib.sha256((src/n).read_bytes()).hexdigest() for n in files},'checks':{},'cases':[]}
report['reference_sha256'][kernel['path']]=kernel['sha256']


def check(name,value):
    assert value,name
    report['checks'][name]={'passed':True}


def raw(t):return t.contiguous().view(torch.uint8).numpy().tobytes() if t.numel() else b''


class DispatchCapture:
    """Record only launch arguments; never execute a Triton/native kernel."""
    def __init__(self):self.calls=[]
    def __getitem__(self,grid):
        def record(*args):self.calls.append((grid,args))
        return record


def function(tree,name,namespace):
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'extracted_wrapper','exec'),namespace)
    return namespace[name]


try:
    c=CANNIndexerContract.from_config(Glm5NextConfig.from_checkpoint(a.checkpoint));report['contract']=c.to_dict()
    check('actual_indexer_contract',(c.heads,c.head_dim,c.pool_size,c.topk_tokens,c.output_width)==(32,128,4,2048,2051))
    report['source_calls']=validate_indexer_source_calls(a.indexer_reference.read_text(),a.adapter_reference.read_text())
    check('installed_source_calls_only_not_binaryABI',report['source_calls']['source_only'] and not report['source_calls']['native_ABI_validated'])
    tree=ast.parse(a.sentinel_reference.read_text());kn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_prepare_nonempty_keypool_inputs_kernel')
    stores={ast.unparse(n.args[0]):n.args[1] for n in ast.walk(kn) if isinstance(n,ast.Call) and ast.unparse(n.func)=='tl.store'}
    for target,value in [('padded_lengths + offsets','tl.maximum(length, 1)'),('padded_ends + offsets','tl.where(offsets < requests, end, tokens + 1)'),('padded_tails + offsets','tail')]:
        assert ast.dump(stores[target])==ast.dump(ast.parse(value,mode='eval').body)
    loads=[ast.unparse(n) for n in ast.walk(kn) if isinstance(n,ast.Call) and ast.unparse(n.func)=='tl.load']
    check('installed_sentinel_kernel_AST_clamp_and_zeros',any('query + offsets' in t and 'other=0' in t for t in loads) and any('table + offsets' in t and 'other=0' in t for t in loads))
    ktree=ast.parse(kernel['content']);packnode=next(n for n in ktree.body if isinstance(n,ast.FunctionDef) and n.name=='_pack_indices_kernel')
    assignments={ast.unparse(n.targets[0]):ast.unparse(n.value) for n in ast.walk(packnode) if isinstance(n,ast.Assign)}
    index_loads=[ast.unparse(n) for n in ast.walk(packnode) if isinstance(n,ast.Call) and ast.unparse(n.func)=='tl.load']
    check('installed_pack_original_ends_and_token_load',assignments['valid_row']=='row < tl.load(cumulative + num_reqs - 1)' and any('indices + row * INPUT_STRIDE + cols' in load for load in index_loads))
    # Record the extracted wrapper's allocations/contiguous launch inputs only.
    # Kernel output buffers remain uninitialized and are never read as data.
    sentinel_dispatch=DispatchCapture();pack_dispatch=DispatchCapture();namespace={'triton':NS(cdiv=lambda a,b:(a+b-1)//b,next_power_of_2=lambda x:1<<(x-1).bit_length()),'_prepare_nonempty_keypool_inputs_kernel':sentinel_dispatch,'_pack_indices_kernel':pack_dispatch}
    prepare_ref=function(tree,'prepare_nonempty_keypool_inputs',namespace);pack_ref=function(ktree,'pack_keypool_indices',namespace)
    cases=[(0,[],[],[],[],1),(3,[3],[0],[3],[0,1,2],2),(4,[0,4],[0,1],[0,0],[0,1,2,3],2),(4,[0,0],[0,0],[0,0],[-999]*4,1),(5,[2,4],[1,3],[0,0],[2,3,10,11,-999],2),(2,[2],[33],[1],[131,132],2),(128,[128],[32],[0],list(range(128)),1)]
    for case,(T,ends_list,lengths_list,tails_list,position_list,cols) in enumerate(cases):
        n=len(ends_list);q=torch.arange(T*32*256).reshape(T,32,256).bfloat16()[:,:,::2];w=-torch.arange(T*64).reshape(T,64).bfloat16()[:,::2]
        ends=torch.tensor(ends_list,dtype=torch.int64);lengths=torch.tensor(lengths_list,dtype=torch.int64);tails=torch.tensor(tails_list,dtype=torch.int64);table=torch.zeros((n,cols),dtype=torch.int32)
        if n and cols>1:table[:,-1]=1
        args=(q,w,ends,lengths,tails,table);out=prepare_nonempty_pki_cpu(*args,c,num_kernel_blocks=2)
        check(f'case{case}_query_weight_bytes_and_dummy',raw(out[0])==raw(q)+bytes(32*128*2) and raw(out[1])==raw(w)+bytes(32*2))
        check(f'case{case}_metadata_scalar_clamp',out[2].tolist()==ends_list+[T+1] and out[3].tolist()==[max(x,1) for x in lengths_list]+[1] and out[4].tolist()==tails_list+[0] and out[5].tolist()==table.tolist()+[[0]*cols])
        ref=prepare_ref(*args);grid,launch=sentinel_dispatch.calls[-1]
        check(f'case{case}_extracted_wrapper_shapes_dispatch',all((x.shape,x.dtype)==(y.shape,y.dtype) for x,y in zip(out,ref)) and all(t.is_contiguous() for t in launch[:6]) and tuple(launch[12:])==(T,n,n,32,128,cols,8192))
        positions=torch.tensor(position_list,dtype=torch.int64);actual=ends_list[-1] if n else 0;indices=torch.full((T,2048),12345,dtype=torch.int32);expected=[]
        for row,pos in enumerate(position_list):
            selected=[]
            if row<actual:
                start=(pos+1)//4*4;history=min(start,2048);indices[row,:history]=torch.arange(history,dtype=torch.int32)
                selected=list(range(history))+list(range(start,pos+1))
            expected.append([selected+[-1]*(2051-len(selected))])
        packed=pack_cann_token_indices_cpu(indices,positions,ends,c,index_namespace='expanded_token')
        check(f'case{case}_scalar_token_pack',packed.tolist()==expected and packed.shape==(T,1,2051))
        output_buffer=torch.empty((T,2051),dtype=torch.int32);ref_packed=pack_ref(indices,NS(positions=positions,cum_query_lens=ends),2048,4,output_buffer)
        pgrid,plaunch=pack_dispatch.calls[-1]
        check(f'case{case}_extracted_pack_wrapper_explicit_width',ref_packed.shape==packed.shape and pgrid==(T,) and plaunch[9]==2051)
        report['cases'].append({'tokens':T,'real_requests':n,'actual_queries':actual,'compressed_lengths':lengths_list,'padded_shapes':[list(t.shape) for t in out],'padded_bytes':sum(t.numel()*t.element_size() for t in out),'packed_shape':list(packed.shape),'reference_execution':'allocation/argument stubs only;no kernel'})
    check('native_imports_absent',all(name not in sys.modules for name in ('torch_npu','triton','vllm','cann_ops_transformer')))
    report['status']='passed'
except Exception as exc:
    report['status']='failed';report['failure']=repr(exc);raise
finally:
    report['ended_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (a.output_dir/'cann-indexer-result.json').write_text(json.dumps(report,indent=2)+'\n')
