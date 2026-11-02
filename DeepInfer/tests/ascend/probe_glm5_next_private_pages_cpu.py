# SPDX-License-Identifier: Apache-2.0
"""CPU-only private page/read-table fixtures and extracted wrapper dispatch stubs."""
import argparse,ast,copy,datetime,hashlib,json,os,platform,__future__
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
from deepinfer.model_executor.models.glm5_next.cache import DSACacheLayout
from deepinfer.model_executor.models.glm5_next.cann_cache import CANNCacheGeometry,view_main_index,view_tail,expand_kernel_block_table
from deepinfer.model_executor.models.glm5_next.private_pages import plan_private_dsa_pages,prepare_private_dsa_cpu,gather_private_latent_cpu,inspect_ascend_wrapper_source

p=argparse.ArgumentParser(description=__doc__)
for n in ('checkpoint','wrapper-reference','utils-reference','prior-evidence','output-dir'):p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args();assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0' and not list(Path('/dev').glob('davinci[0-9]*'))
torch.set_num_threads(2);config=Glm5NextConfig.from_checkpoint(a.checkpoint);prior=json.loads(a.prior_evidence.read_text());config_sha=hashlib.sha256((a.checkpoint/'config.json').read_bytes()).hexdigest()
assert prior['status']=='passed' and not prior['NPU_execution'] and prior['config_sha256']==config_sha
src=Path(__file__).resolve().parents[2];files=['deepinfer/model_executor/models/glm5_next/private_pages.py','tests/ascend/test_glm5_next_private_pages.py','tests/ascend/probe_glm5_next_private_pages_cpu.py']
report={'status':'in_progress','native_round':32,'NPU_execution':False,'native_imports':False,'native_ABI_validated':False,'model_generation':False,'weights_read_bytes':0,
 'architecture':platform.machine(),'torch':torch.__version__,'config_sha256':config_sha,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'source_sha256':{n:hashlib.sha256((src/n).read_bytes()).hexdigest() for n in files},'reference_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.wrapper_reference,a.utils_reference)},
 'prior_evidence_sha256':hashlib.sha256(a.prior_evidence.read_bytes()).hexdigest(),'checks':{},'cases':[]}
def check(name,value):
    if not value:raise AssertionError(name)
    report['checks'][name]={'passed':True}
def extracted(node,namespace):
    node=copy.deepcopy(node);node.decorator_list=[];exec(compile(ast.Module(body=[node],type_ignores=[]),'extracted_CPU_dispatch','exec',flags=__future__.annotations.compiler_flag),namespace);return namespace[node.name]
try:
    source=a.wrapper_reference.read_text();utils=a.utils_reference.read_text();report['source_contract']=inspect_ascend_wrapper_source(source,utils)
    check('source_only_OOT_context_ledger',report['source_contract']['source_only'] and not report['source_contract']['native_ABI_validated'])
    for name,altered,other in [('mapping',source,utils.replace('"MultiHeadLatentAttentionWrapper": AscendMultiHeadLatentAttention','"MultiHeadLatentAttentionWrapper": WrongWrapper')),
        ('application',source,utils.replace('CustomOp.register_oot(_decorated_op_cls=op_cls, name=name)','CustomOp.register_oot(_decorated_op_cls=op_cls, name="wrong")')),
        ('norm_wiring',source.replace('q_a_layernorm=mla_modules.q_a_layernorm','q_a_layernorm=mla_modules.kv_a_layernorm'),utils),
        ('cache_owner',source.replace('kv_cache = self.mla_attn.kv_cache','kv_cache = output'),utils),
        ('raw_hidden',source.replace('torch.ops.vllm.mla_forward(hidden_states, output, self.prefix)','torch.ops.vllm.mla_forward(positions, output, self.prefix)'),utils)]:
        try:inspect_ascend_wrapper_source(altered,other)
        except ValueError:check('source_drift_'+name,True)
        else:raise AssertionError('source drift accepted:'+name)
    tree=ast.parse(source);outer_cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AscendMultiHeadLatentAttention');outer=next(n for n in outer_cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    inner=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='mla_forward');events=[];context_cache=object();context_metadata=object()
    def backend(layer,hidden,cache,metadata,output,**kw):events.append((layer,hidden,cache,metadata,kw));output.copy_(hidden+1)
    obj=NS(hidden_size=4096,output_token_shard_size=1,prefix='fixture-layer',mla_attn=NS(layer_name='fixture-attn',kv_cache=context_cache,impl=NS(forward=backend)))
    ctx=NS(no_compile_layers={'fixture-layer':obj},attn_metadata={'fixture-attn':context_metadata})
    low=extracted(inner,{'get_forward_context':lambda:ctx});proxy=NS(empty=torch.empty,ops=NS(vllm=NS(mla_forward=low)))
    high=extracted(outer,{'torch':proxy});hidden=torch.arange(2*4096).reshape(2,4096).bfloat16();position=torch.tensor([10,11]);ignored_cache=object();ignored_meta=object()
    output=high(obj,position,hidden,kv_cache=ignored_cache,attn_metadata=ignored_meta)
    check('extracted_wrapper_uses_context_not_optional_override',len(events)==1 and events[0][1] is hidden and events[0][2] is context_cache and events[0][3] is context_metadata)
    check('extracted_raw_hidden_output_dispatch_stub',torch.equal(output,hidden+1) and output.shape==hidden.shape)
    index_cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='IndexerWrapper');index_method=next(n for n in index_cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward');index_events=[]
    def index_impl(*args,**kwargs):index_events.append((args,kwargs));return 'fixture-result'
    index=extracted(index_method,{});rank=object();khidden=object();meta=object();handle=object()
    result=index(NS(impl=index_impl),hidden,rank,khidden,meta,False,attn_q_gather_handle=handle)
    check('extracted_indexer_preserves_roles_compute_flag',result=='fixture-result' and index_events[0][0]==(hidden,rank,khidden,meta,False) and index_events[0][1]['attn_q_gather_handle'] is handle)
    check('actual_GLM_geometry',config.mla_dimensions['kv_lora_rank']==512 and config.indexer_config['index_head_dim']==128 and config.indexer_config['index_kpool']==4)
    for number,(start,tokens,block) in enumerate(((0,1,128),(3,2,128),(127,5,128),(255,4,128),(255,4,256),(128,0,128),(256,128,128))):
        tag=f'case{number}';layout=DSACacheLayout.from_config(config,logical_block_size=block,lookahead=3);g=CANNCacheGeometry(layout,13*256*4);pages=10;tails=5
        prefix=tuple((1,4,2)[i] for i in range((start+block-1)//block));count=(start+tokens-1)//block-start//block+1 if tokens else 0
        plan=plan_private_dsa_pages(start,tokens,prefix,(8,3)[:count],geometry=g,num_main_pages=pages,num_tail_pages=tails,occupied_main=frozenset((*prefix,5)),occupied_tail=frozenset({0,1}),tail_source=0 if start else None,tail_private=3 if tokens else None)
        main=torch.full((pages*g.main_page_bytes,),0x55,dtype=torch.uint8);index_raw=torch.full((pages*g.index_page_bytes,),0x33,dtype=torch.uint8);tail_raw=torch.full((tails*g.tail_page_bytes,),0x66,dtype=torch.uint8)
        old_latent=((torch.arange(start)[:,None]%97)+(torch.arange(512)[None,:]%13)*.25).bfloat16();new_latent=((torch.arange(start,plan.end)[:,None]%97)+(torch.arange(512)[None,:]%13)*.25).bfloat16()
        old_compressed=((torch.arange(start//4)[:,None]%31)+(torch.arange(128)[None,:]%7)*.5).bfloat16();new_compressed=((torch.arange(start//4,plan.end//4)[:,None]%31)+(torch.arange(128)[None,:]%7)*.5).bfloat16()
        main_view=main.view(torch.bfloat16).reshape(pages,block,512);idx_view=index_raw.view(torch.bfloat16).reshape(pages,block//4,128)
        for logical,physical in enumerate(prefix):
            a0=logical*block;a1=min(start,a0+block);main_view[physical,:a1-a0]=old_latent[a0:a1]
            p0=logical*(block//4);p1=min(start//4,p0+block//4);idx_view[physical,:p1-p0]=old_compressed[p0:p1]
        saved=[v.clone() for v in (main,index_raw,tail_raw)];tail_after=torch.arange(layout.tail_capacity*256).reshape(layout.tail_capacity,256).float() if tokens else None
        stage=prepare_private_dsa_cpu(main,index_raw,tail_raw,plan,g,new_latent,new_compressed,tail_after)
        check(tag+'/committed_all_bytes_unchanged',all(torch.equal(v,b) for v,b in zip((main,index_raw,tail_raw),saved)))
        expected=torch.cat((old_latent,new_latent));ids=torch.arange(plan.end,dtype=torch.int64);read=gather_private_latent_cpu(main,stage,g,ids)
        check(tag+'/logical_prefix_plus_append',torch.equal(read,expected))
        table=expand_kernel_block_table(torch.tensor([plan.read_table],dtype=torch.int32),g,num_logical_pages=pages)
        check(tag+'/kernel_table',table.flatten().tolist()==list(plan.kernel_read_table))
        # CPU TEST CLONES ONLY: install compact staged bytes at declared IDs
        # to independently check compatibility with existing kernel128 views.
        installed_main=main.clone();installed_index=index_raw.clone()
        for row,page in enumerate(plan.private_ids):
            installed_main.reshape(pages,-1)[page]=stage.main[row];installed_index.reshape(pages,-1)[page]=stage.index[row]
        kernel_main,kernel_index,_=view_main_index(installed_main,installed_index,pages,g)
        observed=torch.stack([kernel_main[plan.kernel_read_table[pos//128],pos%128,0] for pos in range(plan.end)]) if plan.end else torch.empty(0,512,dtype=torch.bfloat16)
        check(tag+'/installed_testclone_kernel_main',torch.equal(observed,expected))
        compressed_expected=torch.cat((old_compressed,new_compressed));pool_count=plan.end//4
        cmp_observed=torch.stack([kernel_index[plan.kernel_read_table[(pool*4)//128],pool%32,0] for pool in range(pool_count)]) if pool_count else torch.empty(0,128,dtype=torch.bfloat16)
        check(tag+'/coupled_index_pool_history',torch.equal(cmp_observed,compressed_expected))
        if tokens:
            logical_tail=stage.tail.view(torch.float32).reshape(1,13,256)[0,:layout.tail_capacity]
            padding=stage.tail[0,layout.tail_capacity*256*4:];sentinel=0x66 if plan.tail_source is not None else 0
            check(tag+'/tail_payload_padding',torch.equal(logical_tail,tail_after) and bool((padding==sentinel).all()))
        else:check(tag+'/empty_no_private_bytes',stage.main.numel()+stage.index.numel()+stage.tail.numel()==0)
        size=sum(t.numel()*t.element_size() for t in (stage.main,stage.index,stage.tail));check(tag+'/bounded_private_stage',size<=16*1024**2)
        report['cases'].append({'start':start,'tokens':tokens,'block_size':block,'committed_table':prefix,'read_table':plan.read_table,'kernel_read_table':plan.kernel_read_table,'new_pools':plan.new_pools,'private_bytes':size})
    report['status']='passed'
except BaseException as exc:report.update(status='failed',failure=repr(exc));raise
finally:
    a.output_dir.mkdir(parents=True,exist_ok=True);report['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (a.output_dir/'private-pages-result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
