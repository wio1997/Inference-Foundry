# SPDX-License-Identifier: Apache-2.0
"""Actual-dimension source-role packet and CPU address replay;no native call."""
import argparse,ast,datetime,hashlib,json,os,platform,sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import torch
from test_glm5_next_sfa_inputs import fixture,prepare
from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
from deepinfer.model_executor.models.glm5_next.absorbed_mla import AbsorbedMLAContract,split_kv_b_cpu,absorbed_sparse_mla_cpu,inspect_sfa_source_contract
from deepinfer.model_executor.models.glm5_next.private_request import CandidateDSAReadView
from deepinfer.model_executor.models.glm5_next.sfa_inputs import NoPESFAContract,SFAHeadPartition,NoPESFAInputPlan

p=argparse.ArgumentParser(description=__doc__)
for name in ('checkpoint','prior-evidence','reference-source','output-dir'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0' and not list(Path('/dev').glob('davinci[0-9]*'))
torch.set_num_threads(2);torch.manual_seed(5336);config=Glm5NextConfig.from_checkpoint(a.checkpoint);sha=hashlib.sha256((a.checkpoint/'config.json').read_bytes()).hexdigest();prior=json.loads(a.prior_evidence.read_text());assert prior['status']=='passed' and prior['config_sha256']==sha and not prior['NPU_execution']
source=a.reference_source.read_text();source_sha=hashlib.sha256(a.reference_source.read_bytes()).hexdigest();assert source_sha=='54357ea7189e849ec0aa62a7c4b6d80fe58f849047e111138be04c2981246aa7'
ledger=inspect_sfa_source_contract(source);contract=NoPESFAContract.from_config(config,reference_source_sha256=source_sha)
src=Path(__file__).resolve().parents[2];files=['deepinfer/model_executor/models/glm5_next/sfa_inputs.py','tests/ascend/test_glm5_next_sfa_inputs.py','tests/ascend/probe_glm5_next_sfa_inputs_cpu.py']
r={'status':'in_progress','native_round':36,'architecture':platform.machine(),'torch':torch.__version__,'config_sha256':sha,'prior_evidence_sha256':hashlib.sha256(a.prior_evidence.read_bytes()).hexdigest(),'reference_source_sha256':source_sha,
 'source_sha256':{n:hashlib.sha256((src/n).read_bytes()).hexdigest() for n in files},'NPU_execution':False,'native_imports':False,'native_ABI_validated':False,'operator_dispatched':False,'native_page_leases_or_completion':False,'model_generation':False,'weights_read_bytes':0,'full45_layer_forward':False,
 'scope':'source-aligned single-request BF16 NoPE CPU input roles;synthetic cache/query/weights;CPU replay is not native execution','checks':{},'cases':[],'source_contract':ledger,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def check(name,value):
    if name in r['checks']:raise AssertionError('duplicate check name '+name)
    if not value:raise AssertionError(name)
    r['checks'][name]={'passed':True}
try:
    check('source_roles_ast_verified_not_imported',ledger['source_only'] and not ledger['native_ABI_validated'])
    check('actual_contract_scale_geometry',(contract.total_heads,contract.original_qk_dim,contract.latent_dim,contract.scale)==(64,256,512,1/16))
    check('actual11DSA_layers',tuple(s.layer_id for s in config.layers if s.attention=='deepseek_sparse_attention')==tuple(range(3,44,4)))
    for case,(start,count,B,world,rank,layer) in enumerate(((127,5,128,16,0,3),(255,4,256,16,15,43),(2051,4,128,32,31,43))):
        tag=f'case{case}';plan,g,base,stage,_,_,_=fixture(start,count,B,world,rank);plan=replace(plan,contract=contract,layer_id=layer,revision=7)
        check(tag+'/actual_layer_is_DSA',config.layers[layer].attention=='deepseek_sparse_attention')
        # Existing CPU private selection reads candidate compressed pages.
        with torch.inference_mode():
            idx=stage.index.view(torch.bfloat16).reshape(len(stage.plan.private_ids),B//4,128)
            changed={logical:i for i,logical in enumerate(stage.plan.changed_logical_pages)}
            for pool in range(start//4,(start+count)//4):idx[changed[(pool*4)//B],pool%(B//4)]=10
        index_base=torch.zeros(plan.candidate.num_main_pages*g.index_page_bytes,dtype=torch.uint8)
        if start==2051:
            index_base.view(torch.bfloat16).reshape(plan.candidate.num_main_pages,B//4,128)[plan.candidate.committed_table[0],0]=-100
        view=CandidateDSAReadView(SimpleNamespace(main_raw=base,index_raw=index_base),stage,g)
        iq=torch.ones(count,32,128,dtype=torch.bfloat16);hw=torch.full((count,32),1/32,dtype=torch.bfloat16);selection=view.select_pools(iq,hw)
        ac=AbsorbedMLAContract.from_config(config,plan.heads.head_ids);weights=split_kv_b_cpu((torch.randn(ac.heads*(256+256),512)*.02).bfloat16(),ac);query=torch.full((count,ac.heads,256),1/16,dtype=torch.bfloat16)
        positions=torch.tensor(view.positions);latent=view.gather(torch.arange(view.tokens_seen));oracle=absorbed_sparse_mla_cpu(query,latent,selection.token_indices,positions,weights,sink_policy='absent')
        from deepinfer.model_executor.models.glm5_next.cann_indexer import CANNIndexerContract,pack_cann_token_indices_cpu
        packed=pack_cann_token_indices_cpu(selection.token_indices,positions,torch.tensor([count]),CANNIndexerContract(32,128,4,2048),index_namespace='expanded_token')
        check(tag+'/existing_CPU_indexer_packer_compatible',torch.equal(packed,selection.token_indices[:,None]))
        before=base.clone();packet=prepare(plan,base,stage,oracle.query_latent,packed);kw=packet.source_kwargs_cpu()
        check(tag+'/exact_source_keyword_roles',set(kw)==set(ledger['native_call']))
        literals=('sparse_block_size','query_rope','key_rope','layout_query','layout_kv','sparse_mode','attention_mode','return_softmax_lse')
        check(tag+'/source_literals_match',all(kw[k]==ast.literal_eval(ledger['native_call'][k]) for k in literals))
        check(tag+'/key_value_share_one_CPU_backing',kw['key'] is kw['value'] and kw['key'] is packet.cache and packet.cache.device.type=='cpu')
        check(tag+'/absorbed_query_shape',tuple(packet.query.shape)==plan.query_shape and packet.query.shape[-1]==512)
        check(tag+'/head_partition_complete',tuple(plan.heads.head_ids)==tuple(range(rank*(64//world),(rank+1)*(64//world))))
        check(tag+'/query_ends_no_leading_zero',packet.query_ends.dtype==torch.int32 and packet.query_ends.tolist()==[count])
        check(tag+'/KV_length_raw_tokens',packet.kv_lengths.dtype==torch.int32 and packet.kv_lengths.tolist()==[start+count] and packet.kv_lengths.item()!=view.tokens_seen//4)
        check(tag+'/kernel_table_matches_candidate',packet.block_table.dtype==torch.int32 and packet.block_table.tolist()==[list(stage.plan.kernel_read_table)])
        check(tag+'/original_QK_scale_retained',packet.scale==1/16 and packet.scale!=512**-.5)
        check(tag+'/token_namespace_INT32_T1N',packet.sparse_indices.dtype==torch.int32 and tuple(packet.sparse_indices.shape)==(count,1,2051))
        output=torch.zeros_like(packet.query);lse=torch.zeros(count,ac.heads)
        for row in range(count):
            ids=packet.sparse_indices[row,0];ids=ids[ids>=0].long();physical=packet.block_table[0,ids//128].long();cache=packet.cache[physical,ids%128,0]
            check(tag+'/candidate_address_gather_row'+str(row),torch.equal(cache,latent[ids]))
            logits=torch.einsum('hl,sl->hs',packet.query[row].float(),cache.float())*packet.scale
            output[row]=torch.einsum('hs,sl->hl',torch.softmax(logits,dim=-1),cache.float()).bfloat16();lse[row]=torch.logsumexp(logits,dim=-1)
        check(tag+'/CPU_address_replay_attention_latent_bitwise',torch.equal(output,oracle.attention_latent) and torch.equal(lse,oracle.lse))
        check(tag+'/base_backing_unchanged',torch.equal(before,base))
        check(tag+'/no_ABI_or_native_ownership_claim',not packet.native_ABI_validated and r['operator_dispatched'] is False)
        if start==2051:check(tag+'/newpool512_in_top512_and0_dropped',bool((selection.pool_ids==512).any(1).all()) and not bool((selection.pool_ids==0).any()))
        r['cases'].append({'start':start,'tokens':count,'logical_block':B,'layer':layer,'declared_world':world,'rank':rank,'heads':plan.heads.head_ids,'query_shape':list(packet.query.shape),'cache_shape':list(packet.cache.shape),'raw_frame_bytes':base.numel(),'selection_completed_pools':view.tokens_seen//4})
    full=replace(plan,heads=SFAHeadPartition(64,1,0,tuple(range(64))))
    check('unsharded64head_descriptor_only_no_math',full.query_shape==(4,64,512))
    check('no_native_backend_modules_imported',not any(n in sys.modules for n in ('torch_npu','fla_npu','_C_ascend')))
    r['status']='passed'
except BaseException as e:r.update(status='failed',failure=repr(e));raise
finally:
    r['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();a.output_dir.mkdir(parents=True,exist_ok=True);(a.output_dir/'sfa-input-result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
