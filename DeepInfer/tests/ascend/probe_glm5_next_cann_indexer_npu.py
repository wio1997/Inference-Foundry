# SPDX-License-Identifier: Apache-2.0
"""Future authorized ABI inspection / bounded synthetic native PKI probe.

NEVER grants authorization or operates services/containers/cards. Do not execute
either native stage during preparation-only. key_pool export/schema is inspected
but key_pool numerical/state execution awaits a verified output/state ABI.
"""
import argparse,datetime,hashlib,importlib.metadata,importlib.util,json,platform
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--authorized-npu',action='store_true')
    p.add_argument('--stage',choices=['inspect-abi','pki'],default='inspect-abi')
    for name in ('assignment','profile','checkpoint','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--task-id',required=True);p.add_argument('--executor',required=True);p.add_argument('--generation',type=int,required=True)
    a=p.parse_args()
    if not a.authorized_npu:p.error('later explicit user NPU authorization required; no backend imported')
    from deepinfer.hugging_face.glm5_next_cann_probe import require_owner_authorization,require_probe_prerequisites,require_export_schemas,OPERATORS,pki_probe_cases,plan_pki_case,expected_pki_tokens
    profile=json.loads(a.profile.read_text());assignment=json.loads(a.assignment.read_text())
    identity={'task_id':a.task_id,'executor':a.executor,'generation':a.generation}
    require_owner_authorization(a.authorized_npu,assignment,identity)
    spec=importlib.util.find_spec('cann_ops_transformer')
    declaration=require_probe_prerequisites(authorized=a.authorized_npu,assignment=assignment,
        identity=identity,profile=profile,stage=a.stage,
        architecture=platform.machine(),version=importlib.metadata.version,module_origin=None if spec is None else spec.origin)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    report={'status':'in_progress','stage':a.stage,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'prerequisites':declaration,'native_imports':False,'NPU_execution':False,'key_pool_executed':False,
            'model_generation':False,'weights_read_bytes':0,'checks':{},
            'profile_sha256':hashlib.sha256(a.profile.read_bytes()).hexdigest(),
            'assignment_sha256':hashlib.sha256(a.assignment.read_bytes()).hexdigest(),
            'source_sha256':{n:hashlib.sha256((Path(__file__).resolve().parents[2]/n).read_bytes()).hexdigest() for n in [
                'deepinfer/hugging_face/glm5_next_cann_probe.py','tests/ascend/probe_glm5_next_cann_indexer_npu.py',
                'deepinfer/model_executor/models/glm5_next/cann_indexer.py']}}
    def save():
        path=a.output_dir/'cann-indexer-npu-result.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(path)
    save()
    try:
        # No torch/backend/operator import precedes the current-owner/file gate.
        import torch
        import torch_npu
        import cann_ops_transformer  # noqa: F401 -- registers native operators
        report['native_imports']=True
        schemas={name:str(getattr(torch.ops.cann_ops_transformer,name).default._schema) for name in OPERATORS}
        report['schemas']=schemas;save()
        if a.stage=='inspect-abi':
            report.update(status='inspected',ABI_validated=False)
            return
        require_export_schemas(schemas,profile['schemas'])
        from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
        from deepinfer.model_executor.models.glm5_next.cann_indexer import CANNIndexerContract,prepare_nonempty_pki_cpu,pack_cann_token_indices_cpu
        c=CANNIndexerContract.from_config(Glm5NextConfig.from_checkpoint(a.checkpoint))
        if (c.heads,c.head_dim,c.pool_size,c.topk_tokens)!=(32,128,4,2048):raise ValueError('actual GLM indexer geometry differs')
        report['config_sha256']=hashlib.sha256((a.checkpoint/'config.json').read_bytes()).hexdigest()
        def check_current_owner():
            current=json.loads(a.assignment.read_text())
            require_owner_authorization(a.authorized_npu,current,identity)
            if declaration['device_resource'] not in current.get('NPU_resources_claimed',[]):raise ValueError('device resource claim no longer present')
        check_current_owner()
        report['device_access']=True;save()
        torch.set_num_threads(2);torch_npu.npu.set_device(declaration['device'])
        device=f"npu:{declaration['device']}";name=torch_npu.npu.get_device_name(declaration['device']);report['device_name']=name
        if name!=profile['device_name']:raise ValueError('device name differs from verified910C profile')
        for case in pki_probe_cases():
            plan=plan_pki_case(case);t=case['tokens'];expected=expected_pki_tokens(case)
            if not t:
                report['checks'][case['name']]={'passed':True,'operator_called':False,'shape':[0,1,c.output_width]}
                continue
            query=torch.ones(t,c.heads,c.head_dim,dtype=torch.bfloat16);weights=torch.ones(t,c.heads,dtype=torch.bfloat16)
            ends=torch.tensor(case['ends'],dtype=torch.int32);lengths=torch.tensor(case['lengths'],dtype=torch.int32);tails=torch.tensor(case['tails'],dtype=torch.int32)
            table=torch.tensor(plan['table'],dtype=torch.int32)
            inputs=prepare_nonempty_pki_cpu(query,weights,ends,lengths,tails,table,c,num_kernel_blocks=plan['blocks'])
            cache=torch.zeros(plan['blocks'],32,1,c.head_dim,dtype=torch.bfloat16)
            for rows in plan['pool_rows']:
                for block,row,ordinal in rows:
                    cache[block,row,0,0]=1+ordinal//128;cache[block,row,0,1]=(ordinal%128)/128
            check_current_owner()
            report['NPU_execution']=True;save()
            q,w,e,l,tail,bt=[v.to(device) for v in inputs];native_cache=cache.to(device)
            before=[v.cpu().clone() for v in (q,w,e,l,tail,bt,native_cache)]
            check_current_owner()
            report['NPU_execution']=True;save()
            with torch.inference_mode():
                raw,values=torch.ops.cann_ops_transformer.pool_key_indexer(q,native_cache,w,tail,
                    actual_seq_q=e,actual_seq_k=l,block_table=bt,layout_q='TND',layout_k='PA_BBND',topk=c.topk_tokens,pool_size=c.pool_size,mask_mode=3)
            torch_npu.npu.synchronize()
            if raw.dtype!=torch.int32 or raw.ndim!=2 or raw.shape[0]!=t+1 or raw.shape[1]<c.topk_tokens or raw.device!=q.device:
                raise ValueError('native PKI raw output rank/dtype/rows/width/device differs')
            for old,current in zip(before,(q,w,e,l,tail,bt,native_cache),strict=True):
                if not torch.equal(old,current.cpu()):raise ValueError('native PKI mutated input/cache')
            packed=pack_cann_token_indices_cpu(raw[:t].cpu(),torch.tensor(case['positions']),ends,c,index_namespace='expanded_token')
            for row,wanted in enumerate(expected):
                got=packed[row,0].tolist();valid=[v for v in got if v>=0]
                if sorted(valid)!=wanted or len(valid)!=len(set(valid)) or got[len(valid):]!=[-1]*(c.output_width-len(valid)):
                    raise ValueError(f"{case['name']} row{row}: ranked history/tail/padding differs")
            report['checks'][case['name']]={'passed':True,'operator_called':True,'raw_shape':list(raw.shape),
                'raw_dtype':str(raw.dtype),'packed_shape':list(packed.shape),'cache_bytes':plan['cache_bytes'],
                'selected_counts':[len(v) for v in expected],'values_metadata':{'shape':list(values.shape),'dtype':str(values.dtype)}}
            save()
            del raw,values,q,w,e,l,tail,bt,native_cache
        report.update(status='passed',ABI_validated=False,scope='synthetic_PKI_only; key_pool and model readiness unverified')
    except BaseException as exc:
        report.update(status='failed',failure=repr(exc));raise
    finally:
        report['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
        print(json.dumps(report,indent=2))


if __name__=='__main__':main()
