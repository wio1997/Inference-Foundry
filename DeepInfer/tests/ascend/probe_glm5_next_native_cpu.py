# SPDX-License-Identifier: Apache-2.0
"""Metadata-only actual selected-expert budgets; no weights or native execution."""
import argparse,ast,datetime,hashlib,json,os,platform,sys
from pathlib import Path
from deepinfer.hugging_face.glm5_next_native import KDA_PROFILE,KDA_W8A8_BINDINGS,NativeDependencyDeclaration,plan_expert_residency

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--prior-banks',type=Path,required=True)
p.add_argument('--prior-boundary',type=Path,required=True);p.add_argument('--historical-kda',type=Path,required=True)
p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0' and not list(Path('/dev').glob('davinci[0-9]*'))
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
inventory=NativeDependencyDeclaration.from_metadata();dependency_report=inventory.report()
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
old=json.loads(a.prior_banks.read_bytes());boundary=json.loads(a.prior_boundary.read_bytes());history=json.loads(a.historical_kda.read_bytes())
assert old['status']==boundary['status']=='passed' and old['NPU_execution'] is boundary['NPU_execution'] is False
assert history['architecture']=='aarch64' and dict(KDA_PROFILE)==history['versions']
assert all(v['passed'] for v in history['checks'].values())
config_sha=hashlib.sha256((a.checkpoint/'config.json').read_bytes()).hexdigest();assert config_sha==old['config_sha256']==boundary['config_sha256']
src=Path(__file__).resolve().parents[2]
files=['deepinfer/hugging_face/glm5_next_native.py','deepinfer/model_executor/models/glm5_next/expert_bank.py','tests/ascend/test_glm5_next_native.py','tests/ascend/probe_glm5_next_native_cpu.py']
report={'status':'in_progress','native_round':25,'NPU_execution':False,'weights_read_bytes':0,'expert_FFN_executed':False,
        'architecture':platform.machine(),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'config_sha256':config_sha,
        'reused_evidence_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (a.prior_banks,a.prior_boundary,a.historical_kda)},
        'source_sha256':{n:hashlib.sha256((src/n).read_bytes()).hexdigest() for n in files},'dependency_declaration':dependency_report,'checks':{},'layers':{}}

def check(name,condition):
    assert condition,name
    report['checks'][name]={'passed':True}

try:
    check('metadata_only_import_does_not_load_torch', 'torch' not in sys.modules and 'torch_npu' not in sys.modules)
    check('KDA_profile_matches_retained_native_evidence',dict(KDA_PROFILE)==history['versions'])
    check('metadata_inventory_does_not_claim_exports',len(dependency_report['missing_bindings'])==6 and not dependency_report['declaration_matches'])
    check('matching_declaration_still_unverified',not NativeDependencyDeclaration('aarch64',KDA_PROFILE,KDA_W8A8_BINDINGS).require_matching_declaration()['native_execution_validated'])
    # Metadata checks below use the existing reader and expert manifest; no
    # native import or checkpoint payload access is allowed.
    from deepinfer.hugging_face.modelslim910c import ModelSlimCheckpoint
    from deepinfer.model_executor.models.glm5_next.expert_bank import selected_expert_manifest
    class NoPayloadReader(ModelSlimCheckpoint):
        def denied(self,*args,**kwargs):raise AssertionError('weight payload access is forbidden')
        tensor=matrix=linear_payload=raw_rows=row_slice=denied
    reader=NoPayloadReader(a.checkpoint,max_read_bytes=256*1024)
    for layer,previous in old['layers'].items():
        ids=previous['bank_global_ids'];layer_id=int(layer)
        layer_report={}
        for label,chosen,shared in (('retained_bank',ids,True),('bounded_eight',ids[:8],True),('shared_only',[],True),('empty',[],False)):
            manifest=selected_expert_manifest(reader,layer_id,chosen,include_shared=shared)
            plan=plan_expert_residency(manifest,max_checkpoint_bytes=2**60,max_resident_bytes=2**60)
            checkpoint=resident=largest=0
            for spec in (*manifest.routed,*(() if manifest.shared is None else (manifest.shared,))):
                for prefix in spec.prefixes:
                    weight=reader.descriptor(prefix+'.weight')[2];scale=reader.descriptor(prefix+'.weight_scale')[2];offset=reader.descriptor(prefix+'.weight_offset')[2]
                    assert (weight['dtype'],scale['dtype'],offset['dtype'])==('I8','F32','F32')
                    weight_bytes=weight['data_offsets'][1]-weight['data_offsets'][0]
                    assert weight_bytes==weight['shape'][0]*weight['shape'][1]
                    checkpoint+=sum(d['data_offsets'][1]-d['data_offsets'][0] for d in (weight,scale,offset))
                    resident+=weight_bytes+2*scale['shape'][0];largest=max(largest,weight_bytes)
            check(f'layer{layer}_{label}_independent_header_bytes',(plan.checkpoint_payload_bytes,plan.resident_logical_bytes,plan.largest_weight_bytes)==(checkpoint,resident,largest))
            check(f'layer{layer}_{label}_preserves_global_ids',[s.global_id for s in manifest.routed]==chosen and plan.routed_experts==len(chosen) and plan.includes_shared==shared)
            if checkpoint:
                for budget_kind in ('checkpoint','resident'):
                    limits={'max_checkpoint_bytes':checkpoint,'max_resident_bytes':resident};limits['max_'+budget_kind+'_bytes']-=1
                    try:plan_expert_residency(manifest,**limits)
                    except ValueError:pass
                    else:raise AssertionError('budget not enforced')
                check(f'layer{layer}_{label}_both_exact_minus_one_reject',True)
            layer_report[label]=plan.to_dict()
        report['layers'][layer]=layer_report
    # Verify guard ordering in the actual native factory without calling it.
    tree=ast.parse((src/files[1]).read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_selected_expert_bank')
    budget=next(n.lineno for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='plan_expert_residency')
    native=next(n.lineno for n in fn.body if isinstance(n,ast.ImportFrom) and n.module=='deepinfer.npu_ops.w8a8')
    payload=next(n.lineno for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='linear_payload')
    check('native_loader_budget_before_import_and_payload',budget<native<payload)
    check('no_native_imports', 'torch_npu' not in sys.modules and 'fla_npu' not in sys.modules)
    report['status']='passed'
except Exception as exc:
    report['status']='failed';report['failure']=repr(exc);raise
finally:
    report['ended_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (a.output_dir/'native-preparation-result.json').write_text(json.dumps(report,indent=2)+'\n')
