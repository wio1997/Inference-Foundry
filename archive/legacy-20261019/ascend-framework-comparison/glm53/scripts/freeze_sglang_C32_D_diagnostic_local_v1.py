"""Local-only freeze after actual C32 control/native and fresh-tail evidence."""
import ast,hashlib,importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RID='20261017T2058-sglang-C32-D-two-step-profile-v1'
CONTROLLER='397997aae778814836429fa1e6f2cd57ef242768ea06223e41ac299b4c0650a8'
def sha(relative):return hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
def read(relative):return json.loads((ROOT/relative).read_text())
def main():
    prior='results/sglang-C32-L04-control-v3-measurement-v1.json';cert=read(prior)
    assert cert['measurement_valid'] and cert['formal_requests']==128 and cert['gate_payload_reads']==384 and cert['native_postrun_audit_required'] is False
    proof_path='runs/20261017T2056-sglang-C32-fifth-suffix-inputs-v1/manifest-proof.json';proof=read(proof_path);manifest=proof['manifest']
    assert manifest['concurrency']==32 and manifest['requests']==32 and manifest['warmups']==0 and manifest['fresh_suffix_lcp_exact'] and len(manifest['rows'])==32
    raw=(json.dumps(manifest,indent=2)+'\n').encode();assert hashlib.sha256(raw).hexdigest()==proof['manifest_sha256']
    canonical=read('runs/20261017T0615-formal-inputs-v3/C32-manifest.json')['data']
    for index,row in enumerate(manifest['rows']):
        assert row['prefix_id']==index and row['variant']==5 and row['input_tokens']==80000 and row['requested_output_tokens']==600 and row['lcp_against_previous_four']==[74400]*4 and row['prefix_sha256_le_u32']==canonical['prefixes'][index]['ids_sha256_le_u32']
    accepted=read('config/C32-only-L04-restore/formal-C32-control-v3.json')
    files=dict(accepted['client_files'])
    files['glm53_C32_D_diagnostic_client_v1.py']='scripts/glm53_C32_D_diagnostic_client_v1.py'
    files['glm53_C32_D_profile_control_v1.py']='scripts/glm53_C32_D_profile_control_v1.py'
    profile=dict(project_id='glm53',framework='sglang',diagnostic_only=True,run_id=RID,
                 profile_endpoint='http://172.16.12.36:30600',metrics_endpoints=['http://172.16.12.36:30600/metrics','http://172.16.12.41:30600/metrics'],
                 profile_body={'output_dir':'/run/glm53/profile/'+RID,'num_steps':2,'activities':['CPU','GPU'],'with_stack':False,'record_shapes':False,'profile_by_stage':False,'profile_id':RID,'merge_profiles':False,'profile_prefix':'glm53-D','detailed_annotations':False})
    client=dict(project_id='glm53',framework='sglang',concurrency=32,diagnostic_only=True,performance_certificate_allowed=False,
                run_id=RID,output='/run/glm53/output',dataset='/diagnostic-inputs',prior_report='/prior-report.json',
                prior_report_sha256=cert['raw_formal_report_sha256'],prior_run_id=cert['run_id'],cache_namespace=cert['namespace'],input_manifest_sha256=proof['manifest_sha256'],profile_settings=profile)
    prerequisites={p:sha(p) for p in [prior,proof_path,'results/sglang-C32-L04-restore-startup-v1.json','results/sglang-C32-L04-before-control-native-audit-v1.json','runs/20261017T2054-sglang-C32-L04-control-native-v3/receipt.json','results/C32-two-step-profile-control-pure-fixture-v1.json']}
    settings=dict(project_id='glm53',framework='sglang',concurrency=32,diagnostic_only=True,performance_certificate_allowed=False,
                  run_id=RID,shared_run_id=accepted['shared_run_id'],shared_remote_root=accepted['shared_remote_root'],meta_container_id=accepted['meta_container_id'],runtime_containers=accepted['runtime_containers'],model_image=accepted['model_image'],
                  files=files,files_sha256={name:sha(relative) for name,relative in files.items()},
                  client_settings=client,prerequisite_sha256=prerequisites)
    assert settings['files_sha256']['glm53_formal_ais_client_v1.py']=='935a4a1c3188b3c7d2041f91e946423838dece7bcc5669daee6ae81c6e6c56a3'
    path='scripts/glm53_C32_D_diagnostic_controller_v1.py';assert sha(path)==CONTROLLER
    spec=importlib.util.spec_from_file_location('controller',ROOT/path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    payload={}
    import base64
    for name,relative in files.items():
        data=(ROOT/relative).read_bytes();payload[name]={'base64':base64.b64encode(data).decode(),'sha256':sha(relative)}
    cpu=module.render(module.CPU,{'CPU_ROOT':'/data/wio/inference-foundry/glm53/runs/'+RID,'RID':RID,'FILES':payload,'CLIENT':client})+module.codec.ENCODER;ast.parse(cpu)
    prechecks={}
    for ns,item in accepted['runtime_containers'].items():
        node=int(ns);code=module.render(module.identity.HOST+module.PRECHECK,{'CID':item['container_id'],'NAME':item['name'],'IMAGE':accepted['model_image'],'NODE':node,'URL':f'http://172.16.12.{node}:'+('30500' if node in [34,35] else '30600'),'DIAG_RID':RID})
        prechecks[ns]=hashlib.sha256(code.encode()).hexdigest()
    relative='config/C32-only-L04-restore/D-two-step-diagnostic-v1.json';p=ROOT/relative;assert not p.exists();p.write_text(json.dumps(settings,indent=2)+'\n')
    source=('"""Frozen C32 diagnostic; exact once through ZCode only."""\nimport hashlib,importlib.util\nfrom pathlib import Path\nROOT=Path(__file__).resolve().parents[1]\ndef main():\n'
            f' p=ROOT/{path!r};assert hashlib.sha256(p.read_bytes()).hexdigest()=={CONTROLLER!r}\n'
            " s=importlib.util.spec_from_file_location('controller',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n"
            f' m.execute({relative!r},{sha(relative)!r})\n'
            "if __name__=='__main__':main()\n")
    ast.parse(source);launcher='scripts/run_sglang_C32_D_two_step_diagnostic_v1.py';p=ROOT/launcher;assert not p.exists();p.write_text(source)
    capture='scripts/glm53_C32_D_native_profile_capture_v1.py';capture_sha='e1b5595e5b9ff29c43df4ef6ef68f90f583342e009717d3d545094476da785c6';assert sha(capture)==capture_sha
    spec=importlib.util.spec_from_file_location('capture_review',ROOT/capture);cap=importlib.util.module_from_spec(spec);spec.loader.exec_module(cap);native,extension_sha=cap.load()
    native_AST={}
    for ns,item in accepted['runtime_containers'].items():
        node=int(ns);values={'ARCHIVE':'/data/wio/inference-foundry/glm53/runs/20261017T2060-sglang-C32-D-two-step-native-v1','FORMAL_RID':RID,'CONCURRENCY':32,'NEED_FORMAL':False,'CID':item['container_id'],'NAME':item['name'],'IMAGE':accepted['model_image'],'NODE':node,'IDENTITY':native.identity.HOST,'IDENTITY_ENV':{'CID':item['container_id'],'NAME':item['name'],'IMAGE':accepted['model_image'],'NODE':node},'STREAM':native.STREAM,'SDK_SOURCE':native.SDK_SOURCE,'PORT':30500 if node in [34,35] else 30600,'META':accepted['meta_container_id']}
        code=native.render(native.HOST,values)+native.codec.ENCODER;ast.parse(code);native_AST[ns]=hashlib.sha256(code.encode()).hexdigest()
    post_source=source.replace(path,capture).replace(CONTROLLER,capture_sha).replace("m.execute("+repr(relative)+","+repr(sha(relative))+")", "m.execute("+repr(relative)+","+repr(sha(relative))+",'20261017T2060-sglang-C32-D-two-step-native-v1')")
    ast.parse(post_source);assert '20261017T2060-sglang-C32-D-two-step-native-v1' in post_source
    post='scripts/inspect_sglang_C32_D_two_step_native_v1.py';p=ROOT/post;assert not p.exists();p.write_text(post_source)
    review=dict(project_id='glm53',run_id=RID,diagnostic_only=True,performance_certificate_allowed=False,settings=relative,settings_sha256=sha(relative),launcher=launcher,launcher_sha256=sha(launcher),CPU_rendered_AST_sha256=hashlib.sha256(cpu.encode()).hexdigest(),native_precheck_AST_sha256=prechecks,native_capture_launcher=post,native_capture_launcher_sha256=sha(post),native_capture_AST_sha256=native_AST,dispatched=False,actual_resource_calls=0,model_requests=0)
    p=ROOT/'results/C32-D-two-step-diagnostic-frozen-review-v1.json';assert not p.exists();p.write_text(json.dumps(review,indent=2)+'\n');print(json.dumps(review,indent=2))
if __name__=='__main__':main()
