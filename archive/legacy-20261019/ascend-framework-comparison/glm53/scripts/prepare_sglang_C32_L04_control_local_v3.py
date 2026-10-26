"""Prepare only C32 control, reusing identical-program semantics across a restart.

Local preparation only. Actual inference/capture must use reviewed ZCode launchers.
No historical qualification or measurement file is overwritten.
"""
import ast, hashlib, importlib.util, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RID = '20261017T2052-sglang-formal-C32-L04-control-v3'
POST = '20261017T2054-sglang-C32-L04-control-native-v3'
LAUNCH = 'c57a105a1803ca1d28bcd098f6ea722c77883757ecbd3180c00ae5798b8a4b2d'
CAPTURE = '91a9b273904338516c8be93d0e953a977bece82709deaa10346147a588701a6b'

def sha(relative):
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()

def read(relative):
    return json.loads((ROOT / relative).read_text())

def fresh(relative, value):
    path = ROOT / relative
    assert not path.exists(), ('Do not overwrite frozen evidence', relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')

def load(relative, digest):
    assert sha(relative) == digest
    spec = importlib.util.spec_from_file_location('reviewed', ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def wrapper(relative, digest, call):
    source = ('"""Frozen C32 L04 control; execute via ZCode exactly once."""\n'
              'import hashlib,importlib.util\nfrom pathlib import Path\n'
              'ROOT=Path(__file__).resolve().parents[1]\n'
              'def main():\n'
              f' p=ROOT/{relative!r};assert hashlib.sha256(p.read_bytes()).hexdigest()=={digest!r}\n'
              " s=importlib.util.spec_from_file_location('operation',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n"
              f' {call}\n'
              "if __name__=='__main__':main()\n")
    ast.parse(source)
    return source

def prepare():
    runtime = read('config/C32-only-L04-restore/runtime.json')
    assert runtime['concurrency'] == 32 and not runtime['L05_P_cache_candidate_enabled']
    startup = read('results/sglang-C32-L04-restore-startup-v1.json')
    equivalent = read('results/sglang-C32-L04-program-equivalence-v1.json')
    old = read('results/sglang-metadata-native-readiness-v1.json')
    assert startup['pass'] and startup['all32_D_actual_first_use_verified'] and startup['P_archived_L05_candidate_disabled']
    assert equivalent['program_equivalence_verified'] and equivalent['all_model_provider_arguments_environment_and_code_identical']
    assert old['pass'] and old['all_DP_short_and_80K_semantic_verified'] and old['native_cache_and_SDMA_verified']
    cids = {node: item['container_id'] for node, item in runtime['runtime_containers'].items()}
    assert startup['runtime_CIDs'] == cids and startup['shared_run_id'] == runtime['shared_run_id'] == equivalent['restored_shared_run']
    sources = {p: sha(p) for p in [
        'config/C32-only-L04-restore/runtime.json',
        'results/sglang-C32-L04-restore-startup-v1.json',
        'results/sglang-C32-L04-program-equivalence-v1.json',
        'results/sglang-C32-L04-restore-pool-capacity-v1.json',
        'results/sglang-metadata-native-readiness-v1.json',
        'config/sglang-metadata-v1-stage-v1.json',
        'config/sglang-buffer-dedup-rollback-v1-stage-v1.json']}
    for p, digest in equivalent['source_sha256'].items():
        assert sha(p) == digest
    bridge_path = 'results/sglang-C32-L04-restored-program-readiness-v1.json'
    bridge = dict(project_id='glm53', framework='sglang', pass_=True)
    bridge['pass'] = bridge.pop('pass_')
    bridge.update(shared_run_id=runtime['shared_run_id'], runtime_CIDs=cids,
                  source_evidence_sha256=sources,
                  program_semantic_qualification_reused=True,
                  current_runtime_semantics_retested=False,
                  all32_D_current_actual_first_use_verified=True,
                  current_startup_native_SDK_and_idle_verified=True,
                  L05_P_candidate_enabled=False,
                  qualification_scope='Identical L04 program and image across controlled restart; previous all-DP short/two80K semantic qualification reused. This bridge does not claim new semantic requests or current-prefix payload reads.',
                  remaining_current_runtime_gates=['Fresh32realprefixwarmups and all55776key/384actualpayloadread gate', '128formal80000input/600output requests and count/cache/lifecycle/fullclock', 'Postrun SDK-worker/typed-SDMA/all32Ddevice/native errors/metrics audit'])
    fresh(bridge_path, bridge)
    settings = read('config/optimization-overlays/routing-metadata-v1/formal-C32-v2.json')
    for key in ['shared_run_id', 'shared_remote_root', 'runtime_containers', 'model_image', 'meta_container_id']:
        settings[key] = runtime[key]
    settings.update(run_id=RID, namespace=f'glm53-formal-{RID}-C32',
                    native_qualification_path=bridge_path, native_qualification_sha256=sha(bridge_path),
                    model_identity_helper='scripts/glm53_C32_L04_restore_model_identity_v1.py',
                    model_identity_sha256='f4394bab300ba689c3d5bd1a7a47c795a52b999c680e448154869f86e6665ab5',
                    prefix_transfer_helper='scripts/glm53_C32_L04_restore_prefix_transfer_v1.py',
                    prefix_transfer_sha256='04a53b4a9e4cde3572f5eb621bf51489f1251361460a47c0a3f44e2577ac9bce',
                    client_container_name='glm53-sglang-C32-L04-control-v3-20261017-43')
    settings['complete_before_start'] = [dict(framework='sglang', run_id='20261017T1990-sglang-formal-C32-L04-v2', native_capture_run='20261017T1992-sglang-L04-C32-native-v2', nodes=[34,35,36,41])]
    settings['C32_only_scope'] = 'Historical C16 certificate is a read-only harness ancestry check only; no new C16/C64 work or resource dependency.'
    relative = 'config/C32-only-L04-restore/formal-C32-control-v3.json'
    fresh(relative, settings)
    paired = load('scripts/glm53_sglang_metadata_launch_v1.py', LAUNCH)
    paired.validate_static(ROOT, relative, sha(relative))
    paired.validate_no_overlap(ROOT, settings)
    capture = load('scripts/glm53_sglang_C32_L04_native_capture_v1.py', CAPTURE)
    ast.parse(capture.render(capture.CPU_HOST, {'ARCHIVE':'/data/wio/inference-foundry/glm53/runs/'+POST,'FORMAL_RID':RID,'CONCURRENCY':32,'NEED_FORMAL':True}) + capture.codec.ENCODER)
    rendered = {}
    for ns, item in settings['runtime_containers'].items():
        node = int(ns)
        values = {'ARCHIVE':'/data/wio/inference-foundry/glm53/runs/'+POST,'FORMAL_RID':RID,'CONCURRENCY':32,'NEED_FORMAL':True,
                  'CID':item['container_id'],'NAME':item['name'],'IMAGE':settings['model_image'],'NODE':node,
                  'IDENTITY':capture.identity.HOST,'IDENTITY_ENV':{'CID':item['container_id'],'NAME':item['name'],'IMAGE':settings['model_image'],'NODE':node},
                  'STREAM':capture.STREAM,'SDK_SOURCE':capture.SDK_SOURCE,'PORT':30500 if node in [34,35] else 30600,'META':settings['meta_container_id']}
        code = capture.render(capture.HOST, values)
        ast.parse(code + capture.codec.ENCODER)
        rendered[ns] = hashlib.sha256(code.encode()).hexdigest()
    files = {
        'scripts/run_sglang_C32_L04_control_v3.py': wrapper('scripts/glm53_sglang_metadata_launch_v1.py', LAUNCH, f'assert {settings["concurrency"]}==32; m.execute({relative!r},{sha(relative)!r})'),
        'scripts/inspect_sglang_C32_L04_control_v3.py': wrapper('scripts/glm53_sglang_C32_L04_native_capture_v1.py', CAPTURE, f'm.execute({relative!r},{sha(relative)!r},{POST!r})')}
    for name, source in files.items():
        assert not (ROOT/name).exists()
        (ROOT/name).write_text(source)
    fresh('results/sglang-C32-L04-control-v3-frozen-review.json', dict(project_id='glm53', concurrency=32, run_id=RID, settings=relative, settings_sha256=sha(relative), launchers={p:sha(p) for p in files}, rendered_native_node_AST_SHA256=rendered, static_validation_pass=True, actual_resource_calls=0, model_requests=0, dispatched=False, current_runtime_semantics_retested=False))
    print(json.dumps(read('results/sglang-C32-L04-control-v3-frozen-review.json'), indent=2))

if __name__ == '__main__':
    prepare()
