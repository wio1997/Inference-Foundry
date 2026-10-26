"""CPU-only fresh seventh suffixes; exact tokenizer/prefix identity, no model or SDK."""
import ast
import hashlib
import importlib.util
import json
import os
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RID = '20261019T-L22-C32-seventh-suffix-inputs-v1'
REVIEW = 'results/C32-L22-seventh-input-preparation-review-v1.json'
SOCKET = '/Users/wio/work/sglang-glm53-pd/ssh-control-l20-foreground-20261019'

def load(name, relative, digest):
    path = ROOT / relative
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def plans():
    old_path = ROOT / 'scripts/prepare_sglang_C32_fifth_suffixes_v1.py'
    spec = importlib.util.spec_from_file_location('sixth_input_template', old_path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    old.RID = RID
    old.REMOTE = '/data/wio/inference-foundry/glm53/runs/' + RID
    generator = ROOT / 'scripts/glm53_C32_seventh_suffix_generator_v1.py'
    old.FILES['generator.py'] = (str(generator.relative_to(ROOT)), hashlib.sha256(generator.read_bytes()).hexdigest())
    mount = "'-v','/data/wio/models/GLM-5.3-w8a8/tokenizer.json:/tokenizer.json:ro',"
    assert old.HOST.count(mount) == 1
    old.HOST = old.HOST.replace(mount, "'-v','/data/wio/inference-foundry/glm53/runs/20261017T2056-sglang-C32-fifth-suffix-inputs-v1/artifacts:/fifth:ro',\n " + mount)
    old.HOST = old.HOST.replace("'-v','/data/wio/models/GLM-5.3-w8a8/tokenizer.json:/tokenizer.json:ro',", "'-v','/data/wio/inference-foundry/glm53/runs/20261019T-L21-C32-sixth-suffix-inputs-v1/artifacts:/sixth:ro', '-v','/data/wio/models/GLM-5.3-w8a8/tokenizer.json:/tokenizer.json:ro',")
    old.HOST = old.HOST.replace('glm53-C32-fifth-suffix-inputs-v1-20261017-43', 'glm53-C32-L22-seventh-inputs-v1-20261019-43')
    old.HOST = old.HOST.replace("IMAGE,'900','python3'", "IMAGE,'180','python3'")
    old.HOST = old.HOST.replace("'--tokenizer','/tokenizer.json']", "'--tokenizer','/tokenizer.json','--fifth','/fifth','--sixth','/sixth']")
    old.HOST = old.HOST.replace('timeout=960)', 'timeout=210)')
    code = old.render()
    ast.parse(code)
    assert "'180', 'python3'" in code or "'180','python3'" in code
    return code

def main():
    import sys
    code = plans()
    raw = (ROOT / REVIEW).read_bytes()
    review = json.loads(raw)
    assert len(sys.argv) == 2 and hashlib.sha256(raw).hexdigest() == sys.argv[1]
    assert review['decision'] == 'APPROVED_CPU_ONLY_SEVENTH_SUFFIX_PREPARATION'
    assert review['launcher_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert review['host_sha256'] == hashlib.sha256(code.encode()).hexdigest()
    inv = load('sixth_inventory', 'scripts/inventory_readonly.py', 'e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777')
    inv.SSH = list(inv.SSH)
    inv.SSH[2] = SOCKET
    run = ROOT / 'runs' / RID
    run.mkdir(exist_ok=False)
    (run / 'execution-once.lock').write_text(str(os.getpid()))
    pod, _ = inv.NODES[43]
    mapping = inv.invoke(['kubectl get pod -n default ' + shlex.quote(pod) + " -o jsonpath='{.status.hostIP}'"], timeout=20)
    (run / 'mapping.json').write_text(json.dumps(mapping) + '\n')
    assert mapping['rc'] == 0 and mapping['stdout'] == '172.16.12.43'
    command = shlex.join(['kubectl', 'exec', '-i', '-n', 'default', pod, '-c', 'hostctl', '--', 'nsenter', '-t', '1', '-m', '--', 'python3', '-'])
    wire = inv.invoke([command], code, timeout=230)
    (run / 'node-43.json').write_text(json.dumps(wire) + '\n')
    receipt = {'project_id': 'glm53', 'run_id': RID, 'rc': wire['rc'], 'model_requests': 0, 'SDK_init_calls': 0, 'NPU_devices_exposed': False}
    if wire['rc'] == 0:
        data = json.loads(wire['stdout'])
        receipt['generator_rc'] = data['rc']
        if data['rc'] == 0:
            proof = data['input_proof']
            assert proof['manifest']['fresh_suffix_lcp_exact']
            assert all(row['variant'] == 7 and row['lcp_against_previous_six'] == [74400]*6 for row in proof['manifest']['rows'])
            (run / 'manifest-proof.json').write_text(json.dumps(proof, indent=2) + '\n')
            receipt['manifest_sha256'] = proof['manifest_sha256']
    (run / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))

if __name__ == '__main__':
    main()
