"""Exact-image stub/source and fresh-interpreter hook checks; no SDK/model calls."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    root = Path('/run/glm53')
    assert not any(Path('/dev').glob('davinci*'))
    environment = os.environ.copy()
    for key in ('GLM53_DEEPEP_METADATA_BATCH', 'GLM53_BUFFER_DEDUP_VALIDATE', 'GLM53_NODE'):
        environment.pop(key, None)
    result = {'project_id': 'glm53', 'python_version': sys.version, 'model_requests': 0, 'SDK_init_calls': 0, 'NPU_devices': False, 'cases': []}
    commands = [('native_AST_properties', [sys.executable, str(root / 'python/glm53_buffer_dedup_CPU_fixture_v1.py'), str(root / 'native-source'), str(root / 'fixture-result.json')], environment, 0)]
    checks = {
        'P_ready': ({'GLM53_BUFFER_DEDUP_VALIDATE': '1', 'GLM53_NODE': '34'}, "import sys;assert sum(type(x).__module__=='glm53_buffer_dedup_import_hook_v1' for x in sys.meta_path)==1;assert not any(type(x).__module__=='glm53_deepep_metadata_import_hook_v1' for x in sys.meta_path)", 0),
        'D_metadata_preserved_P_noop': ({'GLM53_DEEPEP_METADATA_BATCH': '1', 'GLM53_NODE': '36'}, "import sys;assert sum(type(x).__module__=='glm53_deepep_metadata_import_hook_v1' for x in sys.meta_path)==1;assert not any(type(x).__module__=='glm53_buffer_dedup_import_hook_v1' for x in sys.meta_path)", 0),
        'P_hook_disabled': ({'GLM53_NODE': '34'}, "import sys;assert not any(type(x).__module__=='glm53_buffer_dedup_import_hook_v1' for x in sys.meta_path)", 0),
        'wrong_role_fail_closed': ({'GLM53_BUFFER_DEDUP_VALIDATE': '1', 'GLM53_NODE': '36'}, "raise AssertionError('site guard should exit86 before this code')", 86),
    }
    for name, (updates, code, expected) in checks.items():
        commands.append((name, [sys.executable, '-c', code], environment | updates, expected))
    tampered = root / 'tampered-python'
    shutil.copytree(root / 'python', tampered)
    with (tampered / 'glm53_buffer_dedup_import_hook_v1.py').open('a') as file:
        file.write('\n# intentional source-guard negative fixture\n')
    commands.append(('hook_SHA_fail_closed', [sys.executable, '-c', "raise AssertionError('badSHA should exit86 before this code')"], environment | {'PYTHONPATH': str(tampered), 'GLM53_BUFFER_DEDUP_VALIDATE': '1', 'GLM53_NODE': '34'}, 86))
    for name, command, env, expected in commands:
        process = subprocess.run(command, env=env, capture_output=True, text=True, timeout=40)
        (root / (name + '.stdout.txt')).write_text(process.stdout)
        (root / (name + '.stderr.txt')).write_text(process.stderr)
        row = {'case': name, 'expected_rc': expected, 'actual_rc': process.returncode, 'pass': process.returncode == expected}
        result['cases'].append(row)
        if not row['pass']:
            break
    result['pass'] = len(result['cases']) == len(commands) and all(row['pass'] for row in result['cases'])
    result['property_result'] = json.loads((root / 'fixture-result.json').read_text()) if (root / 'fixture-result.json').exists() else None
    (root / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print('GLM53_BUFFER_DEDUP_IMAGE_CPU=' + json.dumps(result), flush=True)
    sys.exit(0 if result['pass'] else 1)


if __name__ == '__main__':
    main()
