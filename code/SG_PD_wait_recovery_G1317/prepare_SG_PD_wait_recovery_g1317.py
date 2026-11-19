"""Prepare the same fixed P/D model for the 32K KV-wait recovery epoch."""
import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / 'sg_pair_DSPARK_Q_policy_preparation_g1178'
OUTPUT = ROOT / 'sg_pd_wait_recovery_g1317'
LIFETIME_SECONDS = 108000
DECODE_WAIT_SECONDS = 14400


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_text(value):
    return json.dumps(value, indent=2, ensure_ascii=False) + '\n'


def entries(source):
    manifest = source / 'deployment_source_sha256.json'
    pins = json.loads(manifest.read_bytes())
    assert len(pins) == 58
    for name, digest in pins.items():
        path = source / name
        assert path.resolve().is_relative_to(source.resolve())
        raw = path.read_bytes()
        assert sha(raw) == digest, name
        yield name, raw


def revision(name, raw, role):
    if name == 'node_controller.py':
        code = raw.decode().replace('sg_pd_g1178', 'sg_pd_g1317').replace("'-g1178'", "'-g1317'")
        changes = {'generation=1178':'generation=1317', 'LIFETIME=72000':f'LIFETIME={LIFETIME_SECONDS}',
                   'exec sleep 72100':f'exec sleep {LIFETIME_SECONDS + 100}',
                   "'72000s PD lifecycle bound'":"f'{LIFETIME}s PD lifecycle bound'"}
        for old, new in changes.items():
            assert code.count(old) == 1, old
            code = code.replace(old, new)
        return code.encode()
    if name == 'run_pd.sh':
        return raw.replace(b'/task/sg_pd_g1178', b'/task/sg_pd_g1317')
    if name == 'candidate.json':
        config = json.loads(raw)
        assert config['role'] == role
        env = config['environment']
        assert env['SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT'] == env['SGLANG_DISAGGREGATION_WAITING_TIMEOUT'] == '3600'
        allowed_env = {key:value.replace('sg_pd_g1178', 'sg_pd_g1317') for key,value in env.items()}
        if role == 'd':
            allowed_env['SGLANG_DISAGGREGATION_WAITING_TIMEOUT'] = str(DECODE_WAIT_SECONDS)
        config['environment'] = allowed_env
        return json_text(config).encode()
    return raw


def main():
    assert not OUTPUT.exists()
    prepared = {}
    for role in ('p','d'):
        source = BASELINE / role
        original = dict(entries(source))
        after = {name:revision(name,raw,role) for name,raw in original.items()}
        changed = {name for name in original if original[name] != after[name]}
        assert changed == {'node_controller.py','run_pd.sh','candidate.json'}
        for name, raw in after.items():
            if name.endswith('.py'):
                ast.parse(raw, filename=name)
        before_cfg, cfg = (json.loads(items['candidate.json']) for items in (original,after))
        assert before_cfg['argv'] == cfg['argv'], 'Inference arguments must remain identical'
        assert all(before_cfg[name] == cfg[name] for name in before_cfg if name != 'environment')
        for key, prior in before_cfg['environment'].items():
            expected = str(DECODE_WAIT_SECONDS) if role=='d' and key=='SGLANG_DISAGGREGATION_WAITING_TIMEOUT' else prior.replace('sg_pd_g1178','sg_pd_g1317')
            assert cfg['environment'][key]==expected
        prepared[role] = (after,changed)

    report = {'epoch':1317,'baseline_epoch':1178,'lifetime_seconds':LIFETIME_SECONDS,
              'D_KV_wait_after_metadata_seconds':DECODE_WAIT_SECONDS,'P_bootstrap_seconds':3600,
              'P_and_D_model_arg_vectors_byte_equivalent':True,'roles':{},
              'scope':'Prepared for full original 32K recovery. Actual release, deployment, 8+8 readiness, P/D replay and five independent warmups pending.'}
    for role, (files, changed) in prepared.items():
        dest = OUTPUT / role
        dest.mkdir(parents=True, exist_ok=False)
        for name, raw in files.items():
            path = dest / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        manifest = dest / 'deployment_source_sha256.json'
        manifest.write_text(json_text({name:sha(raw) for name,raw in files.items()}))
        report['roles'][role] = {'verified_source_files':len(files),'changed_files':sorted(changed),
                                 'deployment_manifest_SHA256':sha(manifest.read_bytes())}
    (OUTPUT/'preparation_manifest.json').write_text(json_text(report))
    print(json_text(report))


if __name__ == '__main__':
    main()
