"""Offline deployment record checks; does not start models or touch NPUs."""
from pathlib import Path
import ast
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]


def main():
    entries=json.loads((ROOT/'evidence/deployment_provenance.json').read_text())
    for item in entries:
        p=ROOT/item['artifact'];assert hashlib.sha256(p.read_bytes()).hexdigest()==item['published_sha256'],p
    pinned=json.loads((ROOT/'metadata/pinned_files.json').read_text())
    files={x['Path']:x for x in pinned['files']}
    assert len(files)==95 and sum(x['Size'] for x in files.values())==528805292902
    for port,role in [(2203,'p'),(2204,'d')]:
        base=ROOT/f'evidence/full_model_G65/{port}'
        weights=json.loads((base/'verified_weight_snapshot.json').read_text())
        assert weights['phase']=='complete' and not weights['failed']
        assert set(weights['verified'])==set(files)
        for name,value in weights['verified'].items():
            assert value['sha256']==files[name]['Sha256']
            assert value['bytes']==files[name]['Size']
            assert value['revision']==files[name]['Revision']
        state=json.loads((base/'state.json').read_text())
        assert state['role']==role and state['phase']=='failed_released'
        assert state['cleanup']['all8_process_lists_clear']
        assert not state['cleanup']['container_state']['Running']
        log=(base/'server.log').read_text()
        assert 'validate_deepseek_v41_features' in log
        assert 'requires static verify, Mooncake, NIXL' in log
        argv=json.loads((base/'container_argv.json').read_text())
        assert '--read-only' in argv and any(x.endswith(':/task:ro') for x in argv)
        assert any(x.endswith(':/models/ds41:ro') for x in argv)
    for role in ['p','d']:
        c=json.loads((ROOT/f'code/full_model_candidates/sg_pd_g68/candidate_{role}.json').read_text())
        a=c['argv'];value=lambda flag:a[a.index(flag)+1]
        assert value('--tp-size')==value('--ep-size')=='8'
        assert value('--pp-size')==value('--attn-cp-size')==value('--dcp-size')=='1'
        assert value('--speculative-algorithm')=='DSPARK' and value('--speculative-num-draft-tokens')=='6'
        assert value('--kv-cache-dtype')==value('--dtype')=='bfloat16'
        assert value('--context-length')=='263296' and value('--max-running-requests')=='16'
        assert c['environment']['SGLANG_RAGGED_VERIFY_MODE']=='static'
        assert c['environment']['ASCEND_MF_TRANSFER_PROTOCOL']=='device_rdma'
        if role=='d':assert json.loads(value('--cuda-graph-config'))['decode']['backend']=='full'
    for path in (ROOT/'code/full_model_candidates').rglob('*.py'):ast.parse(path.read_text(),filename=str(path))
    for port in [2203,2204]:
        base=ROOT/f'evidence/full_model_G69/{port}'
        state=json.loads((base/'state.json').read_text())
        assert state['phase']=='failed_released' and state['cleanup']['all8_process_lists_clear']
        assert not state['cleanup']['container_state']['Running']
        log=(base/'server.log').read_text()
        assert 'FileExistsError' in log and 'multiprocessing/spawn.py' in log and 'server_help.txt' in log
    entry=ast.parse((ROOT/'code/full_model_candidates/sg_pd_g70/launch_pd.py').read_text())
    assert isinstance(entry.body[-1],ast.If)
    assert ast.unparse(entry.body[-1].test)=="__name__ == '__main__'"
    assert all(isinstance(x,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.If)) for x in entry.body)
    regression=json.loads((ROOT/'evidence/full_model_G70/spawn_import_regression.json').read_text())
    assert regression['actual_runpy_spawn_import_name']=='__mp_main__'
    assert regression['entry_imported_without_candidate_file'] and regression['no_file_writes']
    sdk=json.loads((ROOT/'evidence/fixed_image_SDK_binary_inventory.json').read_text())['inventory']
    assert sdk['memcache-hybrid']['version']==sdk['memfabric-hybrid']['version']=='1.3.0'
    print(json.dumps({'deployment_provenance_files':len(entries),'pinned_files_per_host':95,'bytes_per_host':528805292902,
                      'G65_failure_and_cleanup_retained':True,'SDK_binary_inventory_retained':True,
                      'full_model_PD_or_performance_acceptance':False}))


if __name__=='__main__':main()
