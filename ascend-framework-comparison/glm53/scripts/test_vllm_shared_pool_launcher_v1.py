"""Local HOST/inner syntax, Docker argv and main cleanup fixture; no resources."""
import ast,contextlib,hashlib,importlib.util,io,json,shutil,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'scripts/start_vllm_shared_pool_v1.py'
spec=importlib.util.spec_from_file_location('pool',P);pool=importlib.util.module_from_spec(spec);spec.loader.exec_module(pool)

def main():
    generated=[]
    def fake(node,code,timeout=120):
        ast.parse(code);generated.append((node,code))
        assert 'sglang' not in code
        if 'GLM53_SHARED_HARDWARE=' in code:
            strings=[n.value for n in ast.walk(ast.parse(code)) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
            ast.parse(next(v for v in strings if 'GLM53_SHARED_HARDWARE=' in v))
            assert 'export PYTHONPATH=/opt/glm53/sdk13:/opt/glm53/source/python:${PYTHONPATH:-}' in code
        for node_ast in ast.walk(ast.parse(code)):
            if not isinstance(node_ast,ast.Call) or not isinstance(node_ast.func,ast.Attribute) or node_ast.func.attr!='run' or not node_ast.args or not isinstance(node_ast.args[0],ast.List):continue
            try:args=ast.literal_eval(node_ast.args[0])
            except (ValueError,TypeError):continue
            if args[:2]!=['docker','run']:continue
            assert '--privileged' not in args and args.count('--device') in [0,19]
            assert all('MMC_LOCAL_CONFIG_PATH' not in arg for arg in args)
            assert 'export PYTHONPATH=/opt/glm53/sdk13:/opt/glm53/source/python:${PYTHONPATH:-}' in args[-1]
        return {'rc':0,'stdout':json.dumps({'ready':True}),'stderr':''}
    pool.invoke=fake
    for node in [30,31,43,45]:
        pool.precheck(node);pool.start(node);pool.native_preflight(node);pool.snapshot_stop(node)
    pool.start(30,True);pool.snapshot_stop(30,True);pool.live_meta_snapshot()
    assert len(generated)==19
    cases=[]
    for scenario in ['success_keeps_pool_no_model','native_preflight_failure_closes_all_then_Meta']:
        with tempfile.TemporaryDirectory(prefix='glm53-vllm-pool-fixture-') as tmp:
            fake_root=Path(tmp)
            for name in ['runs/20261017T1746-vllm-shared-stage-v2/receipt.json','runs/20261017T1745-stop-vllm-v4-for-shared-v1/receipt.json','results/sdk-postinit-signal-qualification-v1.json']:
                (fake_root/name).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,fake_root/name)
            name='runs/20261017T1751-vllm-empty-memory-scan-v1/receipt.json';(fake_root/name).parent.mkdir(parents=True);(fake_root/name).write_text(json.dumps({'nodes':{n:0 for n in [30,31,43,45]},'fixture_only':True}))
            pool.b.ROOT=fake_root;pool.RUN=fake_root/'run'
            cleanup=[]
            def stop(n,meta=False):cleanup.append((n,meta));return {'rc':0,'stdout':'{}','stderr':''}
            pool.snapshot_stop=stop
            original=pool.native_preflight
            if 'failure' in scenario:pool.native_preflight=lambda n:{'rc':1 if n==43 else 0,'stdout':'{}','stderr':'injected fixture failure' if n==43 else ''}
            with contextlib.redirect_stdout(io.StringIO()):rc=pool.main()
            pool.native_preflight=original
            receipt=json.loads((pool.RUN/'receipt.json').read_text());success='failure' not in scenario
            assert receipt['pool_kept_running']==success and receipt['model_requests']==0 and rc==(0 if success else 1)
            assert len(cleanup)==(0 if success else 5)
            if not success:assert cleanup[-1]==(30,True) and {n for n,meta in cleanup if not meta}=={30,31,43,45}
            cases.append({'scenario':scenario,'kept_running_fixture_only':success,'cleanup_calls':len(cleanup),'model_requests':0})
    result={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'generated_HOST_AST':19,'inner_AST':4,'cases':cases,'resource_actions':0,'pass':True,'scope':'Local generated code/main mock; no native SDK, SSH, Docker or model qualification'}
    (ROOT/'results/vllm-shared-pool-launcher-local-review-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
