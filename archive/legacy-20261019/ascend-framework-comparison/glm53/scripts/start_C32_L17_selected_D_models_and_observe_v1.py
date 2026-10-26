"""Start the independently qualified actual D pools once; no external inference.

The review must pin the actual capacity file and every rendered host plan.
Cleanup and mixed identity plans are frozen for later primary use, not executed.
"""
import ast
import concurrent.futures
import hashlib
import importlib.util
import json
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
RID = '20261019T-L17-D-models-and-observe-v1'
RUN = ROOT / 'runs' / RID
REVIEW = ROOT / 'results/C32-L17-D-model-start-actual-review-v1.json'
CAPACITY = ROOT / 'results/sglang-C32-L17-selected-D-pool-capacity-v2.json'
LIBRARY = ROOT / 'scripts/glm53_C32_L17_D_transition_library_v2.py'
LIBRARY_SHA = 'da4bbcab4ddf1dd4670c45db78a20aace39f12e1429e73aa5da9d27d36b7b8ce'
IDENTITY = ROOT / 'scripts/glm53_C32_L17_mixed_model_identity_v1.py'
IDENTITY_SHA = 'ec8a167da59504dcfcd9dec1347960e93fbab2764d9545f96a6f80768aff1526'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path, expected, name):
    assert sha(path) == expected
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plan():
    lib = load(LIBRARY, LIBRARY_SHA, 'model_start_library')
    identity = load(IDENTITY, IDENTITY_SHA, 'model_start_identity')
    capacity = json.loads(CAPACITY.read_text())
    assert capacity['pass'] is True and capacity['model_started'] is False
    assert capacity['selected_P_variant'] == capacity['selected_D_variant'] == 'candidate'
    p_capacity = json.loads((ROOT / lib.CAPACITY).read_text())
    assert sha(ROOT/lib.CAPACITY) == lib.CAPACITY_SHA
    # The D auditor deliberately stores D rows only. Lifecycle cleanup also
    # needs the retained P identities; derive this view without editing either
    # immutable capacity source.
    providers = {str(n):p_capacity['providers'][str(n)] for n in (34,35)}
    providers.update(capacity['providers'])
    capacity = {**capacity, 'providers':providers,
                'retained_P_capacity_sha256':lib.CAPACITY_SHA}
    codes = {name: code for name, code in lib.health_plans().items()
             if name in ('health-34', 'health-35')}
    codes['meta-identity'] = (
        'import subprocess,json\n'
        'r=subprocess.run('+repr(['docker', 'inspect', lib.META_CID])+',capture_output=True,text=True,timeout=20)\n'
        'assert r.returncode==0,r.stderr\n'
        'd=json.loads(r.stdout)[0]\n'
        'assert d["Id"]=='+repr(lib.META_CID)+' and d["Image"]=='+repr(lib.IMAGE)+' and d["Name"]=='+repr('/'+lib.META_NAME)+'\n'
        'assert d["Config"]["Labels"]["project_id"]=="glm53" and d["Config"]["Labels"]["run_id"]=='+repr(lib.OLD_D_RID)+' and d["State"]["Running"]\n'
        'print(json.dumps({"container_id":d["Id"],"running":True,"model_requests":0}))\n'
    )
    for node in (36, 41):
        start = lib.model_start_plan('candidate', node, capacity)
        tree = ast.parse(start)
        stop = next(i for i, item in enumerate(tree.body)
                    if isinstance(item, ast.With)
                    and any(isinstance(x, ast.Constant) and x.value == 'model-start-controller.once'
                            for x in ast.walk(item)))
        prefix = ast.Module(body=tree.body[:stop], type_ignores=[])
        codes['prestart-'+str(node)] = ast.unparse(prefix) + '\nprint(json.dumps({"node":NODE,"container_id":d["Id"],"prestart_pass":True,"model_requests":0}))\n'
        codes['start-'+str(node)] = start + '''
import time
deadline=time.monotonic()+15
while True:
    observed=json.loads((root/'supervisor-status.json').read_text())
    if observed.get('phase')=='MODEL_RUNNING' and observed.get('model_started') and observed.get('model_pid'):
        assert 'model_exit_code' not in observed,observed
        break
    assert observed.get('phase') not in ('STOPPED','FAILED'),observed
    assert time.monotonic()<deadline,observed
    time.sleep(.1)
print('GLM53_L17_MODEL_PARENT='+json.dumps({'node':NODE,'container_id':d['Id'],'supervisor':observed,'external_model_requests':0}))
'''
        codes['observe-'+str(node)] = lib.observe_plan('candidate', node, capacity)
        codes['cleanup-'+str(node)] = lib.cleanup_plan('candidate', node, capacity)
    for node in (34, 35, 36, 41):
        codes['identity-'+str(node)] = identity.render(node, p_capacity, capacity, 'candidate')
    for name, code in codes.items():
        checked = lib.analyze_code(name, code)
        assert checked['ast_parse'] and checked['compile'] and not checked['unbound_globals'], (name, checked)
    return lib, capacity, codes


def main():
    assert REVIEW.is_file(), 'Actual capacity and dynamic-plan primary review required'
    review = json.loads(REVIEW.read_text())
    assert review['review_status'] == 'APPROVED_FOR_ACTUAL_D_MODEL_START_AND_OBSERVE_ONLY'
    assert review['launcher_sha256'] == sha(pathlib.Path(__file__))
    assert review['capacity_sha256'] == sha(CAPACITY)
    lib, capacity, codes = plan()
    assert review['host_code_sha256'] == {key: hashlib.sha256(value.encode()).hexdigest() for key, value in codes.items()}
    assert review['model_requests_authorized'] == 0
    transport = lib.transport()
    RUN.mkdir(exist_ok=False)
    (RUN/'execution-once.lock').write_text(str(__import__('os').getpid()))
    retained_p = json.loads((ROOT/lib.CAPACITY).read_text())['providers']
    runtime_cids = {str(n):retained_p[str(n)]['container_id'] for n in (34,35)}
    runtime_cids.update({str(n):capacity['providers'][str(n)]['container_id'] for n in (36,41)})
    receipt = {'project_id':'glm53','phase':'PRECHECK','pass':False,'capacity_sha256':sha(CAPACITY),
               'review_sha256':sha(REVIEW),'launcher_sha256':sha(pathlib.Path(__file__)),
               'precheck':{},'starts':{},'observe':{},'max_SSH':2,'external_model_requests':0,
               'cleanup_or_rollback_executed':False,'runtime_CIDs':runtime_cids}
    def save(name, value):
        (RUN/name).write_text(json.dumps(value,indent=2)+'\n')
        return value
    def checkpoint():
        save('receipt.json',receipt)
    def batch(jobs, timeout):
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(transport.invoke,node,codes[key],timeout):(node,key) for node,key in jobs}
            for future in concurrent.futures.as_completed(futures):
                node,key = futures[future]
                try:
                    value=future.result()
                except Exception as error:
                    value={'error':repr(error)}
                save(key+'.json',value)
                results[key]=value
                print(key,value.get('rc',value.get('error')),flush=True)
        return results
    try:
        checkpoint()
        jobs=[(34,'health-34'),(35,'health-35'),(34,'meta-identity'),(36,'prestart-36'),(41,'prestart-41')]
        results=batch(jobs,150)
        receipt['precheck']={key:value.get('rc',value.get('error')) for key,value in results.items()}
        checkpoint()
        assert len(results)==5 and all(value.get('rc')==0 for value in results.values())
        receipt['phase']='STARTING_D_PARENTS';checkpoint()
        results=batch([(n,'start-'+str(n)) for n in (36,41)],90)
        receipt['starts']={key:value.get('rc',value.get('error')) for key,value in results.items()}
        checkpoint()
        assert len(results)==2 and all(value.get('rc')==0 for value in results.values())
        receipt['phase']='OBSERVING_NATIVE_D_STARTUP';checkpoint()
        results=batch([(n,'observe-'+str(n)) for n in (36,41)],1950)
        receipt['observe']={key:value.get('rc',value.get('error')) for key,value in results.items()}
        checkpoint()
        assert len(results)==2 and all(value.get('rc')==0 for value in results.values())
        summaries={}
        for key,value in results.items():
            marker='GLM53_L17_D_OBSERVE='
            summaries[key]=json.loads(value['stdout'].split(marker,1)[1].strip().splitlines()[0])
            assert summaries[key]['startup_pass'] is True and summaries[key]['formal_ready'] is False
        receipt['native_startup_summaries']=summaries
        receipt['phase']='STARTUP_OBSERVED_AWAITING_INDEPENDENT_CAPTURE_AND_FUNCTIONAL';receipt['pass']=True
    except BaseException as error:
        receipt['phase']='FAILED_OR_PARTIAL_STATE_PRESERVED';receipt['error']=repr(error)
        raise
    finally:
        checkpoint()
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
