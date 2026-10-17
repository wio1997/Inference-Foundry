"""Local real-child/file/stdin handoff fixture; zero network, SDK and model calls."""
import base64, contextlib, gzip, hashlib, importlib.util, io, json, sys, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'scripts/glm53_formal_host_controller_v2.py'
spec=importlib.util.spec_from_file_location('host_controller',path)
controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)

def main():
    cases=[]
    for scenario in ['cache_gate_then_success','child_failure_before_gate']:
        with tempfile.TemporaryDirectory(prefix='glm53-formal-control-') as directory:
            root=Path(directory);(root/'output').mkdir()
            settings={'framework':'sglang','concurrency':16,'namespace':'glm53-formal-local-IPC-C16','input_manifest_sha256':'0'*64}
            child=r'''import json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]);scenario=sys.argv[2];settings=json.loads(sys.argv[3]);output=root/'output'
if scenario=='cache_gate_then_success':
 request={'project_id':'glm53','framework':'sglang','namespace':settings['namespace'],'concurrency':16,'manifest_sha256':settings['input_manifest_sha256']}
 (output/'formal-progress.json').write_text(json.dumps({'phase':'WARMUP','warmup':[{'pass':True} for _ in range(16)]}))
 (output/'need-cache-gate.json').write_text(json.dumps(request))
 deadline=time.monotonic()+5
 while not (output/'cache-gate.json').exists():
  assert time.monotonic()<deadline;time.sleep(.01)
 response=json.loads((output/'cache-gate.json').read_text());assert response['namespace']==settings['namespace'] and response['pass']
 report={'phase':'COMPLETE','baseline_accepted':True,'backend_HTTP_attempts':0,'fixture_only':True}
else:report={'phase':'FAILED','baseline_accepted':False,'backend_HTTP_attempts':0,'fixture_only':True}
(output/'formal-result.json').write_text(json.dumps(report))
raise SystemExit(0 if report['baseline_accepted'] else 1)
'''
            response={'project_id':'glm53','framework':'sglang','namespace':settings['namespace'],'pass':True,'fixture_only':True}
            stdin=sys.stdin;stdout=io.StringIO()
            try:
                sys.stdin=io.StringIO(json.dumps(response)+'\n')
                with contextlib.redirect_stdout(stdout):
                    rc=controller.control(root,[sys.executable,'-c',child,str(root),scenario,json.dumps(settings)],settings)
            finally:sys.stdin=stdin
            lines=stdout.getvalue().splitlines();done=json.loads(next(line.split('=',1)[1] for line in lines if line.startswith('GLM53_FORMAL_DONE=')))
            raw=gzip.decompress(base64.b64decode(done['formal_report_gzip_base64']))
            assert hashlib.sha256(raw).hexdigest()==done['formal_report_sha256']
            state=done['state'];assert state['backend_HTTP_attempts']==0
            expected=scenario=='cache_gate_then_success'
            assert (rc==0)==expected and state['baseline_accepted']==expected and state['cache_gate_received']==expected
            assert any(line.startswith('GLM53_FORMAL_NEED_GATE=') for line in lines)==expected
            assert any(line.startswith('GLM53_FORMAL_PROGRESS=') for line in lines)==expected
            cases.append({'scenario':scenario,'rc':rc,'cache_gate_received':state['cache_gate_received'],'baseline_flag_is_fixture_only':state['baseline_accepted'],'backend_HTTP_attempts':0})
    result={'project_id':'glm53','pass':True,'scope':'Actual local subprocess + files + JSON stdio; excludes Docker/SSH/native AIS/model/SDK qualification',
            'controller_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'cases':cases,'network_calls':0,'model_requests':0,'SDK_init_calls':0}
    (ROOT/'results/formal-host-controller-ipc-fixture-v2.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
