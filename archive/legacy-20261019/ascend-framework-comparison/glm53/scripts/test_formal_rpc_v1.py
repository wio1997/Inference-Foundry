"""Real local JSON stdio/subprocess fixture. No Docker, SSH, SDK or model calls."""
import hashlib,importlib.util,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];source=ROOT/'scripts/glm53_formal_rpc_v1.py';s=importlib.util.spec_from_file_location('rpc',source);module=importlib.util.module_from_spec(s);s.loader.exec_module(module)
CHILD=r'''import sys,json,gzip,base64,hashlib
expected=json.loads(sys.argv[1]);scenario=sys.argv[2]
request=dict(expected)
if scenario=='bad_manifest':
 request['manifest_sha256']='b'*64;print('GLM53_FORMAL_NEED_GATE='+json.dumps(request),flush=True);raise SystemExit(2)
print('GLM53_FORMAL_NEED_GATE='+json.dumps(request),flush=True)
response=json.loads(sys.stdin.readline());report={k:expected[k] for k in ['project_id','framework','namespace','concurrency']};report.update(baseline_accepted=response.get('pass') is True,backend_HTTP_attempts=0,fixture_only=True)
raw=json.dumps(report).encode();state=dict(report);done={'state':state,'formal_report_sha256':hashlib.sha256(raw).hexdigest(),'formal_report_gzip_base64':base64.b64encode(gzip.compress(raw,mtime=0)).decode(),'artifacts':{}}
print('GLM53_FORMAL_DONE='+json.dumps(done),flush=True)
raise SystemExit(0 if report['baseline_accepted'] else 1)
'''
def main():
 cases=[]
 for scenario in ['success','gate_callback_failure','bad_manifest']:
  with tempfile.TemporaryDirectory(prefix='glm53-rpc-fixture-') as tmp:
   root=Path(tmp);expected={'project_id':'glm53','framework':'vllm','concurrency':16,'namespace':'glm53-formal-rpc-fixture-C16','manifest_sha256':'a'*64};calls=[]
   def gate(request):
    calls.append(request)
    if scenario=='gate_callback_failure':raise RuntimeError('injected fixture callback failure')
    return dict(request,prefix_count=16,all_expected_keys_present=True,actual_payload_samples_read=True,close_rc=0,pass_=True)|{'pass':True}
   error=None
   try:state=module.rpc([sys.executable,'-c',CHILD,json.dumps(expected),scenario],root,expected,gate)
   except AssertionError as err:error=repr(err);state=json.loads((root/'controller-rpc-state.json').read_text())
   assert len(calls)==(0 if scenario=='bad_manifest' else 1)
   if scenario=='bad_manifest':assert error is not None and not state['done_received']
   else:
    assert state['done_received'] and state['backend_HTTP_attempts']==0
    assert state['client_baseline_accepted']==(scenario=='success') and state['gate_pass']==(scenario=='success')
    assert state['connection_rc']==(0 if scenario=='success' else 1)
    assert hashlib.sha256((root/'formal-result.remote.json').read_bytes()).hexdigest()==state['report_sha256']
   cases.append({'scenario':scenario,'callback_calls':len(calls),'done_received':state['done_received'],'error_expected':scenario=='bad_manifest','fixture_only':True})
 result={'project_id':'glm53','pass':True,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'scope':'Actual local subprocess/stdin/stdout/hash framing; no SSH/Docker/native AIS/SDK/model qualification','cases':cases,'network_calls':0,'model_requests':0,'SDK_init_calls':0}
 (ROOT/'results/formal-rpc-local-fixture-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
