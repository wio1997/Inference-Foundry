"""One long host-controller connection plus one gate callback; no shell/network defaults.

A reviewed ZCode launcher supplies an exact command and callback. Errors preserve
remote children. This helper neither retries nor kills a running native client.
"""
import base64,gzip,hashlib,json,os,subprocess,time
from pathlib import Path

FRAME_CAP=128*1024**2
REPORT_CAP=64*1024**2

def rpc(command,root,expected,on_gate):
    root=Path(root);state={'project_id':'glm53','framework':expected['framework'],'namespace':expected['namespace'],'gate_requests':0,'done_received':False,'controller_connection_pid':None,'connection_rc':None}
    process=None;done=None
    try:
        with (root/'controller-stdout.txt').open('x') as stdout,(root/'controller-stderr.txt').open('x') as stderr:
            process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,text=True,encoding='utf-8',start_new_session=True)
            state['controller_connection_pid']=process.pid
            while True:
                line=process.stdout.readline(FRAME_CAP+1)
                if not line:break
                stdout.write(line);stdout.flush()
                assert len(line)<=FRAME_CAP and line.endswith('\n'),'Incomplete or oversize controller frame; preserve remote client'
                if line.startswith('GLM53_FORMAL_NEED_GATE='):
                    request=json.loads(line.split('=',1)[1]);assert state['gate_requests']==0
                    for key in ['project_id','framework','namespace','concurrency','manifest_sha256']:assert request[key]==expected[key]
                    state['gate_requests']+=1
                    try:response=on_gate(request)
                    except Exception as error:
                        response={'project_id':'glm53','framework':expected['framework'],'namespace':expected['namespace'],'prefix_count':expected['concurrency'],'pass':False,'all_expected_keys_present':False,'actual_payload_samples_read':False,'close_rc':None,'gate_controller_error':repr(error)}
                    for key in ['project_id','framework','namespace']:assert response[key]==expected[key]
                    state['gate_pass']=response.get('pass') is True
                    process.stdin.write(json.dumps(response)+'\n');process.stdin.flush()
                elif line.startswith('GLM53_FORMAL_PROGRESS='):
                    progress=json.loads(line.split('=',1)[1])
                    for key in ['project_id','framework','namespace','concurrency']:assert progress[key]==expected[key]
                    assert 0<=progress['warmup_passes']<=progress['completed_real_prefix_warmups']<=expected['concurrency']
                    state['progress']=progress
                    (root/'controller-progress.json').write_text(json.dumps(progress,indent=2)+'\n')
                    print('GLM53_FORMAL_PROGRESS='+json.dumps(progress),flush=True)
                elif line.startswith('GLM53_FORMAL_DONE='):
                    assert done is None;done=json.loads(line.split('=',1)[1]);state['done_received']=True
                elif line.strip():
                    # Retain transport diagnostics without mistaking them for RPC.
                    state.setdefault('other_lines',[]).append(line[:1000]);assert len(state['other_lines'])<=100
            process.stdin.close();state['connection_rc']=process.wait(timeout=60)
            assert done is not None,'Controller connection ended without complete result; preserve remote client'
            for key in ['project_id','framework','namespace']:assert done['state'][key]==expected[key]
            raw=None
            if 'formal_report_gzip_base64' in done:
                packed=base64.b64decode(done['formal_report_gzip_base64'],validate=True)
                with gzip.GzipFile(fileobj=__import__('io').BytesIO(packed)) as stream:raw=stream.read(REPORT_CAP+1)
                assert len(raw)<=REPORT_CAP and hashlib.sha256(raw).hexdigest()==done['formal_report_sha256']
                report=json.loads(raw)
                for key in ['project_id','framework','namespace','concurrency']:assert report[key]==expected[key]
                (root/'formal-result.remote.json').write_bytes(raw)
                state['report_sha256']=done['formal_report_sha256'];state['client_baseline_accepted']=report.get('baseline_accepted') is True
                state['backend_HTTP_attempts']=report.get('backend_HTTP_attempts')
            else:state['report_transport_limitation']=done.get('report_transport_limitation','No formal report produced')
            # Compact wire metadata excludes duplicated base64 body.
            compact={k:v for k,v in done.items() if k!='formal_report_gzip_base64'}
            (root/'controller-done.json').write_text(json.dumps(compact,indent=2)+'\n')
    except BaseException as error:
        state['error']=repr(error);raise
    finally:
        state['terminal_monotonic_ns']=time.monotonic_ns()
        (root/'controller-rpc-state.json').write_text(json.dumps(state,indent=2)+'\n')
    return state
