"""CPU43 host controller: native AIS child, one cache-gate RPC, complete artifacts.

Only a reviewed ZCode launcher may invoke this file. JSON lines on stdin/stdout
carry control results; the model's streams remain in the owned run directory.
"""
import argparse, base64, gzip, hashlib, json, os, socket, subprocess, sys, time
from pathlib import Path

def atomic(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)

def main(settings):
    assert settings['project_id']=='glm53'
    assert socket.gethostname()=='B7-23U-Atlas900RCKA3-43'
    framework=settings['framework'];C=settings['concurrency'];run_id=settings['run_id']
    assert framework in ['sglang','vllm'] and C in [16,32,64]
    namespace=settings['namespace']
    assert namespace.startswith('glm53-formal-') and run_id in namespace and namespace.endswith(f'-C{C}')
    root=Path(settings['root'])
    assert str(root)=='/data/wio/inference-foundry/glm53/runs/'+run_id
    assert root.is_dir() and not (root/'host-controller.once').exists()
    with (root/'host-controller.once').open('x') as file:file.write(namespace)
    for name,sha in settings['files_sha256'].items():
        assert Path(name).name==name and hashlib.sha256((root/name).read_bytes()).hexdigest()==sha
    data=Path('/data/wio/inference-foundry/glm53/runs/20261017T0615-formal-inputs-v3/artifacts')/f'C{C}'
    assert hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest()==settings['input_manifest_sha256']
    client=settings['client_settings']
    assert client=={'framework':framework,'concurrency':C,'namespace':namespace,'run_id':run_id,
                    'dataset':'/formal-inputs','output':'/run/glm53/output'}
    raw=json.dumps(client,indent=2).encode()+b'\n';(root/'client-settings.json').write_bytes(raw)
    output=root/'output';output.mkdir()
    image='sha256:008d2ebcd28a18f4dfbc855d23d763a2bdc63c4ce82c6b7b761580ad6d15d0a7'
    inspected=subprocess.run(['docker','image','inspect',image,'--format','{{.Id}}'],capture_output=True,text=True,timeout=20)
    assert inspected.returncode==0 and inspected.stdout.strip()==image
    args=['docker','run','--rm','--pull=never','--network=host','--pids-limit=512','--memory=16g','--cpus=4',
          '--label','project_id=glm53','--label','run_id='+run_id,'--name',settings['container_name'],
          '--cidfile',str(root/'client.cid'),'--workdir','/run/glm53',
          '-v',str(root)+':/run/glm53','-v',str(data)+':/formal-inputs:ro',
          '-v','/data/wio/aisbench-offline-43-20261014:/work:ro']
    environment={'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1',
                 'PYTHONPATH':'/run/glm53:/work/benchmark','OMP_NUM_THREADS':'1',
                 'MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false',
                 'CUDA_VISIBLE_DEVICES':'','ASCEND_RT_VISIBLE_DEVICES':'',
                 'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HOME':'/run/glm53/hf',
                 'XDG_CACHE_HOME':'/run/glm53/cache','MPLCONFIGDIR':'/run/glm53/mpl',
                 'TORCH_HOME':'/run/glm53/torch','NLTK_DATA':'/work/nltk_data',
                 'TIKTOKEN_CACHE_DIR':'/work/tiktoken_cache'}
    for key,value in environment.items():args+=['-e',key+'='+value]
    args+=['--entrypoint','/work/venv/bin/python',image,'/run/glm53/glm53_formal_ais_client_v1.py',
           '--settings','/run/glm53/client-settings.json','--sha256',hashlib.sha256(raw).hexdigest()]
    return control(root,args,settings)

def control(root,args,settings):
    output=root/'output'
    framework=settings['framework'];C=settings['concurrency'];namespace=settings['namespace']
    state={'project_id':'glm53','framework':framework,'concurrency':C,'namespace':namespace,
           'cache_gate_received':False,'client_rc':None,'model_requests':'COUNT_FROM_ACTUAL_GATEWAY_ONLY'}
    process=None
    try:
        with (root/'client-stdout.txt').open('x') as stdout,(root/'client-stderr.txt').open('x') as stderr:
            process=subprocess.Popen(args,stdout=stdout,stderr=stderr,start_new_session=True)
            state['host_docker_client_pid']=process.pid
            atomic(root/'controller-status.json',state)
            while process.poll() is None:
                request_path=output/'need-cache-gate.json'
                if request_path.is_file() and not state['cache_gate_received']:
                    request=json.loads(request_path.read_text())
                    assert request['project_id']=='glm53' and request['framework']==framework
                    assert request['namespace']==namespace and request['concurrency']==C
                    assert request['manifest_sha256']==settings['input_manifest_sha256']
                    print('GLM53_FORMAL_NEED_GATE='+json.dumps(request),flush=True)
                    line=sys.stdin.readline()
                    if not line:raise EOFError('Cache-gate RPC disconnected; preserve client/logs')
                    response=json.loads(line)
                    assert response['project_id']=='glm53' and response['namespace']==namespace and response['framework']==framework
                    atomic(output/'cache-gate.json',response)
                    state['cache_gate_received']=True;state['cache_gate_pass']=response.get('pass') is True
                    atomic(root/'controller-status.json',state)
                time.sleep(.2)
            state['client_rc']=process.returncode
    except BaseException as error:
        state['error']=repr(error)
        # No automatic cancellation: an active child remains owned and observable.
        raise
    finally:
        result_path=output/'formal-result.json'
        report=None
        if result_path.is_file():
            try:report=json.loads(result_path.read_text())
            except (OSError,ValueError) as error:state['report_read_error']=repr(error)
        state['baseline_accepted']=bool(report and report['baseline_accepted'])
        state['backend_HTTP_attempts']=report.get('backend_HTTP_attempts') if report else None
        atomic(root/'controller-status.json',state)
        artifacts={}
        for path in sorted(root.rglob('*')):
            if path.is_file():
                raw=path.read_bytes();artifacts[str(path.relative_to(root))]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
        message={'state':state,'artifacts':artifacts}
        if report is not None:
            raw=result_path.read_bytes();message['formal_report_sha256']=hashlib.sha256(raw).hexdigest()
            if len(raw)<=64*1024**2:message['formal_report_gzip_base64']=base64.b64encode(gzip.compress(raw,mtime=0)).decode()
            else:message['report_transport_limitation']='Full report exceeds64MiB; retained on CPU43, requires separate complete audit'
        print('GLM53_FORMAL_DONE='+json.dumps(message),flush=True)
    return 0 if state['client_rc']==0 and state['baseline_accepted'] else 1

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--settings',required=True);parser.add_argument('--sha256',required=True)
    args=parser.parse_args();raw=Path(args.settings).read_bytes();assert hashlib.sha256(raw).hexdigest()==args.sha256
    raise SystemExit(main(json.loads(raw)))
