"""Build the no-generation transfer and gate command for current G1178 receipts."""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import re


def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def main(args):
    assert re.fullmatch(r'[A-Za-z0-9_-]{1,80}',args.run_id)
    paths={'prefill':args.prefill,'decode':args.decode,'runtime':args.runtime}
    if args.prefill_compile:
        paths['prefill_compile']=args.prefill_compile
    assert ('prefill_compile' in paths)==(args.stage=='formal')
    raw={key:path.read_bytes() for key,path in paths.items()}
    objects={key:json.loads(value) for key,value in raw.items()}
    p,d,smoke=[objects[key] for key in ('prefill','decode','runtime')]
    assert p['ready']==d['ready']==200 and d['graph_capture_all8_verified']
    assert p['guard_FD_verified'] and p['guard_PID']==d['handle']['guard_PID']==2227468
    assert smoke['terminal_verified'] and smoke['raw_files_verified'] and smoke['result']['completed']==3
    peers=smoke['handle']['router']
    assert peers['P_container_ID']==p['container']['Id'] and peers['D_container_ID']==d['container_ID']
    if args.stage=='formal':
        boundary=objects['prefill_compile']
        assert boundary['P_container_ID']==p['container']['Id'] and boundary['persistent_compiler_artifacts_unchanged']
        assert boundary['before_label']==f'after_warmup3_{args.length}'
        assert boundary['after_label']==f'after_warmup5_{args.length}'

    remote=r"""python3 - <<'PY'
from pathlib import Path
import base64, hashlib, json, socket, subprocess

assert socket.gethostname()=='<ROLE_HOSTNAME>'
root=Path('/PRIVATE_TASK_ROOT')
client=root/'client_ds41_g1122'
runner=client/'build_SG_AISBench_gate_g1229.py'
assert hashlib.sha256(runner.read_bytes()).hexdigest()=='0787333203d798997f6e73ab62fa068b8c38a76d78c2b9361b58d357777af05d'
staged=__PAYLOADS__
pins=__PINS__
run_id=__RUN_ID__
stage=__STAGE__
length=__LENGTH__
bound=__BOUND__
target=client/'SG_service_gate_receipts_g1261'/run_id/stage
target.mkdir(parents=True,exist_ok=False)
files={}
for key,encoded in staged.items():
    data=base64.b64decode(encoded)
    assert hashlib.sha256(data).hexdigest()==pins[key]
    file=target/(key+'.json')
    with file.open('xb') as stream:
        stream.write(data)
    assert hashlib.sha256(file.read_bytes()).hexdigest()==pins[key]
    files[key]=file
result=target/'service_gate.json'
argv=['python3','-u',str(runner),'--root',str(root),'--output',str(result),'--run-id',run_id,
      '--stage',stage,'--length',str(length),'--minimum-remaining',str(bound)]
for name,path in files.items():
    argv += ['--'+name.replace('_','-'),str(path)]
subprocess.run(argv,cwd=target,check=True,timeout=80)
gate=json.loads(result.read_bytes())
assert gate['stage']==stage and gate['input_tokens']==length
assert gate['source_receipts_sha256']['PD_runtime']==pins['runtime']
print('G1261_SG_GATE_BUNDLE_READY_JSON '+json.dumps({
    'gate':str(result),'runtime_receipt':str(files['runtime']),'sha256':hashlib.sha256(result.read_bytes()).hexdigest(),
    'requests_submitted':False,'stage':stage}))
PY
"""
    substitutions={'__PAYLOADS__':{n:base64.b64encode(v).decode() for n,v in raw.items()},
                   '__PINS__':{n:hash_bytes(v) for n,v in raw.items()},'__RUN_ID__':args.run_id,
                   '__STAGE__':args.stage,'__LENGTH__':args.length,'__BOUND__':args.minimum_remaining}
    for marker,value in substitutions.items():
        assert remote.count(marker)==1
        remote=remote.replace(marker,repr(value))
    ast.parse(remote.split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0])
    with args.output.open('x') as shell:
        shell.write(remote)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ('prefill','decode','runtime','output'):
        cli.add_argument('--'+name,type=Path,required=True)
    cli.add_argument('--prefill-compile',type=Path)
    cli.add_argument('--run-id',required=True)
    cli.add_argument('--stage',choices=('diagnostic','initial','remaining','formal'),required=True)
    cli.add_argument('--length',type=int,choices=(32768,65536,131072,262144),required=True)
    cli.add_argument('--minimum-remaining',type=int,default=900)
    main(cli.parse_args())
