python3 - <<'PY'
from pathlib import Path
import ast, datetime, hashlib, json, re, socket, subprocess, urllib.request

assert socket.gethostname()=='<ROLE_HOSTNAME>'
root=Path('/PRIVATE_TASK_ROOT')
client=root/'client_ds41_g1122'
control=client/'SG_PD_g1178_full_GSM8K_control_g1217'
output=client/'SG_PD_g1178_full_GSM8K_g1192'
handle=json.loads((control/'actual_launch_handle.json').read_text())
name=handle['argv'][-1]
assert handle['outer_PID']==2879093 and name.endswith('/run_full_gsm8k_SG_PD_g1192.py')
top=subprocess.check_output(['docker','top',handle['CPU_container_ID'],'-eo','pid,args'],text=True)
assert name not in top, 'Quality job remains active'
outer=Path('/proc',str(handle['outer_PID']),'cmdline')
assert not outer.exists() or name.encode() not in outer.read_bytes()
for base in ('http://<ROLE_HOST>:19093','http://127.0.0.1:19094'):
    with urllib.request.urlopen(base+'/metrics',timeout=8) as response:
        metrics=response.read().decode()
    for key in ['num_running_reqs','num_queue_reqs']:
        values=[float(v) for v in re.findall(r'^sglang:'+key+r'(?:\{[^\n]*\})?\s+([0-9.eE+\-]+)',metrics,re.M)]
        assert values and sum(values)==0,(key,values)
state=json.loads((output/'state.json').read_text())
result=json.loads((output/'result.json').read_text()) if (output/'result.json').is_file() else None
source=client/Path(name).name
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(source)==handle['source_SHA256']
data_root=root/'client_ds41_g503/quality_g655'
for entry in json.loads((data_root/'dataset_provenance.json').read_text())['files']:
    assert sha(data_root/entry['path'])==entry['sha256']
dataset=[json.loads(row) for row in (data_root/'grade_school_math/data/test.jsonl').read_bytes().splitlines()]
parts=[]
for node in ast.parse((data_root/'grade_school_math/dataset.py').read_text()).body:
    if isinstance(node,ast.Assign) and any(isinstance(var,ast.Name) and var.id in {'ANS_RE','INVALID_ANS'} for var in node.targets):
        parts.append(node)
    if isinstance(node,ast.FunctionDef) and node.name in {'extract_answer','is_correct'}:
        parts.append(node)
assert len(parts)==4
env={'re':re}
exec(compile(ast.Module(body=parts,type_ignores=[]),'<official grader>','exec'),env)
log=output/'results.jsonl'
rows=[json.loads(line) for line in log.read_bytes().splitlines()] if log.is_file() else []
assert len({row['index'] for row in rows})==len(rows)
correct=transport=limited=0
for row in rows:
    index=row['index']
    expected=dataset[index]
    base=f'{index:04d}'
    request=output/(base+'_request.json')
    body=json.loads(request.read_text())
    prompt=expected['question']+'\nSolve the problem. Show a brief calculation, then end your response with "#### <numeric answer>".'
    assert body=={'model':'ds41flash','messages':[{'role':'user','content':prompt}],
                  'temperature':0,'seed':0,'max_tokens':2048,'stream':False,
                  'chat_template_kwargs':{'thinking':False}}
    if not row['transport_ok']:
        transport+=1
        continue
    reply=output/(base+'_response.json')
    assert sha(request)==row['request_sha256'] and sha(reply)==row['response_sha256']
    payload=json.loads(reply.read_text())
    choice=payload['choices'][0]
    answer=choice['message'].get('content') or ''
    assert payload['model']=='ds41flash' and row['usage']==payload['usage']
    assert row['expected']==env['extract_answer'](expected['answer'])
    assert row['extracted']==env['extract_answer'](answer)
    assert row['correct']==env['is_correct'](answer,expected)
    assert row['finish_reason']==choice['finish_reason']
    correct+=row['correct']
    limited+=row['finish_reason']=='length'
complete=(result is not None and len(rows)==len(dataset)==1319
          and {row['index'] for row in rows}==set(range(1319)))
if complete:
    assert result['completed']==1319 and result['correct']==correct
    assert result['transport_failures']==transport and result['length_limited_responses']==limited
    assert result['results_jsonl_sha256']==sha(output/'results.jsonl')
files=[{'relative':str(p.relative_to(output)),'bytes':p.stat().st_size,'sha256':sha(p)}
       for p in sorted(output.rglob('*')) if p.is_file()]
manifest={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'files':files,'dataset_sha256':sha(data_root/'grade_school_math/data/test.jsonl'),
          'grader_sha256':sha(data_root/'grade_school_math/dataset.py'),
          'request_worker_sha256':sha(source),'state':state,
          'complete_1319':complete,'all_transport_successful':complete and transport==0,
          'completed':len(rows),'correct':correct,'transport_failures':transport,'length_limited':limited,
          'all_1319_accuracy':correct/1319 if complete and transport==0 else None,
          'scope':'Final DSPARK P/D G1178 quality. Terminal raw request and official grade audit. AISBench performance and selected long quality are separate.'}
target=control/'frozen_actual_manifest_g1249.json'
payload=json.dumps(manifest,indent=2)+'\n'
with target.open('x') as file:
    file.write(payload)
report={k:v for k,v in manifest.items() if k!='files'}
report.update(frozen_files=len(files),frozen_manifest_sha256=sha(target),manifest=str(target),
              all_actual_processes_exited=True)
print('G1249_FINAL_PD_GSM8K_FROZEN_JSON '+json.dumps(report))
PY
