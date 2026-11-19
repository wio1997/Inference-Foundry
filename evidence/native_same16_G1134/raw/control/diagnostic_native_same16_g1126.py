"""Native P1121 same16 C1 eager control: original16 short messages/max128, no GSM.
No SP/directnativeP1121/prefixON; stocknative032, no source overlay.
Not full quality or benchmark.
"""
from pathlib import Path
import ast, concurrent.futures, datetime, hashlib, json, os, re, signal, time, urllib.request

ROOT=Path('/task/client_ds41_g503/quality_g655')
OUT=Path('/task/client_ds41_g1122/diagnostic_native_same16_g1126')
os.chdir(ROOT);OUT.mkdir(exist_ok=False)
started=time.monotonic()
URL='http://<ROLE_HOST>:19103/v1/chat/completions'

def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def state(phase,**more):
    value=dict(phase=phase,pid=os.getpid(),at=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.monotonic()-started,**more)
    save(OUT/'state.json',value);print(json.dumps(value),flush=True)

def alarm(*_): raise TimeoutError('C1 serial diagnostic480s bound; no automatic restart')
signal.signal(signal.SIGALRM,alarm);signal.alarm(480)

data=[json.loads(line) for line in (ROOT/'grade_school_math/data/test.jsonl').read_text().splitlines()]
assert len(data)==1319
nodes=[]
for node in ast.parse((ROOT/'grade_school_math/dataset.py').read_text()).body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['ANS_RE','INVALID_ANS'] for t in node.targets): nodes.append(node)
    if isinstance(node,ast.FunctionDef) and node.name in ['extract_answer','is_correct']: nodes.append(node)
env={'re':re};exec(compile(ast.Module(body=nodes,type_ignores=[]),'<official diagnostic extraction>','exec'),env)

simple=[]
for i in range(8):
    a,b=i+13,i+5;answer=str(a*b+7)
    simple.append(dict(case=f'math{i}',kind='arithmetic',prompt=f'Case C16-g999-{i:02d}. Compute {a} times {b}, then add 7. Reply with only the integer.',expected=answer))
for i in range(8):
    code='K'+hashlib.sha256(f'C16-g999-binding-{i}'.encode()).hexdigest()[:9].upper()
    simple.append(dict(case=f'bind{i}',kind='case_binding',prompt=f'Record C16-g999-{i+8:02d} has access code {code}. Copy this record\'s access code exactly. Reply with only the code.',expected=code))
gsm=[dict(case=f'gsm{i:02d}',kind='GSM_prefix',index=i,
    prompt=row['question']+'\nSolve the problem. Show a brief calculation, then end your response with "#### <numeric answer>".',
    expected=env['extract_answer'](row['answer'])) for i,row in enumerate(data[:16])]
manifest=dict(scope='DIAGNOSTIC_NATIVE_P1121_SAME16_NO_SP_EAGER; sameoriginal16simplemessages/max128/C1serial. Notfinalalignment/fullquality/E2Ebenchmark.',
    endpoint=URL,endpoint_scope='DirectnativeP1121; SG_D1098idle; noPD',
    concurrency=1,retries=0,temperature=0,seed=0,thinking=False,HTTP_timeout_s=240,
    simple_max_tokens=128,GSM_diagnostic_max_tokens=None,full_quality_budget_unchanged=2048,
    dataset_SHA256=hashlib.sha256((ROOT/'grade_school_math/data/test.jsonl').read_bytes()).hexdigest(),
    cohorts={'simpleC16':simple,'GSMprefixC16':gsm})
save(OUT/'manifest.json',manifest)

def task(item,limit,cohort):
    tag=item['case'];folder=OUT/cohort
    body=dict(model='ds41flash',messages=[{'role':'user','content':item['prompt']}],
        temperature=0,seed=0,max_tokens=limit,stream=True,stream_options={'include_usage':True},chat_template_kwargs={'thinking':False})
    save(folder/(tag+'_request.json'),body)
    before=time.monotonic();text=[];reasoning=[];usage=None;finish=None;done=False;first=None
    try:
        request=urllib.request.Request(URL,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=240) as response,(folder/(tag+'_response.sse')).open('xb') as raw:
            assert response.status==200
            for line in response:
                raw.write(line);raw.flush()
                if not line.startswith(b'data:'): continue
                payload=line[5:].strip()
                if payload==b'[DONE]': done=True;break
                if not payload:continue
                event=json.loads(payload)
                if first is None:first=time.monotonic()-before
                if event.get('usage'):usage=event['usage']
                for choice in event.get('choices',[]):
                    delta=choice.get('delta') or {}
                    text.append(delta.get('content') or '')
                    reasoning.append(delta.get('reasoning_content') or '')
                    if choice.get('finish_reason'):finish=choice['finish_reason']
                # Persist live content for diagnosis even if the HTTP budget fails.
                save(folder/(tag+'_partial.json'),dict(text=''.join(text),reasoning=''.join(reasoning),usage=usage,
                    finish_reason=finish,elapsed_s=time.monotonic()-before))
        assert done and usage and usage['completion_tokens']>0
        content=''.join(text);value=dict(case=tag,kind=item['kind'],transport_ok=True,SSE_DONE=True,
            expected=item['expected'],text=content,reasoning=''.join(reasoning),usage=usage,
            finish_reason=finish,elapsed_s=time.monotonic()-before,first_event_s=first,
            response_SHA256=hashlib.sha256((folder/(tag+'_response.sse')).read_bytes()).hexdigest())
        if item['kind']=='GSM_prefix':
            value.update(official_extracted_at_diagnostic_budget=env['extract_answer'](content),full_quality_score=False)
        else:value['exact_expected_match']=content.strip().rstrip('。.!')==item['expected']
        return value
    except Exception as error:
        return dict(case=tag,kind=item['kind'],transport_ok=False,error=repr(error),text=''.join(text),
                    reasoning=''.join(reasoning),finish_reason=finish,usage=usage,elapsed_s=time.monotonic()-before)

try:
    all_results={}
    for name,items,limit in [('simpleC1',simple,128)]:
        folder=OUT/name;folder.mkdir();state('cohort_running',cohort=name)
        results=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool,(folder/'results.jsonl').open('x') as log:
            for future in concurrent.futures.as_completed([pool.submit(task,item,limit,name) for item in items]):
                result=future.result();results.append(result);log.write(json.dumps(result,ensure_ascii=False)+'\n');log.flush()
        assert len(results)==16
        all_results[name]=results;save(folder/'result.json',results)
        if any(not result['transport_ok'] for result in results):
            state('failed_transport_requires_review',cohort=name);break
    save(OUT/'result.json',dict(scope=manifest['scope'],cohorts=all_results,full_quality_accepted=False))
    state('diagnostic_complete_requires_parent_review',cohorts=list(all_results))
except BaseException as error:
    state('failed_requires_review',error=repr(error));raise
