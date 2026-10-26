"""Synthetic two-phase artifact admission review; no service or NPU."""
import base64, contextlib, copy, hashlib, importlib.util, io, json, sys, tempfile, types
from pathlib import Path
source=Path(sys.argv[1]);output=Path(sys.argv[2])
sys.path.insert(0,str(Path('scripts').resolve()))
sys.modules['aiohttp']=types.SimpleNamespace()
sp=importlib.util.spec_from_file_location('validator_review',source);v=importlib.util.module_from_spec(sp);sp.loader.exec_module(v)
sha=lambda b:hashlib.sha256(b).hexdigest()
dataset_bytes=Path('/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8-repeatRate0.9.jsonl').read_bytes()
lines=dataset_bytes.decode().splitlines()[:48]
def event(raw,seq,rid,t):
    if isinstance(raw,dict):raw=json.dumps(raw).encode()
    e={'seq':seq,'payload_b64':base64.b64encode(raw).decode(),'payload_sha256':sha(raw),'response_id':rid,'fragment_recv_monotonic_ns':t}
    if raw==b'[DONE]':e['done']=True
    return e
def fixtures(phase,begin):
    records=[]
    for i in range(48):
        rid=phase+str(i);start=begin+(i//12)*200_000_000+1;end=start+100_000_000
        use={'completion_tokens':1024,'prompt_tokens':32768,'total_tokens':33792}
        ev=[event({'id':rid,'choices':[{'index':0,'delta':{'content':'x'},'finish_reason':'length'}]},0,rid,start+20_000_000),event({'id':rid,'choices':[],'usage':use},1,rid,start+30_000_000),event(b'[DONE]',2,rid,start+40_000_000)]
        ev[0]['has_text_delta']=True;ev[1]['has_text_delta']=False
        row={'i':i,'phase':phase,'http_status':200,'error':None,'output_tokens':1024,'usage':use,'input_tokens':32768,'response_id':rid,'start_monotonic_ns':start,'end_monotonic_ns':end,'start':start/1e9,'end':end/1e9,'ttft_ms':20.0,'tpot_ms':80/1023,'chunks':1,'sse_events':len(ev),'dataset_row_sha256':sha(lines[i].encode()),'request_body_sha256':sha(json.dumps(v.body(json.loads(lines[i])['question'],1024),sort_keys=True,ensure_ascii=False).encode())}
        records.append({'row':row,'events':ev})
    summary={'scope':'instrumented_bound_diagnostic_not_formal_tps','phase':phase,'n':48,'success':48,'fail':0,'concurrency':12,'max_tokens':1024,'wall_start_monotonic_ns':begin,'wall_end_monotonic_ns':begin+800_000_000,'duration_s':.8,'clock':{'kernel_boot_id':'boot','time_namespace':'time:[1]','timens_offsets':'monotonic 0 0\nboottime 0 0\n','pid':1,'monotonic_info':'clock_gettime(CLOCK_MONOTONIC)','perf_counter_info':'clock_gettime(CLOCK_MONOTONIC)'}}
    return records,summary
def replace_first(sets,raw):
    rec=sets[1][0][0];old=rec['events'][0];rec['events'][0]=event(raw,0,rec['row']['response_id'],old['fragment_recv_monotonic_ns'])
    rec['events'][0]['has_text_delta']=old.get('has_text_delta',False)
def payload(sets):return json.loads(base64.b64decode(sets[1][0][0]['events'][0]['payload_b64']))
def update_payload(sets,**kw):
    p=payload(sets);p.update(kw);replace_first(sets,p)
def no_finish(sets):
    p=payload(sets);p['choices'][0]['finish_reason']=None;replace_first(sets,p)
def duplicate_usage(sets):
    rec=sets[1][0][0];rec['events'].insert(1,copy.deepcopy(rec['events'][1]))
    for i,e in enumerate(rec['events']):e['seq']=i
    rec['row']['sse_events']=len(rec['events'])
def usage_mismatch(sets):sets[1][0][0]['row']['usage']=dict(completion_tokens=1024,prompt_tokens=1)
def overlap(sets):
    records,s=sets[1];s['wall_start_monotonic_ns']=0;s['wall_end_monotonic_ns']=10_000_000_000
    for rec in records:rec['row']['start_monotonic_ns']=1;rec['row']['end_monotonic_ns']=9_000_000_000
def no_text(s):
    p=payload(s);p['choices'][0]['delta']={};replace_first(s,p);s[1][0][0]['events'][0]['has_text_delta']=False
def reverse_events(s):
    a,b=s[1][0][0]['events'][:2];a['fragment_recv_monotonic_ns'],b['fragment_recv_monotonic_ns']=b['fragment_recv_monotonic_ns'],a['fragment_recv_monotonic_ns']
def empty_clocks(s):
    for _,summary in s:
        for k in ('kernel_boot_id','time_namespace','timens_offsets'):summary['clock'][k]=''
tests={'valid':lambda s:None,'malformed_earlier_json':lambda s:replace_first(s,b'{bad}'),'embedded_wrong_id':lambda s:update_payload(s,id='wrong'),'error_payload':lambda s:update_payload(s,error={'message':'failure'}),'no_finish':no_finish,'duplicate_usage':duplicate_usage,'row_final_usage_mismatch':usage_mismatch,'event_outside_request':lambda s:s[1][0][0]['events'][0].update(fragment_recv_monotonic_ns=0),'clock_mismatch':lambda s:s[1][1]['clock'].update(kernel_boot_id='other'),'over48_peak':overlap,'reversed_phases':lambda s:s.reverse(),'no_text_but_success':no_text,'backward_event_times':reverse_events,'empty_clock_origins':empty_clocks}
def change_id(rec,rid):
    rec['row']['response_id']=rid
    for ev in rec['events']:
        ev['response_id']=rid
        if not ev.get('done'):
            data=json.loads(base64.b64decode(ev['payload_b64']));data['id']=rid;raw=json.dumps(data).encode()
            ev['payload_b64']=base64.b64encode(raw).decode();ev['payload_sha256']=sha(raw)
tests.update({'duplicate_across_phases':lambda s:change_id(s[1][0][0],s[0][0][0]['row']['response_id']),'warmup_only_valid':lambda s:None,'warmup_only_duplicate':lambda s:change_id(s[0][0][1],s[0][0][0]['row']['response_id']),'wrong_body':lambda s:s[1][0][0]['row'].update(request_body_sha256='0'*64),'wrong_hash':lambda s:s[1][0][0]['events'][0].update(payload_sha256='0'*64),'wrong_dataset':lambda s:None})
results={}
for name,mutate in tests.items():
    sets=[fixtures('warmup',1_000_000_000),fixtures('measured',3_000_000_000)]
    mutate(sets)
    if name=='reversed_phases':
        # Preserve phase names but make measured interval precede warmup.
        sets=[fixtures('warmup',3_000_000_000),fixtures('measured',1_000_000_000)]
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);dataset=root/'dataset';dataset.write_bytes(dataset_bytes)
        if name=='wrong_dataset':dataset.write_bytes(dataset_bytes+b'\n')
        for phase,(records,summary) in zip(('warmup','measured'),sets):
            d=root/phase;d.mkdir()
            for i,record in enumerate(records):(d/f'request_{i:03d}.json').write_text(json.dumps(record))
            (d/'summary.json').write_text(json.dumps({'summary':summary,'requests':[r['row'] for r in records]}))
        sys.argv=['review','--dataset',str(dataset),'--warmup-dir',str(root/'warmup'),'--measured-dir',str(root/'measured'),'--output',str(root/'result')]
        if name.startswith('warmup_only'):
            sys.argv=['review','--dataset',str(dataset),'--warmup-dir',str(root/'warmup'),'--warmup-only','--output',str(root/'result')]
        error=None
        try:
            with contextlib.redirect_stdout(io.StringIO()):v.main()
            admitted=True
        except Exception as e:admitted=False;error=repr(e)
        results[name]={'admitted':admitted,'expected_admitted':name in ('valid','warmup_only_valid'),'error':error}
proof={'validator_sha256':sha(source.read_bytes()),'scope':'synthetic JSON files; no service/NPU','cases':results,'pass':all(x['admitted']==x['expected_admitted'] for x in results.values())}
output.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2))
