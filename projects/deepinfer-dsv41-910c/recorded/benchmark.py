"""AISBench single-factor ablations on the fixed C8 workload."""
from pathlib import Path
import concurrent.futures,csv,json,os,shutil,subprocess,sys,time,traceback,urllib.request

p=Path('/data/tiankuan/wio/910c-deepinfer-172')
version='allopt-mc2-cap4096-v1-20261028'
root=p/'benchmarks'/version
control=p/'serving'/version
progress=root/'progress.json'
assert not progress.exists(),'refuse repeat'
modes={'reference': {'varlen': True, 'tail_ffn': True, 'engram_prefetch': False, 'engram_pinned': False, 'engram_uva': True}}
order=[('reference', 'measure1'), ('reference', 'measure2')]
state={'state':'STARTING','started':time.time(),'phases':[],
       'workload':{'input':32768,'output':1024,'concurrency':8,'rounds_per_mode':2,
                   'actual_cached_tokens':235904,'prompt_tokens':262144},
       'mode_definitions':modes,'round_order':order,
       'frozen_vllm_output_tps':518.4171948453181,'frozen_vllm_mean_ttft_ms':831.25,
       'model_math_unchanged':True,'engram_lookup_implementation_changed':False,'fused_mc2_capacity_changed':True,'routing_reference':{},
       'prior_experiment':'single-point-20261028; exploratory, DP arrival-order confound'}

def save(status):
    state.update(state=status,updated=time.time())
    tmp=progress.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(progress)
    print(json.dumps({'state':status,'completed':sum(x.get('state')=='PASS' for x in state['phases'])}),flush=True)

def http(path,body=None):
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request=urllib.request.Request('http://127.0.0.1:8172/'+path,
        data=None if body is None else json.dumps(body).encode(),
        headers={} if body is None else {'Content-Type':'application/json'})
    with opener.open(request,timeout=900 if body is not None else 10) as response:
        return response.read()

def metrics():
    raw=http('metrics').decode()
    return {line.split()[0]:int(float(line.split()[1])) for line in raw.splitlines()
            if line.startswith('deepinfer_')},raw

def complete(row):
    return json.loads(http('v1/completions',{'model':'dsv41','prompt':row['prompt'],
        'max_tokens':row['max_tokens'],'temperature':0,'ignore_eos':row.get('ignore_eos',False),'stream':False}))

def parallel(rows):
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(complete,rows))

try:
    assert json.loads(http('health'))=={'status':'ok','backend':'DeepInfer-native'}
    ready=[json.loads(f.read_text()) for f in control.glob('ready-rank*.json')]
    assert len(ready)==16 and all(r['dspark_graphs_initialized'] and r['target_vocab_parallel'] for r in ready)
    assert all(r['mc2_capacity_floor']==4096 for r in ready)
    for name,phase in order:
        mode=modes[name]
        folder=p/'workload/frozen'/phase
        data=[json.loads(line) for line in next(folder.rglob('*.requests.jsonl')).open()]
        refs={int(row['data_id']):row['prediction'] for row in
              map(json.loads,(folder/'frozen-vllm-details.jsonl').open())}
        dest=root/(name+'-'+phase);dest.mkdir(exist_ok=False)
        tmp=control/'single-point-mode.tmp';tmp.write_text(json.dumps(mode));tmp.replace(control/'single-point-mode.json')
        item={'mode':name,'phase':phase,'controls':mode,'state':'PREFIX_WARMUP'}
        state['phases'].append(item);save(name+'_'+phase+'_PREFIX_WARMUP')
        seeds=json.loads((folder/'warmup.json').read_text())
        warm=parallel([{'prompt':row['prompt'],'max_tokens':1} for row in seeds])
        (dest/'prefix-warmup.json').write_text(json.dumps(warm,ensure_ascii=False))
        item['state']='C8_WARMUP';save(name+'_'+phase+'_C8_WARMUP')
        warm=parallel([{'prompt':row['question'],'max_tokens':32,'ignore_eos':True} for row in data])
        (dest/'c8-warmup.json').write_text(json.dumps(warm,ensure_ascii=False))
        for i,response in enumerate(warm):
            assert refs[i].startswith(response['choices'][0]['text']),('warmup output mismatch',name,phase,i)
            assert response['usage']['prompt_tokens']==32768 and response['usage']['completion_tokens']==32
        assert sum(r['usage']['prompt_tokens_details']['cached_tokens'] for r in warm)==235904
        before_jobs={d.name for d in (control/'jobs').iterdir()}
        before,raw=metrics();(dest/'metrics-before.txt').write_text(raw)
        item['state']='AISBENCH_RUNNING';save(name+'_'+phase+'_AISBENCH_RUNNING')
        command=[sys.executable,'-m','ais_bench.benchmark.cli.main',str(root/(name+'-'+phase+'.py')),
                 '--mode','perf','--num-warmups','0','--num-prompts','8','--max-num-workers','1']
        with (dest/'aisbench.log').open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        assert result.returncode==0,('AISBench exit',result.returncode,name,phase)
        jobs=[d for d in (control/'jobs').iterdir() if d.name not in before_jobs]
        assert len(jobs)==1,('formal job count',len(jobs))
        job=jobs[0]
        import hashlib
        submitted=json.loads((job/'request.json').read_text())['requests']
        assert [tuple(r['token_ids']) for r in submitted]==sorted(tuple(r['token_ids']) for r in submitted)
        routing=[{'dp':i%2,'slot':i//2,'prompt_sha256':hashlib.sha256(json.dumps(r['token_ids'],separators=(',',':')).encode()).hexdigest()}
                 for i,r in enumerate(submitted)]
        if phase in state['routing_reference']:
            assert routing==state['routing_reference'][phase],('DP routing changed',name,phase)
        else:state['routing_reference'][phase]=routing
        item['deterministic_dp_routing']=routing
        acknowledgements=[json.loads(f.read_text()) for f in job.glob('ablation-rank*.json')]
        assert sorted(x['rank'] for x in acknowledgements)==list(range(16))
        assert all(x['mode']==mode and x['all_graphs_retained'] for x in acknowledgements)
        proof=[json.loads(f.read_text()) for f in job.glob('result-dp*.json')]
        assert len(proof)==2 and not list(job.glob('error-rank*.txt'))
        for row in proof:
            st=row['runtime_stats'];sd=st['dspark']
            assert st['tail_ffn_enabled']==mode['tail_ffn']
            assert st['fused_mc2_capacity_floor']==4096
            capacities=st['fused_mc2_capacity_records']
            assert capacities and all(r['max_output_size']>=max(4096,r['global_bs']*r['topk']) for r in capacities)
            assert any(r['global_bs']==48 and r['topk']==6 and r['max_output_size']==4096 for r in capacities)

            assert st['engram_uva_enabled']==mode['engram_uva']
            assert st['engram_uva'] and st['engram_uva']['host_registered_bytes']>0
            assert st['engram_uva']['storage']=='shared memfd MAP_SHARED, original checkpoint bytes'
            assert bool(st['engram_uva']['calls']>0)==mode['engram_uva']

            assert st['engram_prefetch_enabled']==mode['engram_prefetch']
            fetch=st.get('engram_prefetch')
            if mode['engram_prefetch']:
                assert fetch and fetch['batches']>0 and fetch['layers_submitted']==fetch['layers_consumed'] and fetch['h2d_bytes']>0,fetch
            else:assert not fetch or fetch['batches']==0,fetch
            assert st['engram_pinned_staging_enabled']==mode['engram_pinned']
            assert bool(st['engram_pinned_staging_calls']>0)==mode['engram_pinned']

            assert bool(st['tail_ffn_skipped_tokens']>0)==mode['tail_ffn']
            assert st['target_vocab_parallel_enabled'] and st['fused_mc2_decode_layers']==40
            assert st['sequence_parallel_enabled'] and st['context_parallel_enabled'] and st['kv_token_parallel_enabled']
            assert sd['verification_mode']==('confidence-scheduled' if mode['varlen'] else 'fixed-5')
            assert not st['prompt_lookup'] and st['dspark_weights_loaded'] and st['dspark_draft_layers']==3
            assert sd['proposals']>0 and sd['confidence_head_executed'] and sd['markov_head_executed']
            if mode['varlen']:
                assert sd['variable_target_prefix_validation']['state']=='PASS'
        after,raw=metrics();(dest/'metrics-after.txt').write_text(raw)
        deltas={key:after[key]-before[key] for key in before}
        expected={'deepinfer_requests_total':8,'deepinfer_prompt_tokens_total':262144,
                  'deepinfer_cached_tokens_total':235904,'deepinfer_output_tokens_total':8192,
                  'deepinfer_failed_requests_total':0,'deepinfer_tokenizer_cache_hits_total':8,
                  'deepinfer_tokenizer_cache_hit_tokens_total':262144}
        for key,value in expected.items():assert deltas[key]==value,(key,deltas)
        metric_files=[f for f in (dest/'aisbench').glob('*/performances/*/*.json') if not f.name.endswith('_details.json')]
        assert len(metric_files)==1,metric_files
        mf=metric_files[0];native=json.loads(mf.read_text())
        details=[json.loads(line) for line in mf.with_name(mf.stem+'_details.jsonl').open()]
        assert len(details)==8 and all(r['success'] and r['input_tokens']==32768 and r['output_tokens']==1024 for r in details)
        for row in details:assert row['prediction']==refs[int(row['data_id'])],('full output mismatch',name,phase,row['data_id'])
        duration=float(str(native['Benchmark Duration']['total']).split()[0])/1000
        table={row['Performance Parameters']:row for row in csv.DictReader(mf.with_suffix('.csv').open())}
        shutil.copytree(job,dest/'raw-job')
        item.update(state='PASS',duration_seconds=duration,output_tps=8192/duration,
                    mean_ttft_ms=float(table['TTFT']['Average'].split()[0]),
                    mean_tpot_ms=float(table['TPOT']['Average'].split()[0]),
                    exact_full1024_outputs=8,counter_deltas=deltas,
                    path_proof=[row['runtime_stats'] for row in proof])
        save('PASS_'+name+'_'+phase)
    summaries={}
    for name in modes:
        rows=[row for row in state['phases'] if row['mode']==name]
        assert len(rows)==2 and all(row['state']=='PASS' for row in rows)
        summaries[name]={'output_tps':16384/sum(row['duration_seconds'] for row in rows),
                         'mean_ttft_ms':sum(row['mean_ttft_ms'] for row in rows)/2,
                         'round_tps':[row['output_tps'] for row in rows],
                         'mean_tpot_ms':sum(row['mean_tpot_ms'] for row in rows)/2}
    state['summaries']=summaries
    state['single_factor_effects']={name:{'output_gain_fraction':s['output_tps']/summaries['reference']['output_tps']-1,'ttft_reduction_fraction':1-s['mean_ttft_ms']/summaries['reference']['mean_ttft_ms']} for name,s in summaries.items() if name!='reference'}
    state['cross_engine_reference']={'frozen_vllm_output_tps':518.4171948453181,
        'current_reference_gain_fraction':summaries['reference']['output_tps']/518.4171948453181-1,
        'contemporaneous_vllm_rerun':False}
    state['scope']='two rounds per mode; device direct host lookup in decode only; CPU hashes remain; all-optimizations goal remains ACTIVE; near-noise effects inconclusive'

    prior=json.loads((p/'benchmarks/allopt-engram-uva-v2-20261028/progress.json').read_text())['summaries']['uva']
    current=summaries['reference']
    state['workspace_comparison']={'baseline_version':'allopt-engram-uva-v2-20261028','baseline':prior,'candidate':current,
        'output_gain_fraction':current['output_tps']/prior['output_tps']-1,
        'scope':'before/after separate resident services, same fixed input/routing/two rounds; no new target graph budgets; common optimization also requires vLLM tuning'}
    save('COMPLETE')
except BaseException as exc:
    state['error']=repr(exc);save('FAIL');traceback.print_exc();raise
finally:
    tmp=control/'single-point-mode.tmp';tmp.write_text(json.dumps(modes['reference']));tmp.replace(control/'single-point-mode.json')

