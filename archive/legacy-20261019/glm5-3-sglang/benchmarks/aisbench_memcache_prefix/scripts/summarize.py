#!/usr/bin/env python3
"""Audit raw requests and obtain both AISBench calculators from one trace."""
import argparse
import hashlib
import json
from backend_source_audit import verify_backend_source
from pathlib import Path

def dump(path, value):
    def convert(v):
        if hasattr(v, 'tolist'): return v.tolist()
        if hasattr(v, 'item'): return v.item()
        raise TypeError(type(v).__name__)
    path.write_text(json.dumps(value, indent=2, default=convert) + '\n')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='/work/aisbench-glm53-run')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--cc', required=True, type=int)
    a = p.parse_args()
    root = Path(a.root); run = Path(a.run_dir); out=root/'reports'/f'cc{a.cc}'; out.mkdir(parents=True,exist_ok=True)
    manifests = [json.loads(s) for s in (root/'dataset-manifest.jsonl').read_text().splitlines()]
    expected = {r['row']:r for r in manifests if r['cc']==a.cc}
    files = list((run/'performances').glob(f'*/long80k_cc{a.cc}_details.jsonl'))
    assert len(files)==1, files
    rows = [json.loads(s) for s in files[0].read_text().splitlines()]
    assert len(expected)==64 and set(expected)==set(range(64))
    errors=[]; seen_ids=set(); seen_hashes=set()
    for r in rows:
        rid=r.get('id'); seen_ids.add(rid)
        if not r.get('success'): errors.append({'id':rid,'error':r.get('error_info','failed')}); continue
        messages=r.get('input'); content=None
        if isinstance(messages,str): content=messages
        elif isinstance(messages,list) and len(messages)==1:
            content=messages[0].get('content',messages[0].get('prompt'))
        digest=hashlib.sha256(content.encode()).hexdigest() if isinstance(content,str) else None
        if digest in seen_hashes: errors.append({'id':rid,'error':'duplicate prompt hash'})
        seen_hashes.add(digest)
        if rid not in expected or digest != expected[rid]['prompt_sha256']:
            errors.append({'id':rid,'error':'actual request differs from audited dataset','hash':digest})
        extra=r.get('extra_details_data') or {}
        usage=extra.get('native_usage') or {}
        cached=(usage.get('prompt_tokens_details') or {}).get('cached_tokens')
        if cached != 74368:
            errors.append({'id':rid,'error':'actual per-request cache hit mismatch','cached_tokens':cached,'expected':74368})
        sources=extra.get('cached_tokens_details')
        if sources is None:
            errors.append({'id':rid,'error':'cache layer breakdown missing'})
        else:
            try:verify_backend_source(root,'cc'+str(a.cc),sources)
            except AssertionError as e:errors.append({'id':rid,'error':'actual L3 source audit failed','detail':str(e),'sources':sources})
        if r.get('input_tokens')!=80000 or r.get('output_tokens')!=600:
            errors.append({'id':rid,'error':'server usage length mismatch','input_tokens':r.get('input_tokens'),'output_tokens':r.get('output_tokens')})
    if len(rows)!=len(expected) or seen_ids != set(expected):
        errors.append({'error':'request count or IDs mismatch','actual':len(rows),'expected':len(expected)})
    audit={'valid':not errors,'client_concurrency':a.cc,'total':len(rows),
        'success':sum(bool(r.get('success')) for r in rows),'errors':errors,
        'unique_prompt_hashes':len(seen_hashes),'raw_file':str(files[0]),
        'raw_sha256':hashlib.sha256(files[0].read_bytes()).hexdigest()}
    audit['cache_hits']=[{'id':r.get('id'),'cached_tokens':((r.get('extra_details_data')or{}).get('native_usage',{}).get('prompt_tokens_details')or{}).get('cached_tokens'),'sources':(r.get('extra_details_data')or{}).get('cached_tokens_details')}for r in rows]
    audit['target_ratio']=.93
    audit['expected_page_aligned_ratio']=.9296
    dump(out/'request-audit.json',audit)
    if errors: raise RuntimeError(f'Invalid benchmark: see {out}/request-audit.json')
    from mmengine import Config
    from ais_bench.benchmark.summarizers import DefaultPerfSummarizer
    from ais_bench.benchmark.calculators import DefaultPerfMetricCalculator, StablePerfMetricCalculator
    configs=sorted((run/'configs').glob('*.py')); assert configs
    cfg=Config.fromfile(str(configs[0]),format_python_code=False)
    cfg.work_dir=str(run)
    summarizer=DefaultPerfSummarizer(cfg,dict(type=DefaultPerfMetricCalculator))
    groups=list(summarizer.dataset_groups.values()); assert len(groups)==1 and len(cfg.models)==1
    details=summarizer._load_details_perf_data(cfg.models[0],groups[0])
    # Store derived absolute per-request timings as an independent audit artifact.
    trace=[]
    for i in range(len(details['id'])):
        trace.append({k:details[k][i] for k in ('id','start_time','end_time','ttft','tpot','latency','input_tokens','output_tokens','success')})
    dump(out/'request-timing.json',trace)
    result={'valid':True,'client_concurrency':a.cc,'requests':len(rows),
        'same_raw_trace_for_both_views':True,'request_audit':audit}
    for name, cls in [('whole',DefaultPerfMetricCalculator),('steady',StablePerfMetricCalculator)]:
        calc=cls(stats_list=['Average','Min','Max','Median','P90','P99'])
        try:
            calc._init_datas(details,a.cc)
            if name=='steady':assert calc.stage_section[1]>calc.stage_section[0], 'Zero-length native steady window'
            calc.calculate()
            record={'valid':True,'common':calc.get_common_res(),'request_metrics':calc.metrics,
                'request_ids':{k:[details['id'][i] for i in ids] for k,ids in calc.stage_dict.items()}}
            if name=='steady':
                record['absolute_window_seconds']=calc.stage_section
                record['definition']='AISBench: second attainment of target client concurrency to last attainment; selects requests whose START lies within window, even if END lies outside. Client concurrency includes queued requests.'
            dump(out/f'{name}.json',record); calc.save_performance(str(out/f'{name}.csv'))
            result[name]=record
        except Exception as e:
            result[name]={'valid':False,'error':str(e)}
            dump(out/f'{name}.json',result[name])
            if name=='whole':result['valid']=False
    from decode_plateau import analyze
    result['server_decode_plateau']=analyze(rows,root/f'logs/load-cc{a.cc}.jsonl')
    dump(out/'server-decode-plateau.json',result['server_decode_plateau'])
    result['backend_source_proof']=verify_backend_source(root,'cc'+str(a.cc),(rows[0].get('extra_details_data')or{}).get('cached_tokens_details'))
    result['memcache_host_io_mode']='native_direct_pointer_api'
    result['sdk_internal_temporary_hbm_staging']=True
    result['measurement_scope']='formal_warm_requests_only'
    result['cohort']=json.loads((root/'dataset-audit.json').read_text())
    result['prewarm_time_included_in_throughput']=False
    result['prewarm_tokens_included_in_throughput']=False
    warm=json.loads((root/f'warmups/cc{a.cc}.complete.json').read_text())
    result['prewarm_separate']=dict(requests=warm['requests'],prefix_tokens_each=74400,duration_s=warm['elapsed_s'])
    result['cold_comparison_requested']=False
    result['logical_input_tokens_per_request']=80000
    result['cached_input_tokens_per_request']=74368
    result['remaining_prefill_tokens_per_request']=5632
    result['native_direct_note']='Official paired1.3 SDK and native SGLang pointer API; no added CPU staging adapter. SDK SDMA HOST/poolDRAM uses temporaryHBM internally; not a single-transfer path. Wire bytes and bandwidth need separate measurements.'
    result['sdk_note']='MemCache/MemFabric paired1.3 development. User requests warm-only results; old cold artifacts remain archived and are not part of this performance report.'
    result['protocol']='64 independent74400-token prefixes prewarmed; storage validated and L1L2 flushed; then same64 full80000/600 requests timed. All prewarm and smoke work excluded from formal throughput.'
    result['steady_note']='NativeAIS steady may be unavailable for one64-request wave; server decode plateau is a separate sampled occupancy/received-token view.'
    dump(out/'result.json',result); print(json.dumps({'report':str(out),'valid':result['valid']}))
    assert result['valid'], 'Invalid whole trace; preserve missing native steady as N/A, never relabel whole'

if __name__=='__main__': main()
