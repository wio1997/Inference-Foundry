#!/usr/bin/env python3
"""Installed AISBench compatibility test with synthetic chunks; sends no HTTP."""
import argparse
import asyncio
import json
from pathlib import Path
import subprocess
import sys
from mmengine import Config
from cache_report_model import CacheReportAPIChat
from ais_bench.benchmark.models.output import RequestOutput

p=argparse.ArgumentParser();p.add_argument('--root',default='/work/aisbench-glm53-run');p.add_argument('--check-datasets',action='store_true');a=p.parse_args();root=Path(a.root)

async def check():
    for cc in [16,32,64]:
        subprocess.run([sys.executable,str(root/'scripts/build_config.py'),'--root',str(root),'--cc',str(cc)],check=True)
        cfg=Config.fromfile(str(root/f'configs/cc{cc}.py'),format_python_code=False)
        kw=dict(cfg.models[0]);kw.pop('type');kw.pop('attr');kw.pop('abbr');kw.pop('batch_size');kw.pop('request_rate')
        model=CacheReportAPIChat(**kw);out=RequestOutput(perf_mode=True)
        body=await model.get_request_body('offline synthetic',600,out)
        assert body['stream_options']==dict(include_usage=True,continuous_usage_stats=True)
        assert body['max_tokens']==600 and body['return_cached_tokens_details']
        for n in [0,1,2,600]:
            await out.record_time_point()
            usage=dict(prompt_tokens=80000,completion_tokens=n,prompt_tokens_details=dict(cached_tokens=74368))
            await model.parse_stream_response(dict(usage=usage,choices=[],sglext=dict(cached_tokens_details=dict(storage=74368,storage_backend='npu_memcache'))),out)
        assert out.output_tokens==600 and out.input_tokens==80000
        assert [v['completion_tokens']for v in out.extra_details_data['usage_events']]==[0,1,2,600]
        assert out.extra_details_data['cached_tokens_details']['storage']==74368
    print(json.dumps(dict(phase='OFFLINE_AISBENCH_COMPATIBILITY_PASS',http_requests=0,concurrencies=[16,32,64])))
asyncio.run(check())

if a.check_datasets:
    import hashlib
    from ais_bench.benchmark.utils.config.build import build_dataset_from_cfg
    data=json.loads((root/'dataset-audit.json').read_text());total=0
    assert data['formal_requests']==448 and len(data['cohorts'])==7
    for group in data['cohorts']:
        leaf=Path(group['root']);cc=group['cc'];train=leaf/f'datasets/cc{cc}/train.jsonl'
        assert train.exists() and train.stat().st_size==0
        subprocess.run([sys.executable,str(root/'scripts/build_config.py'),'--root',str(leaf),'--cc',str(cc)],check=True)
        cfg=Config.fromfile(str(leaf/f'configs/cc{cc}.py'),format_python_code=False)
        dataset=build_dataset_from_cfg(cfg.datasets[0])
        assert len(dataset.dataset['train'])==0 and len(dataset.dataset['test'])==64
        expected={x['row']:x for x in map(json.loads,(leaf/'dataset-manifest.jsonl').read_text().splitlines())}
        for i,row in enumerate(dataset.dataset['test']):
            assert hashlib.sha256(row['question'].encode()).hexdigest()==expected[i]['prompt_sha256']
        total+=len(dataset.dataset['test'])
    assert total==448
    print(json.dumps(dict(phase='OFFLINE_ALL_COHORT_DATASET_LOAD_PASS',http_requests=0,requests=total,groups=7,train_examples=0)))
