#!/usr/bin/env python3
from bench_profile import PROFILE
import argparse
from pathlib import Path
from urllib.parse import urlsplit

TEMPLATE = '''from cache_report_model import CacheReportAPIChat
from ais_bench.benchmark.datasets import GSM8KDataset
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.summarizers import DefaultPerfSummarizer
from ais_bench.benchmark.calculators import DefaultPerfMetricCalculator

models = [dict(attr='service', type=CacheReportAPIChat,
    abbr='glm53-warm93-cc{cc}', path={model_path!r}, model={model_name!r},
    host_ip={host!r}, host_port={port}, stream=True, trust_remote_code=True,
    # In this installed AISBench, retry is TOTAL attempts (range(self.retry)).
    # One attempt therefore disables retry; zero would send no requests.
    max_out_len=600, batch_size={cc}, request_rate=0, retry=1,
    generation_kwargs=dict(temperature=0, ignore_eos=True,
        return_cached_tokens_details=True, stream_options=dict(include_usage=True, continuous_usage_stats=True), chat_template_kwargs=dict(enable_thinking=False)))]
datasets = [dict(abbr='long80k_cc{cc}', type=GSM8KDataset,
    path={dataset!r}, reader_cfg=dict(input_columns=['question'], output_column='answer'),
    infer_cfg=dict(prompt_template=dict(type=PromptTemplate, template='{{question}}'),
        retriever=dict(type=ZeroRetriever), inferencer=dict(type=GenInferencer)))]
summarizer = dict(attr='performance', type=DefaultPerfSummarizer,
    calculator=dict(type=DefaultPerfMetricCalculator,
        stats_list=['Average','Min','Max','Median','P90','P99']))
'''

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/work/aisbench-glm53-run')
    p.add_argument('--cc', type=int, choices=[16,32,64], required=True)
    p.add_argument('--host', default=urlsplit(PROFILE['router_url']).hostname)
    p.add_argument('--port', type=int, default=urlsplit(PROFILE['router_url']).port)
    a = p.parse_args()
    root = Path(a.root); (root/'configs').mkdir(parents=True, exist_ok=True)
    cfg = root/'configs'/f'cc{a.cc}.py'
    cfg.write_text(TEMPLATE.format(cc=a.cc,host=a.host,port=a.port,model_path=PROFILE['model_path'],model_name=PROFILE['model_name'],dataset=str(root/'datasets'/f'cc{a.cc}')))
    print(cfg)

if __name__ == '__main__': main()
