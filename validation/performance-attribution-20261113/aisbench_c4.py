"""Shared AISBench config; environment changes only endpoint and concurrency.
Run from the pinned AISBench environment with MODELDIR pointing at readonly weights.
Synthetic input generation is performed by AISBench with the same tokenizer.
"""
from ais_bench.benchmark.datasets import CustomDataset
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator
from ais_bench.benchmark.models import VLLMCustomAPI

models = [dict(
    attr='service', type=VLLMCustomAPI, abbr='glm53flash-tp8',
    path='/task/aisbench-tp8/tokenizer-only', model='glm53flash', stream=True,
    request_rate=0, use_timestamp=False, retry=1, api_key='',
    host_ip='127.0.0.1', host_port=18080,
    max_out_len=256, batch_size=4, trust_remote_code=False,
    generation_kwargs=dict(temperature=0, top_p=1, seed=0, ignore_eos=True),
)]
datasets = [dict(
    abbr='frozen-1024-in-256-out', type=CustomDataset,
    path='/task/aisbench-tp8/frozen-1024-32.jsonl', meta_path='',
    reader_cfg=dict(input_columns=['question', 'max_out_len'], output_column='answer'),
    infer_cfg=dict(prompt_template=dict(type=PromptTemplate, template='{question}'),
                   retriever=dict(type=ZeroRetriever), inferencer=dict(type=GenInferencer)),
    eval_cfg=dict(evaluator=dict(type=AccEvaluator), pred_role='BOT'),
)]
