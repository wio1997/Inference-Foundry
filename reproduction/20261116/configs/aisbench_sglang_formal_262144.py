from ais_bench.benchmark.models import VLLMCustomAPI
from ais_bench.benchmark.datasets import CustomDataset
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator
from ais_bench.benchmark.summarizers import DefaultPerfSummarizer
from ais_bench.benchmark.calculators import DefaultPerfMetricCalculator
models=[dict(attr='service',type=VLLMCustomAPI,abbr='sglang-a3-dsv41-1017-w8a8-engram-dram',path='/task/tokenizer',model='Eco-Tech/DeepSeek-V4.1-Flash-1017-w8a8',stream=True,request_rate=0,retry=1,host_ip='127.0.0.1',host_port=18167,max_out_len=1024,batch_size=16,trust_remote_code=True,generation_kwargs=dict(temperature=0.0,ignore_eos=True,min_tokens=1024))]
datasets=[dict(abbr='formal_input_262144_output1024',type=CustomDataset,path='/task/datasets/formal_input_262144.jsonl',meta_path='',reader_cfg=dict(input_columns=['question','max_out_len'],output_column='answer'),infer_cfg=dict(prompt_template=dict(type=PromptTemplate,template='{question}'),retriever=dict(type=ZeroRetriever),inferencer=dict(type=GenInferencer)),eval_cfg=dict(evaluator=dict(type=AccEvaluator),pred_role='BOT'))]
summarizer=dict(attr='performance',type=DefaultPerfSummarizer,calculator=dict(type=DefaultPerfMetricCalculator,stats_list=['Average','Min','Max','Median','P75','P90','P99']))
