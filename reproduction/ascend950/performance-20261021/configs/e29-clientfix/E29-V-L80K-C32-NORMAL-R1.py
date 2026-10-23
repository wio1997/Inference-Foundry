from e29_ais_model import E29MappedAPI
from ais_bench.benchmark.datasets import CustomDataset
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local import LocalRunner
from ais_bench.benchmark.tasks import OpenICLApiInferTask
root = "/work/performance-20261021"
framework = "vllm"
run_id = "E29-V-L80K-C32-NORMAL-R1"
datasets = [dict(abbr="E29-L80K-C32",type=CustomDataset,path=root+"/data/workloads/E29-L80K-C32/questions.jsonl",
 reader_cfg=dict(input_columns=["question"],output_column="answer"),
 infer_cfg=dict(prompt_template=dict(type=PromptTemplate,template="{question}"),
 retriever=dict(type=ZeroRetriever),inferencer=dict(type=GenInferencer)),
 eval_cfg=dict(evaluator=dict(type=AccEvaluator),pred_role="BOT"),meta_path="")]
models = [dict(attr="service",type=E29MappedAPI,abbr=run_id,framework=framework,run_id=run_id,
 cache_namespace="e29-v-l80k-c32-normal",
 cache_action="",
 corpus=root+"/data/workloads/E29-L80K-C32/corpus.jsonl",
 model="DeepSeek-V4.1-Flash-1017",path="/data/wio/DeepSeek-V4.1-Flash-1017",
 inference_url="http://127.0.0.1:18122/v1/completions",
 stream=True,request_rate=0,retry=1,max_out_len=600,batch_size=32,
 generation_kwargs=dict(temperature=0.0,ignore_eos=True,seed=3000))]
infer = dict(partitioner=dict(type=NaivePartitioner),runner=dict(type=LocalRunner,max_num_workers=1,
 task=dict(type=OpenICLApiInferTask)))
work_dir = root+"/runs/e29-ais/"+run_id
