from e27_ais_model import E27MappedAPI
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
run_id = "E27-V-L32K-C128-HOT-W1"
datasets = [dict(abbr="E27-L32K-C128",type=CustomDataset,path=root+"/data/workloads/E27-L32K-C128/questions.jsonl",
 reader_cfg=dict(input_columns=["question"],output_column="answer"),
 infer_cfg=dict(prompt_template=dict(type=PromptTemplate,template="{question}"),
 retriever=dict(type=ZeroRetriever),inferencer=dict(type=GenInferencer)),
 eval_cfg=dict(evaluator=dict(type=AccEvaluator),pred_role="BOT"),meta_path="")]
models = [dict(attr="service",type=E27MappedAPI,abbr=run_id,framework=framework,run_id=run_id,
 cache_namespace="e27-v-l32k-c128-hot",
 cache_action="",
 corpus=root+"/data/workloads/E27-L32K-C128/v-prefix.jsonl",
 model="DeepSeek-V4.1-Flash-1017",path="/data/wio/DeepSeek-V4.1-Flash-1017",
 inference_url="http://127.0.0.1:18122/v1/completions",
 stream=True,request_rate=0,retry=1,max_out_len=1,batch_size=128,
 generation_kwargs=dict(temperature=0.0,ignore_eos=True,seed=3000))]
infer = dict(partitioner=dict(type=NaivePartitioner),runner=dict(type=LocalRunner,max_num_workers=1,
 task=dict(type=OpenICLApiInferTask)))
work_dir = root+"/runs/e27-ais/"+run_id
