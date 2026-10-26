from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator
gsm8k_reader_cfg = dict(input_columns=["question"], output_column="answer")
gsm8k_infer_cfg = dict(prompt_template=dict(type=PromptTemplate, template="{question}"), retriever=dict(type=ZeroRetriever), inferencer=dict(type=GenInferencer))
gsm8k_eval_cfg = dict(evaluator=dict(type=Gsm8kEvaluator), pred_role="BOT", pred_postprocessor=dict(type=gsm8k_postprocess), dataset_postprocessor=dict(type=gsm8k_dataset_postprocess))
gsm8k_datasets = [dict(abbr="gsm8k", type=GSM8KDataset, path='/data/tiankuan/aisbench/warm_w2/codex_4p2d_20261008/diagnostic_search/diag_c32_0adbe7318a7e/runs/diag_c32_0adbe7318a7e/data/gsm8k', reader_cfg=gsm8k_reader_cfg, infer_cfg=gsm8k_infer_cfg, eval_cfg=gsm8k_eval_cfg)]
