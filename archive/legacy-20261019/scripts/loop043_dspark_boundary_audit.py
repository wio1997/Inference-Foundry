#!/usr/bin/env python3
"""Frozen DSpark7 replay-boundary/source and timing audit."""
import json,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1]
borrowed=Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend')
proposer=borrowed/'spec_decode/llm_base_proposer.py'
model=borrowed/'models/deepseek_v4_dspark.py'
src=proposer.read_text();model_src=model.read_text()
assert 'if self.method == "dspark":' in src
assert 'for idx in range(self.num_speculative_tokens):' in src
assert 'logits[:, idx].add_(logits_bias)' in src
assert 'draft_token_ids[:, idx + 1].copy_(logits[:, idx].argmax(dim=-1))' in src
assert 'self.markov_w2 = ReplicatedLinear(' in model_src
run140=json.loads((root/'evidence/20260925_loop042_phase/run140/subphase_analysis.json').read_text())
run98=json.loads((root/'evidence/20260925_loop039_priority/run119/steady_priority.json').read_text())
rt={k:v['median'] for k,v in run98['runtime_stage_ms'].items()}
device_sum=sum(rt.values())
out={'run':'run141','product':'DeepSeek V4 Flash W4A8, 8x910B3, DP1TP8, c12, DSpark7',
     'source_refs':{'proposer':str(proposer),'proposer_sha256':hashlib.sha256(proposer.read_bytes()).hexdigest(),
                    'model':str(model),'model_sha256':hashlib.sha256(model.read_bytes()).hexdigest()},
     'actual_path':'AscendDSparkProposer inherits eager llm_base_proposer._propose/_run_merged_draft; DirectDSparkHandoff rejects use_cuda_graph. Step3p5 is not this path.',
     'segments':[
       {'name':'context_kv_prepare','boundary':'_run_merged_draft build_model_inputs_first_pass', 'state':'dynamic target hidden, token/position and KV group slot writes; not pure replay'},
       {'name':'draft_forward','boundary':'self.model(**model_kwargs)', 'state':'three borrowed DSpark layers, attention and KV state; needs full same-state validation'},
       {'name':'lmhead','boundary':'gather sample_hidden_states then model.compute_logits', 'state':'depends on target and draft forward outputs; TP collectives may occur'},
       {'name':'markov_tail','boundary':'seed buffer copy then seven sequential markov_embed -> markov_bias -> logits.add_ -> argmax -> next token', 'state':'fixed B12 K7; no attention metadata/KV writes or inter-rank communication in replicated Markov head; logits are mutated in place and must be refreshed on every replay'}],
     'timing':{'run140_runnable_host_thread_cpu_ms':run140['summary']['calls']['_runnable']['thread_cpu_ms']['median'],
               'run140_metadata_host_thread_cpu_ms':run140['summary']['calls']['build_draft_attn_metadata']['thread_cpu_ms']['median'],
               'run140_latest_rank_span_mean_ms':run140['summary']['latest_rank_proposer_end_span_mean_ms'],
               'run98_device_stage_medians_ms':rt,'run98_sum_of_stage_medians_ms':device_sum},
     'inference':'Separate cohorts and asynchronous overlap prevent subtracting Host CPU from cycle time or treating the stage-sum as a strict bound. The fixed Markov tail is the smallest correctness-contained replay candidate, but its exposed fraction is unmeasured. Before implementation, Run142 must split proposer into context-KV, model, LMHead and Markov tail with Host wall/thread CPU plus NPU events and steady cadence.',
     'review':{'requested_model':'gpt-6-astra high','actual_model_verified':False,'role':'read-only architecture challenge; Sol decision remains authoritative'},
     'next_action':'Run142 legal eight-rank bounded subphase and NPU-event capture at four actual proposer boundaries, no device sync in timed path.'}
p=root/'evidence/20260925_loop043_dspark/run141/boundary_audit.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'device_sum':device_sum,'segments':out['segments'],'next_action':out['next_action']},indent=2))
