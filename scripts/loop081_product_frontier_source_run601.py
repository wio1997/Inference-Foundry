#!/usr/bin/env python3
"""Pin current Product handoff and delivery hook sites before live acquisition."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
VLLM = Path('/data/wio/vllm_ascend_26/framework/vllm/vllm')
ASC = Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend')
OUT = ROOT / 'evidence/20260928_loop081_bound/run601'

SITES = {
    'runner': (
        ASC / 'worker/model_runner_v1.py',
        [
            'def execute_model(',
            '_extreme_runtime = build_extreme_runtime(',
            'initial_output_counts=[0] * _cfg.batch_size,',
            'self._extreme_serving_output = ModelRunnerOutput(',
            'hidden_states = self._model_forward(',
            'def propose_draft_token_ids(sampled_token_ids):',
            'self._copy_draft_token_ids_to_cpu(scheduler_output)',
            'self.draft_token_ids_copy_stream.wait_stream(default_stream)',
        ],
    ),
    'draft_base': (
        ASC / 'spec_decode/llm_base_proposer.py',
        [
            'def _propose(',
            'target_hidden_states = self.model.combine_hidden_states(target_hidden_states)',
            'num_tokens, token_indices_to_sample, common_attn_metadata, long_seq_args = self.set_inputs_first_pass(',
            'draft_token_ids = run_draft()',
        ],
    ),
    'dspark': (
        ASC / 'spec_decode/dspark_proposer.py',
        [
            'def set_inputs_first_pass(',
            'self._dspark_seed_buffer[:n].copy_(next_token_ids)',
            'self._dflash_hidden_states[: self._dflash_num_context] = target_hidden_states[: self._dflash_num_context]',
            'copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid[1,](',
        ],
    ),
    'scheduler': (
        VLLM / 'v1/core/sched/scheduler.py',
        ['req.num_output_tokens + req.num_output_placeholders'],
    ),
    'kvdelivery': (
        ASC / 'patch/platform/patch_kv_delivery_preemption.py',
        [
            'def update_from_output(',
            'num_output_tokens_before = len(request._output_token_ids)',
            'if model_runner_output.extreme_bulk_output:',
            'KVDeliveryScheduler._update_request_with_output(',
        ],
    ),
    'chat_serving': (
        VLLM / 'entrypoints/openai/chat_completion/serving.py',
        [
            'async def chat_completion_stream_generator(',
            'previous_num_tokens[i] += len(output.token_ids)',
            'yield f"data: {data}\\n\\n"',
            'yield "data: [DONE]\\n\\n"',
        ],
    ),
    'bootstrap': (
        ROOT / 'bootstrap/vllm_extreme_handoff.py',
        [
            'def build_extreme_runtime(',
            'state = FixedDecodeState.bind(',
            'return ExtremeDecodeRuntime(',
        ],
    ),
    'serving': (
        ROOT / 'runtime/fixed_serving.py',
        ['def run(self) -> FixedCohortOutput:'],
    ),
    'client': (
        ROOT / 'scripts/loop080_frontier_client.py',
        [
            'async with session.post(args.url, json=request_body) as response:',
            'fragment_recv_ns = time.monotonic_ns()',
            '"sse_done_monotonic_ns": done_ns,',
        ],
    ),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {}
    for key, (path, needles) in SITES.items():
        raw = path.read_bytes()
        lines = raw.decode().splitlines()
        hits = {}
        for needle in needles:
            where = [i + 1 for i, line in enumerate(lines) if needle in line]
            if not where:
                raise AssertionError((key, needle, 'missing'))
            hits[needle] = where
        sources[key] = {'path': str(path), 'sha256': sha(path), 'anchors': hits}
    evidence = {}
    for rel in (
        'evidence/20260926_loop059_boundary/run239/phase_analysis.json',
        'evidence/20260926_loop074_refill/run341/phase_analysis.json',
        'evidence/20260926_loop074_refill/run341/scheduler_analysis.json',
        'evidence/20260928_loop081_bound/run597/live/b/basis_admission.json',
        'evidence/20260928_loop081_bound/run600/host_envelope.json',
    ):
        path = ROOT / rel
        evidence[rel] = sha(path)
    result = {
        'status': 'source_site_gate_pass',
        'source_count': len(sources),
        'evidence_count': len(evidence),
        'sources': sources,
        'evidence_sha256': evidence,
        'scope': 'Anchor locations and exact source/evidence identity only. No async writer-ready, complete consumer DAG, measured overlap or numerical Bound.',
    }
    (OUT / 'source_gate.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'source_count': len(sources), 'evidence_count': len(evidence)}))


if __name__ == '__main__':
    main()
