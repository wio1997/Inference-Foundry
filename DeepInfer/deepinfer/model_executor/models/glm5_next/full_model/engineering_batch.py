# SPDX-License-Identifier: Apache-2.0
"""Resident-model engineering screening through the native DeepInfer engine.

Preserves raw output. Separates the checkpoint's explicit </think> protocol;
no answer extraction, JSON repair, or expected-answer access is performed.
"""
import json
import time
from pathlib import Path


def run_engineering_batch(llm, worker, args, out):
    import numpy as np
    from transformers import AutoTokenizer
    from deepinfer._C import engine
    from deepinfer.sampling_params import SamplingParams
    from deepinfer.model_executor.models.glm5_next.scheduler_eos import configure_scheduler_eos

    rows = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    if len(rows) != 24 or len({row['case_id'] for row in rows}) != 24:
        raise ValueError('frozen engineering batch requires24 unique inputs')
    for row in rows:
        ids = row['token_ids']
        if (not isinstance(ids, list) or not ids or any(type(i) is not int for i in ids)
                or len(ids) + args.max_new_tokens > args.max_context):
            raise ValueError('engineering input exceeds explicit context')
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    generation = json.loads((Path(args.model) / 'generation_config.json').read_text())
    eos = generation['eos_token_id']
    eos = [eos] if isinstance(eos, int) else eos
    configure_scheduler_eos(llm.scheduler, generation,
                            tokenizer_eos_id=tokenizer.eos_token_id,
                            vocab_size=worker._glm_worker_runtime.backend.vocab_size)
    results = []
    output_file = out / 'outputs.jsonl'
    raw_file = out / 'raw-requests.jsonl'
    with output_file.open('w') as scored, raw_file.open('w') as raw_log:
        for row in rows:
            request_id = row['case_id']
            params = SamplingParams(temperature=0, max_tokens=args.max_new_tokens,
                                    ignore_eos=False, seq_seed=0)
            started = time.monotonic()
            first = None
            ids = []
            finish_reason = None
            llm.add_request('offline', request_id, params, np.array(row['token_ids'], dtype=np.int32))
            for step in range(args.max_context + args.max_new_tokens + 64):
                if not llm.has_unfinished_requests():
                    break
                for ptr in llm.step():
                    ptr = int(ptr)
                    token = int(engine.RequestOutput.get_token_id(ptr))
                    if token >= 0:
                        ids.append(token)
                        if first is None:
                            first = time.monotonic() - started
                    if engine.RequestOutput.get_finished(ptr):
                        finish_reason = engine.RequestOutput.get_finish_reason(ptr)
                (out / 'progress.json').write_text(json.dumps({
                    'case_id': request_id, 'completed_cases': len(results),
                    'generated_tokens': len(ids), 'step': step,
                    'scope': 'engineering_only;not_formal_quality'}) + '\n')
            else:
                raise RuntimeError('native engineering request iteration budget exceeded')
            if worker._glm_worker_runtime.backend._owner is not None:
                raise RuntimeError('terminal engineering request retained cache owner')
            raw_text = tokenizer.decode(ids, skip_special_tokens=False)
            # Remove only the actual final EOS token from the textual payload.
            payload_ids = ids[:-1] if ids and ids[-1] in eos else ids
            payload = tokenizer.decode(payload_ids, skip_special_tokens=False)
            reasoning, delimiter, content = payload.partition('</think>')
            if not delimiter:
                content = ''
            result = {'case_id': request_id, 'prompt_tokens': len(row['token_ids']),
                      'generated_token_ids': ids, 'raw_completion': raw_text,
                      'reasoning_content': reasoning, 'content': content,
                      'output': content, 'reasoning_closed': bool(delimiter),
                      'finish_reason': finish_reason, 'ended_with_checkpoint_eos': bool(ids and ids[-1] in eos),
                      'generation_cap': args.max_new_tokens,
                      'driver_owner_released': True, 'elapsed_seconds': time.monotonic() - started,
                      'first_output_seconds': first,
                      'scope': 'engineering_only;content_after_explicit_think_delimiter;not_formal_quality'}
            raw_log.write(json.dumps(result, ensure_ascii=False) + '\n'); raw_log.flush()
            scored.write(json.dumps({'case_id': request_id, 'output': content}, ensure_ascii=False) + '\n'); scored.flush()
            results.append({'case_id': request_id, 'generated_tokens': len(ids),
                            'finish_reason': finish_reason, 'reasoning_closed': bool(delimiter),
                            'driver_owner_released': True})
            print(json.dumps({'completed_case': request_id, 'count': len(results),
                              'generated_tokens': len(ids), 'reasoning_closed': bool(delimiter)}), flush=True)
    result = {'status': 'executed', 'scope': '24case_engineering_only;unscored',
              'requests': results, 'driver_owner_released': True, 'worker_ranks': args.world_size,
              'max_context_tokens': args.max_context, 'max_step_tokens': args.max_step,
              'generation_cap': args.max_new_tokens, 'checkpoint_eos_ids': eos}
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
