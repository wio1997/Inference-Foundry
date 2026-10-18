"""Select the existing native eager path for padded P-only prefill batches."""
import functools
import inspect
import json
import os
from pathlib import Path
import time


def install(ns, root=None):
    assert os.environ.get('GLM53_P_FIRST_DRAFT_EXPORT') == '1', 'Producer-only overlay'
    Runner = ns['NPUModelRunner']
    Mode = ns['CUDAGraphMode']
    root = Path('/out/prefill-padding-probe') if root is None else Path(root)
    root.mkdir(exist_ok=True)
    original = Runner._determine_batch_execution_and_padding
    signature = inspect.signature(original)
    records = 0

    @functools.wraps(original)
    def determine(self, *args, **kwargs):
        nonlocal records
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        native = original(*bound.args, **bound.kwargs)
        values = bound.arguments
        n = int(values['num_tokens'])
        batch = int(values['num_reqs'])
        ids = list(self.input_batch.req_ids)
        parallel = self.parallel_config
        real = bool(ids and len(ids) == batch and all(rid in self.requests for rid in ids))
        scheduled = [int(x) for x in values['num_scheduled_tokens_np'][:batch]] if real else []
        producer = bool(self.is_kv_producer and not self.is_kv_consumer)
        eligible = bool(
            real and producer and parallel.tensor_parallel_size == 16
            and parallel.data_parallel_size == 1 and parallel.pipeline_parallel_size == 1
            and parallel.decode_context_parallel_size == 1 and parallel.prefill_context_parallel_size == 1
            and not self.model_config.is_hybrid and self.num_spec_tokens == 1
            and native[0] == Mode.PIECEWISE and not native[2] and native[3] is None
            and all(x > self.uniform_decode_query_len for x in scheduled)
            and all(int(self.input_batch.num_computed_tokens_cpu[i]) < int(self.input_batch.num_prompt_tokens[i]) for i in range(batch))
            and all(self.requests[rid].sampling_params is not None and self.requests[rid].sampling_params.max_tokens == 1 for rid in ids)
            and native[1].num_tokens > self._pad_for_sequence_parallelism(n))
        requested = (root / 'enable').exists()
        selected = bool(eligible and requested)
        result = native
        if selected:
            # Call the unchanged native method, including its sequence alignment,
            # metadata construction, stats, and original force_eager handling.
            bound.arguments['force_eager'] = True
            result = original(*bound.args, **bound.kwargs)
            assert result[0] == Mode.NONE and result[1].num_tokens == self._pad_for_sequence_parallelism(n)
            assert not result[2] and result[3] is None
        if real and records < 4000 and not (root / 'disabled').exists():
            records += 1
            row = {'kind': 'prefill_padding_native_path', 'pid': os.getpid(), 'tp_rank': self.tp_rank,
                   'monotonic_ns': time.monotonic_ns(), 'request_ids': ids, 'num_real_tokens': n,
                   'num_reqs': batch, 'scheduled_per_request': scheduled,
                   'computed_before': [int(self.input_batch.num_computed_tokens_cpu[i]) for i in range(batch)],
                   'prompt_tokens': [int(self.input_batch.num_prompt_tokens[i]) for i in range(batch)],
                   'native_mode': str(native[0]), 'native_padded_tokens': int(native[1].num_tokens),
                   'selected_mode': str(result[0]), 'selected_padded_tokens': int(result[1].num_tokens),
                   'eligible': eligible, 'requested': requested, 'selected': selected,
                   'scope': 'Native metadata selection, no model forward/tensor copy or device-time claim'}
            with (root / ('rank-%d-%d.jsonl' % (self.tp_rank, os.getpid()))).open('a') as out:
                out.write(json.dumps(row, sort_keys=True) + '\n')
        return result

    Runner._determine_batch_execution_and_padding = determine
