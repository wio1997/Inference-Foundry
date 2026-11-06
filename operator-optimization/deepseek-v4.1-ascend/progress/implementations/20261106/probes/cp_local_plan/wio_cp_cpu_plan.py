"""Private CPU plan cache; each builder retains its own device publication."""
from collections import Counter
import torch

CACHE_KEY = '_wio_cp_cpu_plan_v1'
COUNTS = Counter()


def lookup(builder, common, seq_cpu, kwargs):
    shared = kwargs.get('common_v41_batch_metadata')
    if not isinstance(shared, dict):
        COUNTS['fallback_scope'] += 1
        return None, None
    qcpu = common.query_start_loc_cpu
    n, tokens, actual = common.num_reqs, common.num_input_tokens, common.num_actual_tokens
    actual_reqs = kwargs.get('num_actual_reqs', n)
    if (any(type(x) is not int for x in (n, tokens, actual, actual_reqs))
            or not (0 <= actual <= tokens and 0 <= actual_reqs <= n)):
        COUNTS['fallback_counts'] += 1
        return None, None
    if not all(isinstance(t, torch.Tensor) and t.device.type == 'cpu'
               and t.dtype in (torch.int32, torch.int64) and t.ndim == 1 for t in (qcpu, seq_cpu)):
        COUNTS['fallback_representation'] += 1
        return None, None
    if tuple(qcpu.shape) != (n+1,) or tuple(seq_cpu.shape) != (n,):
        COUNTS['fallback_shape'] += 1
        return None, None
    partition = tuple(builder._local_token_range(tokens))
    key = (n, tokens, actual, actual_reqs, bool(getattr(common, 'causal', True)),
           qcpu.dtype, seq_cpu.dtype, builder._cp_query_start_loc.dtype, partition)
    cached = shared.get(CACHE_KEY)
    owner = shared.get('batch')
    if (cached is not None and owner is not None and cached['batch_owner'] is owner
            and cached['key'] == key and torch.equal(qcpu, cached['input_qsl'])
            and torch.equal(seq_cpu, cached['input_lengths'])):
        COUNTS['hit'] += 1
        return cached, None
    COUNTS['miss'] += 1
    # The first producer remains original. Snapshot inputs before it executes;
    # store() checks that they stayed unchanged before publishing a plan.
    return None, (shared, key, qcpu, seq_cpu, qcpu.clone(), seq_cpu.clone())


def store(scope, values, max_query_len, max_seq_len):
    if scope is None:
        return
    shared, key, qcpu, seq_cpu, q_snapshot, seq_snapshot = scope
    owner = shared.get('batch')
    if (owner is None or tuple(values[:4]) != key[-1]
            or not torch.equal(qcpu, q_snapshot) or not torch.equal(seq_cpu, seq_snapshot)):
        COUNTS['store_declined'] += 1
        return
    start, end, per_rank, padded, qsl, lengths, actual_start, actual_end = values
    # Never cache the CPU tensors exposed by the first public batch metadata.
    # Subsequent hits require that original batch owner to remain present.
    private_qsl, private_lengths = qsl.clone(), lengths.clone()
    shared[CACHE_KEY] = {
        'key': key, 'batch_owner': owner,
        'input_qsl': q_snapshot, 'input_lengths': seq_snapshot,
        'values': (start, end, per_rank, padded, private_qsl, private_lengths, actual_start, actual_end),
        'pinned_qsl': private_qsl.pin_memory(),
        'max_query_len': max_query_len, 'max_seq_len': max_seq_len,
    }
    COUNTS['stored'] += 1
