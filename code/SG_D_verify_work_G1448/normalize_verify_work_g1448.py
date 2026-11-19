"""Convert observed target VERIFY moments only with a verified logical KV boundary."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


BOUNDARIES = ('logical_prefix', 'logical_endpoint', 'allocation_or_unverified')
GENERATION_MOMENTS = ('g_sq', 'g_sqsq', 'g_sqsk', 'g_sk')


def normalize_verify(step, kv_boundary):
    if step['mode'] != 'VERIFY' or kv_boundary not in BOUNDARIES:
        raise ValueError('Expected a target VERIFY step and an explicit context boundary')
    batch = step['batch_size']
    if type(batch) is not int or batch < 1:
        raise ValueError('A target VERIFY step requires a positive integer batch size')
    fields = step['actual_query_KV_aggregate_fields']
    output = {'forward_id': step.get('forward_id'), 'batch_size': batch, 'KV_boundary': kv_boundary,
              'logical_work_verified': False}
    if not fields:
        return {**output, 'status': 'missing_query_KV_moments'}
    if any(type(fields.get(name)) is not int for name in GENERATION_MOMENTS):
        raise ValueError('Incomplete target VERIFY generation moments')
    qsum, q2sum, qksum, ksum = (fields[name] for name in GENERATION_MOMENTS)
    width, remainder = divmod(qsum, batch)
    if width < 1 or remainder or q2sum != width * qsum or qksum != width * ksum or ksum < 0:
        raise ValueError('Moments do not describe a uniform-width target VERIFY step')
    output.update(query_positions=qsum, query_width_per_request=width, recorded_KV_length_sum=ksum)
    if kv_boundary == 'allocation_or_unverified':
        return {**output, 'status': 'logical_KV_boundary_unverified'}

    if kv_boundary == 'logical_endpoint':
        prefix_k = ksum - qsum
        pairs = qksum - (q2sum - qsum) // 2
    else:
        prefix_k = ksum
        pairs = qksum + (q2sum + qsum) // 2
    if prefix_k < 0 or pairs <= 0:
        raise ValueError('VERIFY moments contradict the declared logical KV boundary')
    return {**output, 'status': 'normalized', 'logical_work_verified': True,
            'logical_prefix_tokens_sum': prefix_k, 'logical_endpoint_tokens_sum': prefix_k + qsum,
            'uncompressed_causal_query_token_pairs': pairs}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('graph_analysis', type=Path)
    cli.add_argument('--boundary-by-span', type=Path,
                     help='Capture-SHA-bound report of independently verified KV boundaries')
    args = cli.parse_args()
    raw = args.graph_analysis.read_bytes()
    source_sha = sha256(raw).hexdigest()
    src = json.loads(raw)
    verify_steps = [step for step in src['spans'] if step['mode'] == 'VERIFY']
    boundaries = {}
    if args.boundary_by_span:
        evidence = json.loads(args.boundary_by_span.read_bytes())
        if evidence['graph_analysis_SHA256'] != source_sha:
            raise ValueError('The KV boundary report belongs to a different capture analysis')
        boundaries = evidence['by_span']
    span_ids = {str(step['id']) for step in verify_steps}
    if len(span_ids) != len(verify_steps) or set(boundaries) - span_ids:
        raise ValueError('The boundary map must refer to unique target VERIFY spans in this capture')
    steps = [{'span_id': step['id'], **normalize_verify(
        step, boundaries.get(str(step['id']), 'allocation_or_unverified'))} for step in verify_steps]
    print(json.dumps({'graph_analysis_SHA256': source_sha,
                      'steps': steps, 'target_verify_steps': len(steps),
                      'normalized_steps': sum(row['logical_work_verified'] for row in steps),
                      'scope': 'Logical sequence positions. Physical page capacity, masked graph lanes, '
                               'draft work and elapsed device time need separate evidence.'}, indent=2))


if __name__ == '__main__':
    main()
