"""Local-only Decimal interval enclosure supplement for accepted L15-D contexts.

Reads only the frozen local `node-{36,41}-context.json` files. It does not read
remote traces, profiles or CSVs, does not modify its inputs, and performs no
float time conversion, unit conversion, clock-base guess or cross-file/rank
join. The original `enclosing_step_spans=null` values are preserved separately
and are never rewritten.
"""
import collections
import hashlib
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUPPLEMENT_ID = 'C32-L15-timeline-Decimal-enclosure-supplement-v1'
L15_RUN = '20261019T-L15-sglang-existing-D-timeline-source-v1'
CONTEXTS = {
    36: 'runs/20261019T-L15-sglang-existing-D-timeline-source-v1/node-36-context.json',
    41: 'runs/20261019T-L15-sglang-existing-D-timeline-source-v1/node-41-context.json',
}
CONTEXT_SHA = {
    36: '25827d55ee9cbe42eba03479d2b9adaac2226c7e67fa4c879aa20b3dc69cdd3e',
    41: '73ce123e302d642b3a01fe37363e56c8b0511741ea7b8d96a19bf6d602615334',
}
CONTEXT_BYTES = {36: 826654, 41: 830334}
EXPECTED_RANKS = {36: 0, 41: 16}
DECISION = 'results/C32-L15-timeline-source-decision-v1.json'
DECISION_SHA = '8136fdf8171aa9170f3ca7462fb64b907f2b49823ebb43b41d738ee71091b2e8'
RECEIPT = 'runs/20261019T-L15-sglang-existing-D-timeline-source-v1/receipt.json'
RECEIPT_SHA = 'ea0e76dc6b3b778349cde0dded5944c7b811c722ab85ca2d2182202704fa7baa'
ACTUAL_REVIEW = 'results/C32-L15-D-timeline-source-actual-review-v1.json'
ACTUAL_REVIEW_SHA = '70b0de421f85080079ab3548157681ad55d7e99b111571629c1ce19e219dbc4c'
OUTPUT = 'results/C32-L15-timeline-Decimal-enclosure-supplement-v1.json'
EXACT_API_GRAPH_NAMES = ['aclrtMemcpy', 'aclrtMemcpyAsync', 'aclmdlRIExecuteAsync', 'aclmdlRICaptureGetInfo']
MAX_ENCLOSING = 16


def _verified_bytes(relative, digest):
    path = ROOT / relative
    assert path.is_file() and not path.is_symlink(), relative
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest, relative
    return raw


def _decimal(value, label):
    if value is None or isinstance(value, bool):
        return None, f'{label}_missing_or_boolean'
    text = value if isinstance(value, str) else str(value)
    try:
        parsed = Decimal(text)
    except (InvalidOperation, ValueError):
        return None, f'{label}_not_decimal'
    if not parsed.is_finite():
        return None, f'{label}_not_finite'
    if parsed < 0:
        return None, f'{label}_negative'
    return parsed, None


def _interval(event):
    start, start_error = _decimal(event.get('ts'), 'ts')
    duration, duration_error = _decimal(event.get('dur'), 'dur')
    if start_error or duration_error:
        return None, None, ';'.join(item for item in (start_error, duration_error) if item)
    return start, start + duration, None


def _span_row(span, selected_index):
    return {
        'selected_event_index': selected_index,
        'name': span.get('name'),
        'pid': span.get('pid'),
        'tid': span.get('tid'),
        'cat': span.get('cat'),
        'ts_raw': span.get('ts'),
        'dur_raw': span.get('dur'),
    }


def _copy_record(row, selected_index, step_rows):
    event = row['event']
    copy_start, copy_end, copy_error = _interval(event)
    record = {
        'selected_event_index': selected_index,
        'selection_role': row.get('selection_role'),
        'name': event.get('name'),
        'pid': event.get('pid'),
        'tid': event.get('tid'),
        'cat': event.get('cat'),
        'ts_raw': event.get('ts'),
        'dur_raw': event.get('dur'),
        'original_enclosing_step_spans': row.get('enclosing_step_spans'),
        'original_enclosing_null_reason': row.get('enclosing_null_reason'),
        'original_enclosing_preserved': True,
        'raw_unit_preserved': True,
        'supplement_enclosing_step_spans': [],
        'supplement_enclosing_count': 0,
        'supplement_enclosing_truncated': False,
        'supplement_enclosing_null_reason': None,
    }
    if copy_error:
        record['supplement_enclosing_null_reason'] = copy_error
        record['copy_interval_raw_units'] = None
        return record
    record['copy_interval_raw_units'] = {'start_decimal': str(copy_start), 'end_decimal': str(copy_end)}
    containing = []
    for span_index, span in step_rows:
        if span.get('pid') != event.get('pid') or span.get('tid') != event.get('tid'):
            continue
        span_start, span_end, span_error = _interval(span)
        if span_error:
            continue
        if span_start <= copy_start and span_end >= copy_end:
            containing.append(_span_row(span, span_index) | {
                'start_decimal': str(span_start),
                'end_decimal': str(span_end),
            })
    containing.sort(key=lambda item: (Decimal(item['start_decimal']), -Decimal(item['end_decimal']), item['selected_event_index']))
    record['supplement_enclosing_step_spans'] = containing[:MAX_ENCLOSING]
    record['supplement_enclosing_count'] = len(containing)
    record['supplement_enclosing_truncated'] = len(containing) > MAX_ENCLOSING
    if not containing:
        record['supplement_enclosing_null_reason'] = 'no_same_file_pid_tid_step_span_contains_event_interval_with_decimal'
    return record


def _inner_draft_after_copy(record, step_rows):
    if not record.get('copy_interval_raw_units'):
        return None
    copy_end = Decimal(record['copy_interval_raw_units']['end_decimal'])
    outer = [span for span in record['supplement_enclosing_step_spans'] if span.get('name') == 'draft_extend']
    candidates = []
    for span_index, span in step_rows:
        if span.get('name') != 'step[DRAFT bs=1]':
            continue
        if span.get('pid') != record.get('pid') or span.get('tid') != record.get('tid'):
            continue
        span_start, span_end, span_error = _interval(span)
        if span_error or span_start < copy_end:
            continue
        if outer and not any(Decimal(item['start_decimal']) <= span_start and Decimal(item['end_decimal']) >= span_end for item in outer):
            continue
        candidates.append((span_start, span_end, _span_row(span, span_index)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2]['selected_event_index']))
    start, end, row = candidates[0]
    return {
        'span': row,
        'start_decimal': str(start),
        'end_decimal': str(end),
        'copy_end_before_inner_draft_start': copy_end <= start,
    }


def _node_supplement(node):
    relative = CONTEXTS[node]
    before = (ROOT / relative).stat()
    raw = _verified_bytes(relative, CONTEXT_SHA[node])
    assert len(raw) == CONTEXT_BYTES[node], relative
    document = json.loads(raw)
    assert document['project_id'] == 'glm53'
    assert document['run_id'] == L15_RUN
    assert document['node'] == node and document['representative_TP_rank'] == EXPECTED_RANKS[node]
    assert document['collection_pass'] is True
    timeline = document['timeline']
    selected = timeline['selected_events']
    step_rows = [(index, row['event']) for index, row in enumerate(selected) if row.get('selection_role') == 'step_span']
    copy_rows = [(index, row) for index, row in enumerate(selected) if str(row.get('selection_role', '')).startswith('copy:')]
    records = [_copy_record(row, index, step_rows) for index, row in copy_rows]
    h2d = [record for record in records if record.get('name') == 'acl_memcpy_host_to_device']
    h2d.sort(key=lambda record: (-Decimal(str(record['dur_raw'])), record['selected_event_index']))
    top_h2d = []
    for record in h2d[:2]:
        outer = [span for span in record['supplement_enclosing_step_spans'] if span.get('name') == 'draft_extend']
        inner = _inner_draft_after_copy(record, step_rows)
        top_h2d.append({
            'selected_event_index': record['selected_event_index'],
            'pid': record['pid'],
            'tid': record['tid'],
            'ts_raw': record['ts_raw'],
            'dur_raw': record['dur_raw'],
            'inside_draft_extend': bool(outer),
            'containing_draft_extend_spans': outer,
            'inner_draft_after_copy': inner,
            'position_observation': (
                'inside_outer_draft_extend_and_before_inner_DRAFT'
                if outer and inner and inner['copy_end_before_inner_draft_start']
                else 'inside_outer_draft_extend_but_no_inner_DRAFT_after_copy'
                if outer and not (inner and inner['copy_end_before_inner_draft_start'])
                else 'not_inside_outer_draft_extend'
            ),
        })
    contained = sum(1 for record in records if record['supplement_enclosing_count'])
    phase_counts = collections.Counter()
    for record in records:
        names = [span['name'] for span in record['supplement_enclosing_step_spans']]
        if any(name == 'draft_extend' for name in names):
            phase_counts['draft_extend'] += 1
        elif any(name == 'draft' for name in names):
            phase_counts['draft'] += 1
        elif any(str(name).startswith('step[DRAFT') for name in names):
            phase_counts['inner_DRAFT_only'] += 1
        elif names:
            phase_counts['other_step_span'] += 1
        else:
            phase_counts['no_enclosing_step_span'] += 1
    exact_selected = collections.Counter()
    for row in selected:
        role = row.get('selection_role')
        if role in ('api:aclrtMemcpy', 'api:aclrtMemcpyAsync', 'graph_api'):
            exact_selected[role] += 1
    prefixed = []
    for row in timeline['all_x_name_counts_by_pid_tid_cat']:
        name = row.get('name')
        if isinstance(name, str) and name.startswith('AscendCL@') and any(exact in name for exact in EXACT_API_GRAPH_NAMES):
            prefixed.append(row)
    after = (ROOT / relative).stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'input context changed during read'
    return {
        'context_file': {'path': relative, 'bytes': len(raw), 'sha256': CONTEXT_SHA[node]},
        'representative_TP_rank': EXPECTED_RANKS[node],
        'selected_event_coverage': {
            'selected_total': timeline['counts']['selected_total'],
            'step_span_rows': len(step_rows),
            'step_spans_observed': timeline['counts']['step_spans_observed'],
            'copy_rows': len(records),
            'exact_api_graph_rows_selected': dict(exact_selected),
        },
        'copy_enclosure': records,
        'copy_enclosure_summary': {
            'total_copy_rows': len(records),
            'contained_rows': contained,
            'not_contained_rows': len(records) - contained,
            'phase_bucket_counts': dict(phase_counts),
        },
        'top_h2d_observation': {
            'ordering': 'longest two acl_memcpy_host_to_device rows by Decimal(str(dur)); raw units retained',
            'rows': top_h2d,
        },
        'selection_limitations': {
            'exact_unprefixed_api_graph_names': EXACT_API_GRAPH_NAMES,
            'exact_selection_note': 'The L15 collector selected only exact unprefixed names. No api:aclrtMemcpy, api:aclrtMemcpyAsync or graph_api rows were selected from these contexts.',
            'actual_prefixed_names_observed_in_context_index': prefixed,
            'actual_prefixed_names_note': 'These rows come from the bounded all_x_name_counts_by_pid_tid_cat index already present in the accepted context. They are recorded as a name/selection limitation only; no event interval or context is invented for them.',
            'cat_missing_count': timeline['missing_field_counts']['cat'],
            'cat_missing_preserved': True,
            'all_x_distinct_key_overflow': timeline['all_x_distinct_key_overflow'],
        },
        'input_file_unchanged': True,
    }


def main():
    decision = json.loads(_verified_bytes(DECISION, DECISION_SHA))
    receipt = json.loads(_verified_bytes(RECEIPT, RECEIPT_SHA))
    review = json.loads(_verified_bytes(ACTUAL_REVIEW, ACTUAL_REVIEW_SHA))
    assert decision['project_id'] == 'glm53' and decision['decision'] == 'ACCEPTED'
    assert decision['collection_receipt_sha256'] == RECEIPT_SHA
    assert decision['source_eager_runner_sha256'] == 'a04d7344319eccbeacefad3f60e559c716f412e680ed0f712645ef149238a6b5'
    assert receipt['project_id'] == 'glm53' and receipt['pass'] is True and receipt['source_sha256'] == 'bdc451ea0927c63cd47d76bd7ec9ab4a46270221db952d48a18e6509a236e3a7'
    assert receipt['model_requests'] == 0 and receipt['HTTP_requests'] == 0 and receipt['NPU_calls'] == 0 and receipt['SDK_init_calls'] == 0
    assert review['decision'] == 'APPROVED_READ_ONLY' and review['read_only'] is True
    nodes = {str(node): _node_supplement(node) for node in (36, 41)}
    result = {
        'project_id': 'glm53',
        'supplement_id': SUPPLEMENT_ID,
        'status': 'SUPPLEMENT_ONLY_BOUNDED_EVIDENCE',
        'scope': 'Local Decimal(raw ts) / Decimal(str(dur)) geometric interval containment of the already selected L15 copy events within same-file, same-pid, same-tid step_span events. No performance, critical-path or optimization conclusion.',
        'run_id': L15_RUN,
        'decision': {'path': DECISION, 'sha256': DECISION_SHA, 'decision': decision['decision']},
        'collection_receipt': {'path': RECEIPT, 'sha256': RECEIPT_SHA, 'pass': receipt['pass']},
        'actual_review': {'path': ACTUAL_REVIEW, 'sha256': ACTUAL_REVIEW_SHA, 'decision': review['decision'], 'read_only': review['read_only']},
        'method': {
            'input': 'already selected events inside the frozen local L15 node contexts only',
            'ts_parse': 'Decimal(raw JSON value as text); ts values are raw strings',
            'dur_parse': 'Decimal(str(dur))',
            'checks': 'finite and non-negative only',
            'containment': 'span_start <= copy_start and span_end >= copy_end',
            'scope_rule': 'same context file, same pid, same tid',
            'units': 'raw units preserved; no conversion to seconds and no clock-base guess',
            'float_time_conversion': False,
            'cross_file_join': False,
            'cross_rank_alignment': False,
            'overlap_sum': False,
            'critical_path_claim': False,
            'original_null_rewritten': False,
        },
        'input_files_unchanged': all(node['input_file_unchanged'] for node in nodes.values()),
        'original_enclosing_nulls_preserved': True,
        'nodes': nodes,
        'observation': {
            'primary': 'On each representative rank, the two longest acl_memcpy_host_to_device rows are geometrically enclosed by an outer draft_extend span and end before the next inner step[DRAFT bs=1] span. Other long copy rows are generally enclosed by draft or inner DRAFT step spans.',
            'basis': 'Decimal containment only; this identifies phase placement, not whether a wait is avoidable or how much time could be saved.',
            'graph_drain_caveat': 'A long H2D host wait can include earlier asynchronous Graph work; this supplement does not add host waits to device time.',
            'device_time_claim': False,
        },
        'model_requests': 0,
        'new_profiles': 0,
        'SSH_sessions': 0,
        'HTTP_requests': 0,
        'NPU_calls': 0,
        'SDK_init_calls': 0,
        'runtime_changes': False,
        'performance_certificate_allowed': False,
        'objective_complete': False,
    }
    target = ROOT / OUTPUT
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({
        'supplement_id': SUPPLEMENT_ID,
        'output': OUTPUT,
        'output_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'nodes': {
            node: {
                'copy_rows': data['copy_enclosure_summary']['total_copy_rows'],
                'contained_rows': data['copy_enclosure_summary']['contained_rows'],
                'phase_bucket_counts': data['copy_enclosure_summary']['phase_bucket_counts'],
                'top_h2d_positions': [row['position_observation'] for row in data['top_h2d_observation']['rows']],
            }
            for node, data in nodes.items()
        },
        'model_requests': 0,
        'new_profiles': 0,
        'SSH_sessions': 0,
        'HTTP_requests': 0,
        'NPU_calls': 0,
        'SDK_init_calls': 0,
    }, indent=2))


if __name__ == '__main__':
    main()
