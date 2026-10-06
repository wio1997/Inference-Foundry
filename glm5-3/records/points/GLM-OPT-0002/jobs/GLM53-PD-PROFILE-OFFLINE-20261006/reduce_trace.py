"""Read-only trace reduction. No inferred removable time or crosshost clock sum."""
import collections
import hashlib
import json
import re
import sys
from pathlib import Path


def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def reduce(root):
    rows = []
    for trace in sorted(root.rglob('trace_view.json')):
        value = json.loads(trace.read_text())
        events = value['traceEvents'] if isinstance(value, dict) else value
        names = {e['pid']: e.get('args', {}).get('name') for e in events
                 if e.get('ph') == 'M' and e.get('name') == 'process_name'}
        hardware_pids = {pid for pid, name in names.items() if name == 'Ascend Hardware'}
        rank_match = re.search(r'(?:^|_)tp(\d+)(?:_|$)', trace.parent.parent.name, re.I)
        if not rank_match:
            rank_match = re.search(r'(?:^|_)tp(\d+)(?:_|$)', str(trace), re.I)
        rank = int(rank_match.group(1)) if rank_match else None
        hardware = [(i, e) for i, e in enumerate(events) if e.get('ph') == 'X'
                    and e.get('pid') in hardware_pids and float(e.get('dur', 0)) > 0]
        groups = collections.defaultdict(list)
        types = collections.Counter()
        top = collections.defaultdict(lambda: [0, 0.0])
        for _, event in hardware:
            args = event.get('args', {})
            typ = args.get('Task Type', args.get('task_type', 'UNKNOWN'))
            types[typ] += 1
            start = float(event['ts'])
            duration = float(event['dur'])
            groups[typ].append([start, start + duration])
            top[event['name']][0] += 1
            top[event['name']][1] += duration
        union = merge([[float(e['ts']), float(e['ts']) + float(e['dur'])] for _, e in hardware])
        gaps = [[left[1], right[0]] for left, right in zip(union, union[1:]) if right[0] > left[1]]
        first = union[0][0] if union else None
        last = union[-1][1] if union else None
        markers = [(i, e) for i, e in enumerate(events) if e.get('ph') == 'X' and
                   any(s in str(e.get('name', '')).lower() for s in ('execute_model', 'model_forward', 'draft', 'sample'))]
        row = dict(path=str(trace), bytes=trace.stat().st_size,
                   sha256=hashlib.sha256(trace.read_bytes()).hexdigest(), tp_rank=rank,
                   process_names=names, hardware_pids=sorted(hardware_pids), hardware_events=len(hardware),
                   task_types=dict(types), hardware_span_us=None if first is None else last-first,
                   hardware_union_us=sum(b-a for a, b in union),
                   task_type_union_us={key: sum(b-a for a, b in merge(intervals)) for key, intervals in groups.items()},
                   largest_between_hardware_gap_us=max([b-a for a, b in gaps] or [0]),
                   largest_between_hardware_gaps=sorted(gaps, key=lambda x: x[1]-x[0], reverse=True)[:5],
                   top_reported_hardware_names=sorted([(name, count, duration) for name, (count, duration) in top.items()], key=lambda x: x[2], reverse=True)[:12],
                   marker_names=collections.Counter(e['name'] for _, e in markers))
        if rank == 0 or not hardware:
            row['decisive_original_events'] = dict(
                first_hardware=[dict(index=i, event=e) for i, e in hardware[:12]],
                longest_hardware=[dict(index=i, event=e) for i, e in sorted(hardware, key=lambda x: float(x[1]['dur']), reverse=True)[:8]],
                markers=[dict(index=i, event=e) for i, e in markers[:14]])
        rows.append(row)
    observed = {row['tp_rank'] for row in rows if row['hardware_events']}
    return dict(profiles=rows, observed_tp_ranks=sorted(observed, key=lambda x: -1 if x is None else x),
                all16_nonempty_device_traces=observed == set(range(16)),
                limits=['Per-trace timestamp coordinates only; no subtraction or summed critical path across hosts.',
                        'Hardware tasks may overlap and include necessary collectives; interval union is not removable waiting.',
                        'First-to-last hardware extent excludes profile control idle boundaries; task types UNKNOWN remain unknown.',
                        'Single8-token profiled diagnostic cannot prove SLA percentiles, semantic final answers or stable capacity.'])


if __name__ == '__main__':
    print(json.dumps(reduce(Path(sys.argv[1])), ensure_ascii=False))
