"""Collect uninstrumented MegaMoE msprof PIPE traces as JSON (no HTML).

    python scripts/profiling/msprof_mega_moe.py --case 7168,3072,64,6,128,1 --output profiling/pipes
    python scripts/profiling/msprof_mega_moe.py --capture profiling/pipes/raw --output profiling/reexport

The output contains rank_<device>.json (Chrome/Perfetto trace), per-device summaries,
and summary.json. Each trace selects the last MegaMoE call and only sampled cores;
BIU timings explain pipeline overlap, not unprofiled end-to-end performance.
"""
import argparse
import collections
from decimal import Decimal
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def worker(local_rank, args):
    # Offline export and --help do not need PyTorch or an NPU environment.
    sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
    import torch
    import torch.distributed as dist
    from test_mega_moe import MegaMoECase
    from deep_gemm.testing.envs import init_dist
    from deep_gemm.testing.bench import _make_l2_flush_buffer

    rank, _, group = init_dist(local_rank, args.ranks)
    case = None
    try:
        case = MegaMoECase(group, *args.case, with_baseline=False)
        case.create_inputs(args.case[4])
        for _ in range(args.warmups):
            case.run_fused()
        torch.npu.synchronize()
        flush = _make_l2_flush_buffer()
        for _ in range(args.iters):
            flush.zero_()
            torch.npu.synchronize()
            dist.barrier(group)
            case.run_fused()
            torch.npu.synchronize()
        print(json.dumps(dict(rank=rank, case=args.case, captured_launches=args.iters,
                              instrumented=False, status='CAPTURE_OK')), flush=True)
    finally:
        if case is not None:
            case.destroy()
        dist.destroy_process_group()


PIPE = {'SU': 'S', 'CUBE': 'M', 'MTE1': 'MTE1', 'MTE2': 'MTE2',
        'FIXP': 'FIX', 'VEC': 'V', 'MTE3': 'MTE3'}
SUBCORE = {'aic': 'AIC', 'aiv0': 'AIV0', 'aiv1': 'AIV1'}


def export(capture, output):
    traces = list(capture.glob('mindstudio_profiler_output/msprof_*.json'))
    if len(traces) != 1:
        raise ValueError(f'expected one exported msprof timeline in {capture}')
    events = json.loads(traces[0].read_text())
    if isinstance(events, dict):
        events = events['traceEvents']
    tasks = [e for e in events if e.get('ph') == 'X' and 'mega_moe_impl' in e.get('name', '')]
    if not tasks:
        raise ValueError('no MegaMoE kernel found in the profiler timeline')
    # Warmup calls are also captured; export the last cold-L2 launch.
    tasks.sort(key=lambda event: Decimal(str(event['ts'])))
    target = tasks[-1]
    origin = Decimal(str(target['ts']))
    duration = Decimal(str(target['dur']))
    device = next(capture.glob('device_*'))
    rank = int(device.name.split('_')[-1])
    info_path = next(f for f in device.glob('info.json.*') if f.suffix[1:].isdigit())
    info = json.loads(info_path.read_text())['DeviceInfo'][0]
    ticks_per_us = Decimal(info['hwts_frequency'])
    lanes = {}
    for event in events:
        if event.get('name') == 'thread_name':
            match = re.fullmatch(r'Group(\d+)-(aic|aiv0|aiv1)', event['args']['name'])
            if match:
                lanes[event['pid'], event['tid']] = (int(match[1]), match[2])
    rows = collections.defaultdict(collections.deque)
    db = device / 'sqlite/biu_perf.db'
    with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as connection:
        for group, core, block, name, ts, dur, checkpoint in connection.execute(
                'select * from BiuInstrStatus order by rowid'):
            rows[group, core].append((block, name, Decimal(str(dur)), checkpoint))
    tracks = collections.defaultdict(list)
    boundary_clipped = collections.Counter()
    matched = 0
    for event in events:
        lane = lanes.get((event.get('pid'), event.get('tid')))
        if not lane or event.get('ph') != 'X' or 'Core Type' not in event.get('args', {}):
            continue
        block, name, db_duration, checkpoint = rows[lane].popleft()
        if (name, block) != (event['name'], event['args']['Block Id']):
            raise ValueError('msprof JSON and BIU DB event ordering differ')
        matched += 1
        if checkpoint is not None:
            continue
        start = Decimal(str(event['ts'])) - origin
        # BIU DB durations are hardware ticks; JSON timestamps are microseconds.
        end = start + db_duration / ticks_per_us
        if end <= 0 or start >= duration or end <= start:
            continue
        key = (block, SUBCORE[lane[1]], PIPE[name])
        if start < 0 or end > duration:
            boundary_clipped[key] += 1
        start, end = max(start, Decimal(0)), min(end, duration)
        tracks[key].append([float(start), float(end-start)])
    if any(rows.values()) or not tracks:
        raise ValueError('unmatched BIU records or no PIPE events in the selected kernel')

    result_events = [dict(ph='M', name='process_name', pid=rank, tid=0,
                          args=dict(name=f'rank {rank}: msprof PIPE timeline')),
                     dict(ph='X', cat='msprof.kernel', name='mega_moe_impl', pid=rank,
                          tid=9999, ts=0, dur=float(duration))]
    statistics = []
    for tid, (key, spans) in enumerate(sorted(tracks.items())):
        core, subcore, pipe = key
        spans.sort()
        track_label = f'core {core:02d} / {subcore} / PIPE_{pipe}'
        result_events.append(dict(ph='M', name='thread_name', pid=rank, tid=tid,
                                  args=dict(name=track_label)))
        for start, span in spans:
            result_events.append(dict(ph='X', cat='msprof.pipe', name=f'PIPE_{pipe}',
                                      pid=rank, tid=tid, ts=start, dur=span,
                                      args=dict(core=core, subcore=subcore)))
        statistics.append(dict(core=core, subcore=subcore, pipe=pipe, events=len(spans),
                               first_us=spans[0][0], last_us=max(s+d for s,d in spans),
                               boundary_clipped=boundary_clipped[key]))
    trace_path = output / f'rank_{rank}.json'
    summary_path = output / f'rank_{rank}_summary.json'
    for path in (trace_path, summary_path):
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite {path}')
    output.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(json.dumps(dict(traceEvents=result_events, displayTimeUnit='us'),
                                     separators=(',', ':')))
    summary = dict(rank=rank, target_calls=len(tasks), selected_call=len(tasks)-1,
                   task_duration_us=float(duration), instrumented=False, source=str(traces[0]),
                   matched_biu_events=matched, exported_pipe_events=sum(len(v) for v in tracks.values()),
                   sampled_cores=sorted({k[0] for k in tracks}), tracks=statistics)
    summary_path.write_text(json.dumps(summary, indent=2))
    return {k:v for k,v in summary.items() if k!='tracks'}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, help='JSON output directory; new captures go in its raw/ subdirectory')
    parser.add_argument('--capture', type=Path, help='export an existing PROF_* directory or its parent without collecting')
    parser.add_argument('--ranks', type=int, default=8)
    parser.add_argument('--case', type=lambda s: tuple(map(int, s.split(','))),
                        default=(4096, 2048, 64, 6, 8192, 1), metavar='H,I,E,K,T,S')
    parser.add_argument('--warmups', type=int, default=3)
    parser.add_argument('--iters', type=int, default=1, help='cold-L2 launches; export the last launch')
    parser.add_argument('--launch-only', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.ranks < 1 or args.warmups < 0 or args.iters < 1 or len(args.case) != 6:
        parser.error('expected positive ranks/iters, nonnegative warmups and H,I,E,K,T,S')
    if args.launch_only:
        import torch
        if args.ranks == 1:
            worker(0, args)
        else:
            torch.multiprocessing.spawn(worker, args=(args,), nprocs=args.ranks)
        return
    if args.output is None:
        parser.error('--output is required')
    args.output = args.output.resolve()
    summary_path = args.output / 'summary.json'
    if summary_path.exists():
        raise FileExistsError(f'Refusing to overwrite {summary_path}')
    if args.capture is None:
        args.capture = args.output / 'raw'
        # Refuse mixed captures: a previous run must never silently replace a rank's trace.
        args.capture.mkdir(parents=True, exist_ok=False)
        subprocess.run(['msprof', '--instr-profiling=on', f'--output={args.capture}',
                        sys.executable, str(Path(__file__).resolve()), '--launch-only',
                        '--ranks', str(args.ranks), '--case', ','.join(map(str, args.case)),
                        '--warmups', str(args.warmups), '--iters', str(args.iters)],
                       cwd=ROOT, check=True)
    captures = [args.capture] if args.capture.name.startswith('PROF_') else sorted(args.capture.glob('PROF_*'))
    if not captures:
        raise ValueError(f'no PROF captures found in {args.capture}')
    summaries = []
    for capture in captures:
        summary = export(capture, args.output)
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    summary_path.write_text(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
