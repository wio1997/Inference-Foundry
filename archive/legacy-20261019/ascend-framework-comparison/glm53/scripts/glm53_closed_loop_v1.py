"""Prefix-affine closed-loop dispatch; no framework imports or resource actions.

The caller supplies the frozen AISBench submission coroutine. One worker per
independent prefix issues its four variants in order, immediately refilling its
slot after termination. There is no cross-prefix round barrier and no global
gateway-idle wait. Native backend drains belong to the submission's own request.
"""
import asyncio
import time


async def closed_loop(groups, submit, *, on_event=None):
    assert groups and all(len(group) == 4 for group in groups)
    identifiers = [item['request_id'] for group in groups for item in group]
    assert len(identifiers) == len(set(identifiers))
    for prefix, group in enumerate(groups):
        assert [item['prefix_id'] for item in group] == [prefix] * 4
        assert [item['variant'] for item in group] == [1, 2, 3, 4]
    gate = asyncio.Event()
    rows, events = [], []
    inflight = 0

    def event(kind, request, stamp):
        record = {'event': kind, 'request_id': request['request_id'],
                  'prefix_id': request['prefix_id'], 'variant': request['variant'],
                  'perf_counter_ns': stamp, 'client_inflight': inflight}
        events.append(record)
        if on_event is not None:
            on_event(record)

    async def worker(group):
        nonlocal inflight
        await gate.wait()
        for request in group:
            begin = time.perf_counter_ns()
            inflight += 1
            event('dispatch', request, begin)
            row = {'request_id': request['request_id'],
                   'prefix_id': request['prefix_id'], 'variant': request['variant'],
                   'outer_dispatch_perf_counter_ns': begin}
            try:
                row['result'] = await submit(request)
            except asyncio.CancelledError:
                row['cancelled'] = True
                raise
            except Exception as error:
                row['submission_exception'] = repr(error)
            finally:
                end = time.perf_counter_ns()
                row['outer_terminal_perf_counter_ns'] = end
                inflight -= 1
                event('terminal', request, end)
                rows.append(row)

    tasks = [asyncio.create_task(worker(group)) for group in groups]
    gate.set()
    try:
        await asyncio.gather(*tasks)
    finally:
        # On caller cancellation, finish cancellation accounting for every task.
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    assert inflight == 0 and len(rows) == len(identifiers)
    return {'rows': rows, 'client_inflight_events': events,
            'client_concurrency_limit': len(groups),
            'first_dispatch_perf_counter_ns': min(r['outer_dispatch_perf_counter_ns'] for r in rows),
            'last_terminal_perf_counter_ns': max(r['outer_terminal_perf_counter_ns'] for r in rows)}
