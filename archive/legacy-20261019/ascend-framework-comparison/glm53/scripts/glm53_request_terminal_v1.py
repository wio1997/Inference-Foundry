"""Per-request frontend drain completion, independent of global gateway idle."""
import asyncio
import time


class RequestTerminals:
    def __init__(self):
        self._pending = {}
        self._seen = set()

    def reserve(self, request_id):
        assert isinstance(request_id, str) and request_id and request_id not in self._seen
        self._seen.add(request_id)
        ticket = {'request_id': request_id, 'event': asyncio.Event(), 'row': None,
                  'reserved_perf_counter_ns': time.perf_counter_ns(), 'finished': False}
        self._pending[request_id] = ticket
        return ticket

    def bind(self, request_id, row):
        ticket = self._pending[request_id]
        assert ticket['row'] is None and not ticket['finished'], 'Duplicate gateway request ID'
        ticket['row'] = row
        return ticket

    def finish(self, ticket, row):
        assert self._pending[ticket['request_id']] is ticket and ticket['row'] is row
        assert not ticket['finished']
        ticket['terminal_perf_counter_ns'] = time.perf_counter_ns()
        ticket['finished'] = True
        ticket['event'].set()

    async def wait_terminal(self, request_id, timeout=15):
        ticket = self._pending[request_id]
        await asyncio.wait_for(ticket['event'].wait(), timeout)
        assert ticket['finished'] and ticket['row'] is not None
        self._pending.pop(request_id)
        return {'request_id': request_id, 'gateway': ticket['row'],
                'reserved_perf_counter_ns': ticket['reserved_perf_counter_ns'],
                'terminal_perf_counter_ns': ticket['terminal_perf_counter_ns']}

    @property
    def pending(self):
        return tuple(self._pending)
