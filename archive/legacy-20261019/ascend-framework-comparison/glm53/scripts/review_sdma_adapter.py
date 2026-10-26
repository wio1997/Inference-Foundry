"""CPU-only contract checks for the custom adapter, without importing any SDK."""
import queue
import io
import json
import logging
import threading
import types
import unittest
from unittest.mock import patch

import glm53_sdma_transport as transport
import glm53_runtime_lifecycle as lifecycle


class Contracts(unittest.TestCase):
    def test_audit_reaches_configured_vllm_logger(self):
        logger = logging.getLogger('vllm')
        old = logger.level, logger.handlers[:], logger.propagate
        buffer = io.StringIO()
        logger.setLevel(logging.INFO)
        logger.handlers = [logging.StreamHandler(buffer)]
        logger.propagate = False
        try:
            transport.audit('audit_fixture', payload_bytes_completed=128)
            message = buffer.getvalue()
            self.assertIn('GLM53_PD_SDMA=', message)
            data = json.loads(message.split('GLM53_PD_SDMA=', 1)[1])
            self.assertEqual(data['event'], 'audit_fixture')
            self.assertEqual(data['payload_bytes_completed'], 128)
        finally:
            logger.setLevel(old[0]); logger.handlers = old[1]; logger.propagate = old[2]

    def configured(self):
        te = transport.GlobalTE()
        te.configure(role='Decode', device_id=0, session_port=31000, store_url='tcp://172.16.12.30:29580')
        te._session = '172.16.12.43:31000'
        return te

    def test_config_is_fixed_and_closed_engine_cannot_reopen(self):
        te = self.configured()
        with self.assertRaises(RuntimeError):
            te.configure(role='Decode', device_id=1, session_port=31000, store_url='tcp://172.16.12.30:29580')
        te._closed = True
        with self.assertRaises(RuntimeError):
            te.get_transfer_engine('172.16.12.43')

    def test_positive_registration_error_and_read_error_do_not_fallback(self):
        te = self.configured()
        calls = []
        te._engine = types.SimpleNamespace(batch_register_memory=lambda *a: 7)
        with self.assertRaises(RuntimeError):
            te.register_buffer([1000], [128])
        self.assertIsNone(te._registered)
        te._engine.batch_register_memory = lambda *a: 0
        te.register_buffer([1000], [128])
        te._engine.batch_transfer_sync_read = lambda *a: calls.append(a) or -1
        with self.assertRaises(RuntimeError):
            te.batch_transfer_sync_read('peer:1', [1000], [2000], [128])
        self.assertEqual(len(calls), 1)
        with self.assertRaises(ValueError):
            te.batch_transfer_sync_read('peer:1', [1000], [], [128])

    def test_queues_drain_before_engine_then_cache_then_backing_release(self):
        order = []
        te = self.configured()
        backing = object()
        te.pin_kv_caches(backing)
        te._engine = types.SimpleNamespace(
            destroy=lambda: order.append('destroy'),
            unInitialize=lambda: order.append('uninitialize'))
        class Worker(threading.Thread):
            def __init__(self):
                super().__init__(); self.request_queue = queue.Queue()
            def run(self):
                while True:
                    work = self.request_queue.get()
                    self.request_queue.task_done()
                    if work is lifecycle.STOP_CACHE_THREAD:
                        order.append('thread_stopped'); return
                    order.append('transfer')
        worker = Worker(); worker.request_queue.put('transfer'); worker.start()
        def close_store():
            self.assertIs(te._kv_refs, backing)
            order.append('cache_closed'); return 0
        store = types.SimpleNamespace(close=close_store)
        child = types.SimpleNamespace(connector_worker=types.SimpleNamespace(
            kv_send_thread=worker, m_store=types.SimpleNamespace(store=store)))
        connector = types.SimpleNamespace(_connectors=[child])
        with patch.object(transport, 'global_te', te):
            lifecycle.shutdown_multi(connector)
            lifecycle.shutdown_multi(connector)
        self.assertEqual(order, ['transfer', 'thread_stopped', 'destroy', 'uninitialize', 'cache_closed'])
        self.assertIsNone(te._kv_refs)


if __name__ == '__main__':
    unittest.main()
