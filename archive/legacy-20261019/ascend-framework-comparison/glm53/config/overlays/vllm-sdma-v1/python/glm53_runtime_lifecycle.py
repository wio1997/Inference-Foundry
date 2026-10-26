"""Bounded shutdown for this project's SDMA+native-cache MultiConnector."""
from __future__ import annotations

import json
import logging
import time

log = logging.getLogger(__name__)
STOP_CACHE_THREAD = object()


def _drain(thread, deadline):
    queue = thread.request_queue
    with queue.all_tasks_done:
        while queue.unfinished_tasks:
            checker = getattr(thread, 'raise_if_failed', None)
            if checker:
                checker()
            if time.monotonic() >= deadline:
                raise TimeoutError(f'{thread.name}: {queue.unfinished_tasks} unfinished transfers at shutdown')
            queue.all_tasks_done.wait(timeout=.1)


def shutdown_multi(connector):
    if getattr(connector, '_glm53_shutdown_complete', False):
        return
    children = connector._connectors
    threads = []
    stores = []
    seen_threads = set()
    for child in children:
        worker = getattr(child, 'connector_worker', None)
        scheduler = getattr(child, 'connector_scheduler', None)
        for name, sentinel in [('kv_send_thread', STOP_CACHE_THREAD),
                               ('kv_recv_thread', STOP_CACHE_THREAD),
                               ('_recving_thread', (None, None))]:
            thread = getattr(worker, name, None)
            if thread is not None and id(thread) not in seen_threads:
                seen_threads.add(id(thread))
                threads.append((thread, sentinel))
        backend = getattr(worker, 'm_store', None) or getattr(scheduler, 'store_scheduler', None)
        if backend is not None and getattr(backend, 'store', None) is not None:
            stores.append(backend.store)
    log.info('GLM53_LIFECYCLE=%s', json.dumps({'event': 'drain_begin', 'threads': len(threads),
                                             'cache_clients': len(stores)}))
    deadline = time.monotonic() + 30
    for thread, _ in threads:
        _drain(thread, deadline)
    for thread, sentinel in threads:
        thread.request_queue.put(sentinel)
    for thread, _ in threads:
        thread.join(timeout=10)
        if thread.is_alive():
            raise TimeoutError(f'{thread.name} did not stop before SDK shutdown')
    from glm53_sdma_transport import global_te
    global_te.close()
    # KV tensors remain pinned while both native SDKs release registrations.
    seen_stores = set()
    for store in stores:
        if id(store) in seen_stores:
            continue
        seen_stores.add(id(store))
        rc = store.close()
        if rc != 0:
            raise RuntimeError(f'Native MemCache close failed: {rc}')
    global_te.release_kv_pins()
    connector._glm53_shutdown_complete = True
    log.info('GLM53_LIFECYCLE=%s', json.dumps({'event': 'shutdown_complete'}))
