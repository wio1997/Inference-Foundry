# SPDX-License-Identifier: Apache-2.0
"""Explicit owned, quiescent Flash PD teardown before communicator abort."""
import hashlib,pathlib

def _event(kind,**fields):
    from sglang.srt.disaggregation.ascend.flash_b103_pd_events import append
    return append(kind,**fields)

def stop_hash_control(adapter):
    with adapter._lock:
        if getattr(adapter,'_hash_closed',False):
            return
        server=getattr(adapter,'_hash_server',None)
        if server is not None:
            server.shutdown()
            server.server_close()
            thread=server._flash_thread
            thread.join(timeout=2.)
            if thread.is_alive():
                raise RuntimeError('Owned payload-hash control thread did not stop')
        hasher=getattr(adapter,'_hasher',None)
        if hasher is not None:
            hasher.seal()
        adapter._hash_closed=True

def close_engine(adapter):
    with adapter._lock:
        if getattr(adapter,'_closed',False):
            return 0
        if getattr(adapter,'_closing',False):
            raise RuntimeError('Prior or concurrent Flash SDK close did not complete')
        adapter._closing=True
        before=len(adapter._registered_ptrs)
        had_probe=adapter._probe is not None
        server=getattr(adapter,'_hash_server',None)
        _event('pd_sdk_close_begin',device=adapter.npu_id,role=adapter.role,
               registered_originals=before,had_probe=had_probe,had_hash_server=server is not None)
        try:
            stop_hash_control(adapter)
            hasher=getattr(adapter,'_hasher',None)
            adapter._torch.npu.synchronize()
            adapter.batch_deregister(sorted(adapter._registered_ptrs))
            assert not adapter._registered_ptrs and adapter._probe is None
            ret=adapter.engine.destroy()
            if ret is not None:
                raise RuntimeError('Unexpected MF1.3 destroy return: '+repr(ret))
            if hasattr(adapter,'_regions'):
                adapter._regions.clear()
            adapter._peer_regions.clear()
            adapter._closed=True
            _event('pd_sdk_close_complete',device=adapter.npu_id,role=adapter.role,
                   unregistered_originals=before,unregistered_probe=had_probe,
                   destroy_returned_none=True,registered_remaining=0,
                   hash_server_closed=server is not None,
                   hash_thread_alive=False,
                   hasher_sealed=hasher is not None and hasher.sealed,
                   source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
            return 0
        except BaseException as e:
            _event('pd_sdk_close_failed',device=adapter.npu_id,role=adapter.role,error=repr(e))
            raise

def close_scheduler_pd(scheduler):
    from sglang.srt.disaggregation.utils import DisaggregationMode
    role=scheduler.disaggregation_mode
    if role==DisaggregationMode.PREFILL:
        queue=scheduler.disagg_prefill_bootstrap_queue
    elif role==DisaggregationMode.DECODE:
        queue=scheduler.disagg_decode_prealloc_queue
    else:
        return
    manager=queue.kv_manager
    if type(manager).__module__!='sglang.srt.disaggregation.ascend.conn':
        raise ValueError('Explicit Flash teardown requires the qualified native ASCEND manager')
    queues=list(getattr(manager,'transfer_queues',[]))
    # This image's native FastQueue has put/get, not Queue.empty(). Pin its
    # source in the preflight and inspect its buffer under its own condition.
    for q in queues:
        if type(q).__module__!='sglang.srt.disaggregation.common.utils' or type(q).__name__!='FastQueue':
            raise TypeError('Unsupported native Flash transfer queue')
        with q._cond:
            if q._buf:
                raise RuntimeError('Cannot close own SDK with native transfers still queued')
    if any(v for v in getattr(manager,'_staging_outstanding',{}).values()):
        raise RuntimeError('Cannot close own SDK with native transfers still queued/outstanding')
    threads=list(getattr(manager,'_worker_threads',[]))
    stop_hash_control(manager.engine)
    phase(scheduler,'native_manager_teardown_begin')
    # Use the native role-switch teardown: it signals/joins transfer, bootstrap,
    # failed-session-probe and heartbeat workers, closes ZMQ, and deregisters.
    manager.teardown()
    if any(t.is_alive() for t in threads):
        raise RuntimeError('Native PD manager teardown left an owned worker alive')
    if manager.engine._registered_ptrs or manager.engine._probe is not None:
        raise RuntimeError('Native PD teardown did not deregister all owned buffers')
    if not manager._zmq_ctx.closed:
        raise RuntimeError('Native PD teardown left its owned ZMQ context open')
    _event('pd_native_manager_teardown_complete',device=manager.engine.npu_id,
           role=manager.engine.role,workers_joined=len(threads),workers_alive=0,
           zmq_closed=True,registered_remaining=0)
    phase(scheduler,'native_manager_teardown_complete')
    manager.engine.close()

def phase(scheduler,name):
    import torch
    _event('pd_worker_shutdown_phase',device=torch.npu.current_device(),phase=name,
           native_graceful_flag=bool(scheduler.gracefully_exit))
