"""Reuse static SFA graph replay without a host wait for absent host updates."""
from contextvars import ContextVar
import functools
import hashlib
import json
import os
from pathlib import Path
import time


def install(ns, root=None, deps=None):
    assert os.environ.get('GLM53_P_FIRST_DRAFT_EXPORT') == '0'
    if deps is None:
        from vllm.forward_context import get_forward_context
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX
        from vllm_ascend.attention import sfa_v1
        from vllm_ascend.compilation import acl_graph
        assert hashlib.sha256(Path(acl_graph.__file__).read_bytes()).hexdigest() == '8e1c269e331d005f1361acc5cec24402bff7ae39b75e952ea890da7812804955'
        assert hashlib.sha256(Path(sfa_v1.__file__).read_bytes()).hexdigest() == '9b94f67f30364d40caf4a7a67af0f0365d1ba0b5048939873ab0464e95b096dc'
        deps = {'Graph': acl_graph.ACLGraphWrapper, 'SFA': sfa_v1.AscendSFABackend,
                'get_context': get_forward_context, 'is_draft': lambda: _EXTRA_CTX.is_draft_model,
                'updatable': acl_graph.use_updatable_graph, 'logger': acl_graph.logger}
    Runner, Mode, Graph = ns['NPUModelRunner'], ns['CUDAGraphMode'], deps['Graph']
    root = Path('/out/sfa-replay-order-probe') if root is None else Path(root)
    root.mkdir(exist_ok=True)
    active = ContextVar('glm53_static_sfa_replay', default=None)
    records = 0
    model_original = Runner._model_forward

    @functools.wraps(model_original)
    def model_forward(self, *args, **kwargs):
        nonlocal records
        ids = list(self.input_batch.req_ids)
        real = bool(ids and all(rid in self.requests and self.requests[rid].sampling_params is not None for rid in ids))
        parallel = self.parallel_config
        ctx = deps['get_context']()
        executor = getattr(ctx, 'device_metadata_executor', None)
        eligible = bool(real and self.is_kv_consumer and not self.is_kv_producer and self.use_async_scheduling
                        and parallel.tensor_parallel_size == 16 and parallel.data_parallel_size == 1
                        and parallel.pipeline_parallel_size == 1 and parallel.decode_context_parallel_size == 1
                        and parallel.prefill_context_parallel_size == 1 and self.num_spec_tokens == 5
                        and not ns['enable_sp']() and self.model_config.use_mla and not self.model_config.is_hybrid
                        and ctx.cudagraph_runtime_mode == Mode.FULL and not ctx.capturing
                        and executor is None and not deps['is_draft']()
                        and all(self.requests[rid].sampling_params.max_tokens > 1 for rid in ids))
        frame = {'request_ids': ids, 'tp_rank': self.tp_rank, 'role_eligible': eligible,
                 'observe': not (root / 'disabled').exists(), 'calls': []} if real else None
        token = active.set(frame)
        try:
            return model_original(self, *args, **kwargs)
        finally:
            active.reset(token)
            if frame is not None and frame['observe'] and records < 3000:
                records += 1
                frame.update(pid=os.getpid(), monotonic_ns=time.monotonic_ns(),
                             kind='static_SFA_native_replay',
                             scope='Native graph/guard/CPU call selection, not a device duration claim')
                with (root / ('rank-%d-%d.jsonl' % (self.tp_rank, os.getpid()))).open('a') as out:
                    out.write(json.dumps(frame, sort_keys=True) + '\n')

    Runner._model_forward = model_forward
    call_original = Graph.__call__

    @functools.wraps(call_original)
    def call(self, *args, **kwargs):
        frame = active.get()
        if frame is None:
            return call_original(self, *args, **kwargs)
        ctx = deps['get_context']()
        descriptor = ctx.batch_descriptor
        entry = self.concrete_aclgraph_entries.get(descriptor)
        backend = self.attn_backend
        static_sfa = bool(isinstance(backend, type) and issubclass(backend, deps['SFA'])
                          and not deps['updatable'](backend))
        eligible = bool(frame['role_eligible'] and self.runtime_mode == Mode.FULL
                        and ctx.cudagraph_runtime_mode == Mode.FULL and not ctx.capturing
                        and not self.enable_enpu and static_sfa and entry is not None
                        and entry.aclgraph is not None and getattr(entry.aclgraph, 'tasks', None) == [])
        requested = (root / 'enable').exists()
        selected = bool(eligible and requested)
        if frame['observe']:
            frame['calls'].append({'wrapper': type(self).__name__, 'backend': getattr(backend, '__name__', str(backend)),
                                   'graph_tokens': getattr(descriptor, 'num_tokens', None),
                                   'static_sfa': static_sfa, 'eligible': eligible, 'requested': requested,
                                   'selected': selected, 'captured_entry_present': entry is not None,
                                   'host_update_tasks': len(entry.aclgraph.tasks) if entry is not None and entry.aclgraph is not None and hasattr(entry.aclgraph, 'tasks') else None})
        if not selected:
            return call_original(self, *args, **kwargs)
        # Exact static-replay tail of the pinned native __call__, excluding its
        # host stream synchronize. SFA update_graph_params is a no-op; neither
        # host update tasks nor device-metadata executor are admitted above.
        # Same graph object, current stream, input buffers, and output handle.
        if self.is_debugging_mode:
            addresses = [x.data_ptr() for x in args if isinstance(x, ns['torch'].Tensor)]
            assert addresses == entry.input_addresses
        entry.aclgraph.replay()
        deps['logger'].info_once('ACL graph replay is active (logged once).')
        return entry.output

    Graph.__call__ = call
