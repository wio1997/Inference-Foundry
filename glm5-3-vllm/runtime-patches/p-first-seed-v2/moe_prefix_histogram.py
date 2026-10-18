"""Ask existing AllGather routing for prefix histograms before unchanged GMMs.

Two target graph families are captured during native startup, so ordinary tests
can alternate count and prefix paths without reloading model weights. Candidate
has a separate graph pool; original graph objects and replay ordering stay native.
"""
import functools
import dataclasses
import hashlib
import inspect
import json
import os
from pathlib import Path
import textwrap
import time


def install(ns, root=None, deps=None):
    assert os.environ.get('GLM53_P_FIRST_DRAFT_EXPORT') == '0'
    if deps is None:
        from vllm.forward_context import get_forward_context
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX
        from vllm_ascend.attention import sfa_v1
        from vllm_ascend.compilation import acl_graph
        from vllm_ascend.ops.fused_moe import token_dispatcher, fused_moe
        assert hashlib.sha256(Path(fused_moe.__file__).read_bytes()).hexdigest() == '2f42ce56a7636bf618204dbc2c1aeb2a2c95fed69fe9e5bcfac995354a00beb4'
        assert hashlib.sha256(Path(token_dispatcher.__file__).read_bytes()).hexdigest() == '1dcfd8181cb2d34c584d8336c3379dbeefba75c0bf664f924063701b93295cbb'
        assert hashlib.sha256(Path(acl_graph.__file__).read_bytes()).hexdigest() == '8e1c269e331d005f1361acc5cec24402bff7ae39b75e952ea890da7812804955'
        assert hashlib.sha256(Path(sfa_v1.__file__).read_bytes()).hexdigest() == '9b94f67f30364d40caf4a7a67af0f0365d1ba0b5048939873ab0464e95b096dc'
        original = token_dispatcher.TokenDispatcherWithAllGather.token_dispatch
        source = textwrap.dedent(inspect.getsource(original))
        assert source.count('expert_tokens_num_type=1,') == 1
        assert source.count('group_list_type = 1  # `count` mode') == 1
        source = source.replace('expert_tokens_num_type=1,', 'expert_tokens_num_type=0,')
        source = source.replace('group_list_type = 1  # `count` mode', 'group_list_type = 0  # prefix from the same routing operator')
        local = {}
        exec(compile(source, '<glm53-routing-prefix>', 'exec'), original.__globals__, local)
        deps = {'Graph': acl_graph.ACLGraphWrapper, 'Dispatcher': token_dispatcher.TokenDispatcherWithAllGather,
                'candidate': local['token_dispatch'], 'get_context': get_forward_context,
                'is_draft': lambda: _EXTRA_CTX.is_draft_model, 'SFA': sfa_v1.AscendSFABackend,
                'updatable': acl_graph.use_updatable_graph, 'W8A8': token_dispatcher.QuantType.W8A8,
                'has_lora': token_dispatcher.has_lora, 'Core': fused_moe.AscendMoERunner}
    torch, Runner, Mode = ns['torch'], ns['NPUModelRunner'], ns['CUDAGraphMode']
    Graph, Dispatcher = deps['Graph'], deps['Dispatcher']
    root = Path('/out/moe-prefix-histogram-probe') if root is None else Path(root)
    root.mkdir(exist_ok=True)
    active = {'frame': None, 'prefix': False, 'capture': None, 'building_family': None}
    states = {}  # Strongly own all comparison families and private pools.
    records = 0

    def config_ok(config):
        p, model, kv, spec = config.parallel_config, config.model_config, config.kv_transfer_config, config.speculative_config
        return bool(kv is not None and kv.kv_role == 'kv_consumer' and spec is not None
                    and spec.num_speculative_tokens == 5 and config.scheduler_config.max_num_seqs == 4
                    and p.tensor_parallel_size == 16 and p.data_parallel_size == 1
                    and p.pipeline_parallel_size == 1 and p.decode_context_parallel_size == 1
                    and p.prefill_context_parallel_size == 1 and not ns['enable_sp']()
                    and model.use_mla and not model.is_hybrid and not getattr(config, 'lora_config', None)
                    and not getattr(p, 'enable_eplb', False) and not getattr(p, 'enable_expert_parallel', False))

    def real_frame(frame):
        if frame is None:
            return False
        runner = frame['runner']
        ids = list(runner.input_batch.req_ids)
        return bool(ids and all(rid in runner.requests and runner.requests[rid].sampling_params is not None
                               and runner.requests[rid].sampling_params.max_tokens > 1 for rid in ids))

    def output_tensors(value, path='out'):
        if isinstance(value, torch.Tensor):
            return [(path, value)]
        if isinstance(value, (list, tuple)):
            return [z for i, v in enumerate(value) for z in output_tensors(v, path + '.' + str(i))]
        if isinstance(value, dict):
            return [z for k, v in sorted(value.items()) for z in output_tensors(v, path + '.' + str(k))]
        return []

    def snapshot(value):
        # Diagnostic only, outside capture and timed requests; preserve raw BF16 bits.
        return [(path, tensor.detach().view(torch.int16).cpu() if tensor.dtype == torch.bfloat16
                 else tensor.detach().cpu()) for path, tensor in output_tensors(value)]

    def compare_snapshots(left, right):
        return bool(len(left) == len(right) and left and all(a == b and torch.equal(x, y)
                    for (a, x), (b, y) in zip(left, right)))

    def input_inventory(value):
        result, seen = {}, set()
        def walk(obj, path, depth):
            if len(result) >= 512 or depth > 7:
                return
            if isinstance(obj, torch.Tensor):
                if obj.device.type == 'npu':
                    result[path] = {'ptr': obj.data_ptr(), 'shape': list(obj.shape),
                                    'stride': list(obj.stride()), 'dtype': str(obj.dtype)}
                return
            if id(obj) in seen:
                return
            seen.add(id(obj))
            if isinstance(obj, dict):
                for key, item in sorted(obj.items(), key=lambda z: str(z[0])):
                    walk(item, path + '.' + str(key), depth + 1)
            elif isinstance(obj, (list, tuple)):
                for i, item in enumerate(obj):
                    walk(item, path + '.' + str(i), depth + 1)
            elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                for field in dataclasses.fields(obj):
                    walk(getattr(obj, field.name), path + '.' + field.name, depth + 1)
        walk(value, 'inputs', 0)
        return result

    native_dispatch = Dispatcher.token_dispatch

    @functools.wraps(native_dispatch)
    def dispatch(self, token_dispatch_input):
        frame = active['frame']
        x = token_dispatch_input.hidden_states
        permitted = bool(token_dispatch_input.quant.quant_type == deps['W8A8']
                         and not deps['has_lora'](self.lora_context)
                         and x.device.type == 'npu' and x.dtype == torch.bfloat16
                         and x.ndim == 2 and 0 < x.shape[0] <= 24 and x.shape[1] == 6144
                         and self.num_experts_local == 256 and self.top_k == 8
                         and token_dispatch_input.routing.expert_map is None
                         and not token_dispatch_input.routing.global_redundant_expert_num)
        selected = bool(active['prefix'] and permitted and (active['capture'] is not None or real_frame(frame)))
        output = (deps['candidate'] if selected else native_dispatch)(self, token_dispatch_input)
        if active['capture'] is not None:
            cap = active['capture'];cap['calls'] += 1
            cap['selected_calls'] += int(selected)
            cap['output_list_types'].add(output.group_list_type)
        elif frame is not None and frame['observe'] and real_frame(frame):
            frame['dispatch'].append({'prefix_requested': active['prefix'], 'selected': selected,
                                      'N': x.shape[0], 'output_list_type': output.group_list_type})
        return output

    Dispatcher.token_dispatch = dispatch
    graph_original = Graph.__call__

    @functools.wraps(graph_original)
    def graph_call(self, *args, **kwargs):
        ctx = deps['get_context']()
        desc = ctx.batch_descriptor
        n = getattr(desc, 'num_tokens', None)
        backend = self.attn_backend
        eligible = bool(self.runtime_mode == Mode.FULL and ctx.cudagraph_runtime_mode == Mode.FULL
                        and not ctx.capturing and not deps['is_draft']() and not self.enable_enpu
                        and config_ok(self.vllm_config) and getattr(ctx, 'device_metadata_executor', None) is None
                        and isinstance(backend, type) and issubclass(backend, deps['SFA'])
                        and not deps['updatable'](backend) and n in [6, 12, 18, 24])
        if not eligible:
            return graph_original(self, *args, **kwargs)
        key = id(self)
        if key not in states:
            states[key] = {'wrapper': self, 'native': self.concrete_aclgraph_entries, 'prefix': {}, 'replica': {},
                           'prefix_pool': torch.npu.graph_pool_handle(), 'replica_pool': torch.npu.graph_pool_handle(),
                           'captures': {}, 'inventories_logged': set()}
        state = states[key]
        assert state['wrapper'] is self and self.concrete_aclgraph_entries is state['native']
        entry = state['native'].get(desc)
        building = active['building_family']
        if building is not None:
            # Each family comes through the original _warmup_and_capture, so
            # _dummy_run constructs a fresh forward context and metadata.
            previous = active['prefix'], active['capture']
            native_pool = self.graph_pool
            cap = {'calls': 0, 'selected_calls': 0, 'output_list_types': set(),
                   'input_inventory': input_inventory({'args': args, 'kwargs': kwargs,
                                                       'attn': getattr(ctx, 'attn_metadata', None)})}
            try:
                active['prefix'] = building == 'prefix'
                active['capture'] = cap
                self.concrete_aclgraph_entries = state[building]
                self.graph_pool = native_pool if building == 'native' else state[building + '_pool']
                output = graph_original(self, *args, **kwargs)
                captured = state[building][desc]
                assert captured.aclgraph is not None and getattr(captured.aclgraph, 'tasks', None) == []
                cap['output_list_types'] = sorted(cap['output_list_types'])
                cap['output_pointers'] = [{'path': path, 'ptr': tensor.data_ptr(), 'shape': list(tensor.shape)}
                                          for path, tensor in output_tensors(output)]
                state['captures'][(building, n)] = cap
                return output
            finally:
                self.concrete_aclgraph_entries = state['native']
                self.graph_pool = native_pool
                active['prefix'], active['capture'] = previous
        if entry is None or entry.aclgraph is None:
            return graph_original(self, *args, **kwargs)
        frame = active['frame']
        requested = bool(frame is not None and frame['requested'] and real_frame(frame))
        requested_family = 'replica' if (root / 'replica-enable').exists() and real_frame(frame) else ('prefix' if requested and not (root / 'MTP-only').exists() else 'native')
        candidate = state[requested_family].get(desc)
        cap = state['captures'].get((requested_family, n), {})
        ready = bool(candidate is not None and candidate.aclgraph is not None and cap.get('calls') == 75
                     and cap.get('selected_calls') == (75 if requested_family == 'prefix' else 0)
                     and cap.get('output_list_types') == ([0] if requested_family == 'prefix' else [1]))
        family = requested_family if ready else 'native'
        selected = family != 'native'
        if frame is not None and frame['observe'] and real_frame(frame):
            brief = lambda value: {k: v for k, v in value.items() if k != 'input_inventory'}
            row = {'tokens': n, 'requested': requested, 'selected': selected,
                   'native_capture': brief(state['captures'].get(('native', n), {})),
                   'prefix_capture': brief(state['captures'].get(('prefix', n), {})),
                   'selected_family': family,
                   'replica_capture': brief(state['captures'].get(('replica', n), {}))}
            if (family, n) not in state['inventories_logged']:
                row['capture_input_inventories'] = {name: state['captures'].get((name, n), {}).get('input_inventory', {})
                                                     for name in ['native', 'replica', 'prefix']}
                row['runtime_input_inventory'] = input_inventory({'args': args, 'kwargs': kwargs,
                                                                  'attn': getattr(ctx, 'attn_metadata', None)})
                state['inventories_logged'].add((family, n))
            frame['graphs'].append(row)
        previous = active['prefix']
        try:
            active['prefix'] = family == 'prefix'
            if selected:
                self.concrete_aclgraph_entries = state[family]
            return graph_original(self, *args, **kwargs)
        finally:
            self.concrete_aclgraph_entries = state['native']
            active['prefix'] = previous

    Graph.__call__ = graph_call

    warmup_original = Runner._warmup_and_capture

    @functools.wraps(warmup_original)
    def warmup(self, desc, cudagraph_runtime_mode, *args, **kwargs):
        eligible = bool(cudagraph_runtime_mode == Mode.FULL and desc.num_tokens in [6, 12, 18, 24]
                        and self.is_kv_consumer and not self.is_kv_producer and not self.dynamic_eplb
                        and self.use_async_scheduling and config_ok(self.vllm_config))
        if not eligible:
            return warmup_original(self, desc, cudagraph_runtime_mode, *args, **kwargs)
        previous = active['building_family'], active['prefix'], active['capture']
        result = None
        try:
            for family in ['native', 'replica', 'prefix']:
                active['building_family'] = family
                active['prefix'] = family == 'prefix'
                # A private non-real scope also selects the variant during the
                # framework's eager warmups, before their capture begins.
                active['capture'] = {'calls': 0, 'selected_calls': 0, 'output_list_types': set()}
                options = dict(kwargs)
                options['num_warmups'] = max(2, options.get('num_warmups') or 0,
                                             self.compilation_config.cudagraph_num_of_warmups)
                output = warmup_original(self, desc, cudagraph_runtime_mode, *args, **options)
                if family == 'native':
                    result = output
            return result
        finally:
            active['building_family'], active['prefix'], active['capture'] = previous

    Runner._warmup_and_capture = warmup

    def wrap(method, phase):
        @functools.wraps(method)
        def run(self, *args, **kwargs):
            nonlocal records
            if not (self.is_kv_consumer and not self.is_kv_producer and self.use_async_scheduling
                    and not self.dynamic_eplb and config_ok(self.vllm_config)):
                return method(self, *args, **kwargs)
            frame = {'runner': self, 'phase': phase, 'requested': (root / 'enable').exists(),
                     'observe': not (root / 'disabled').exists(), 'dispatch': [], 'graphs': []}
            previous = active['frame'], active['prefix']
            active['frame'] = frame
            active['prefix'] = bool(frame['requested'] and phase == 'sample_tokens' and not (root / 'target-only').exists())
            try:
                return method(self, *args, **kwargs)
            finally:
                active['frame'], active['prefix'] = previous
                if frame['observe'] and (frame['dispatch'] or frame['graphs']) and records < 4000:
                    records += 1
                    frame['request_ids'] = list(self.input_batch.req_ids)
                    frame.pop('runner')
                    frame.update(rank=self.tp_rank, pid=os.getpid(), monotonic_ns=time.monotonic_ns())
                    with (root / ('rank-%d-%d.jsonl' % (self.tp_rank, os.getpid()))).open('a') as out:
                        out.write(json.dumps(frame, sort_keys=True) + '\n')
        return run

    if deps.get('Core') is not None:
        Core = deps['Core']
        core_original = Core._forward_impl
        diagnosed = {False: 0, True: 0}

        @functools.wraps(core_original)
        def core(self, *args, **kwargs):
            nonlocal diagnosed
            frame = active['frame']
            serial = (root / 'diagnose-serial').exists()
            diagnose = bool(frame is not None and frame['phase'] == 'sample_tokens' and real_frame(frame)
                            and (root / 'diagnose-MTP').exists() and diagnosed[serial] < 2)
            if not diagnose:
                return core_original(self, *args, **kwargs)
            diagnosed[serial] += 1
            previous = active['prefix']
            shared = self.ascend_shared_experts
            previous_overlap = getattr(shared, 'multistream_overlap', None)
            try:
                if serial and previous_overlap is not None:
                    shared.multistream_overlap = False
                active['prefix'] = True
                candidate = core_original(self, *args, **kwargs)
                candidate_values = snapshot(candidate)
                active['prefix'] = False
                native = core_original(self, *args, **kwargs)
                native_values = snapshot(native)
                row = {'rank': frame['runner'].tp_rank, 'pid': os.getpid(), 'serial_shared': serial,
                       'bitwise_equal': compare_snapshots(candidate_values, native_values),
                       'candidate_output_ptrs': [t.data_ptr() for _, t in output_tensors(candidate)],
                       'native_output_ptrs': [t.data_ptr() for _, t in output_tensors(native)],
                       'N': args[0].shape[0] if args else kwargs['hidden_states'].shape[0],
                       'scope': 'Same real MTP opaque core inputs, candidate first then native; CPU snapshots before later reuse. No KV or attention repeated, no service timing.'}
                with (root / ('diagnostic-rank-%d-%d.jsonl' % (frame['runner'].tp_rank, os.getpid()))).open('a') as out:
                    out.write(json.dumps(row, sort_keys=True) + '\n')
                if serial and previous_overlap is not None:
                    shared.multistream_overlap = previous_overlap
                    native = core_original(self, *args, **kwargs)
                return native
            finally:
                if previous_overlap is not None:
                    shared.multistream_overlap = previous_overlap
                active['prefix'] = previous

        Core._forward_impl = core

    Runner.execute_model = wrap(Runner.execute_model, 'execute_model')
    Runner.sample_tokens = wrap(Runner.sample_tokens, 'sample_tokens')
