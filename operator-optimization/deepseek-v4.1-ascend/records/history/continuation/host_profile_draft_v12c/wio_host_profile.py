"""Same-call CPU markers and Python scalar metadata; no tensor read/copy."""
from pathlib import Path
import ast
import functools
import hashlib
import inspect
import json
import os
import threading
import textwrap
import time

SOURCE_SHA = '568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c'
MAX_CALLS_PER_PROCESS = 4096
MAX_PROCESS_BYTES = 64 * 1024**2
PROFILER_SOURCE_SHA = '5242ee153033afcf12dbfe022dbd807fa248d991ff5c53efbd1adb372d5575f4'
_lock = threading.Lock()
_ordinal = 0

def _emit(root, event):
    event.update(pid=os.getpid(), tid=threading.get_native_id(), epoch_ns=time.time_ns(), monotonic_ns=time.monotonic_ns())
    path = root / ('events-'+str(os.getpid())+'.jsonl')
    data = (json.dumps(event, ensure_ascii=False, separators=(',',':'))+'\n').encode()
    if path.exists() and path.stat().st_size + len(data) > MAX_PROCESS_BYTES:
        raise RuntimeError('Finite diagnostic metadata byte budget exceeded')
    fd = os.open(path, os.O_WRONLY|os.O_CREAT|os.O_APPEND, 0o600)
    try:
        written = os.write(fd, data)
        if written != len(data):
            raise RuntimeError('Incomplete diagnostic metadata write')
    finally:
        os.close(fd)

def _scalar(value):
    return value if value is None or type(value) in (str, int, bool, float) else type(value).__name__

def _scalar_sequence(value):
    if type(value) in (list, tuple, set):
        if len(value) > 4096:
            raise RuntimeError('Unexpected diagnostic scheduler metadata extent')
        return [_scalar(v) for v in value]
    return _scalar(value)

def install_profiler(module):
    """Change two collection flags in the pinned temporary profiler factory."""
    cls = module.TorchNPUProfilerWrapper
    original = cls._create_profiler
    if getattr(original, '_wio_host_profile', False):
        return
    source = Path(module.__file__)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != PROFILER_SOURCE_SHA:
        raise RuntimeError('Installed NPU profiler differs from the inspected constructor')
    root = Path(os.environ['WIO_HOST_PROFILE_ROOT'])
    root.mkdir(parents=True, exist_ok=True)
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    function = tree.body[0]
    function.decorator_list = []
    function.name = '_wio_create_profiler'
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)
             and ast.unparse(node.func) == 'torch_npu.profiler.profile']
    if len(calls) != 1:
        raise RuntimeError('Pinned profiler factory call no longer unique')
    call = calls[0]
    stack = [k for k in call.keywords if k.arg == 'with_stack']
    if len(stack) != 1 or not isinstance(stack[0].value, ast.Constant) or stack[0].value.value is not False:
        raise RuntimeError('Pinned with_stack default no longer matches')
    if any(k.arg == 'record_shapes' for k in call.keywords):
        raise RuntimeError('Pinned record_shapes omission no longer matches')
    stack[0].value = ast.Constant(value=True)
    call.keywords.append(ast.keyword(arg='record_shapes', value=ast.Constant(value=True)))
    ast.fix_missing_locations(tree)
    namespace = dict(module.__dict__)
    exec(compile(tree, str(Path(__file__).resolve())+':temporary_profiler_factory', 'exec'), namespace)
    transformed = namespace['_wio_create_profiler']

    @functools.wraps(original)
    def observed_factory(profiler_config, trace_name):
        profiler = transformed(profiler_config, trace_name)
        _emit(root, dict(event='profiler_created', trace_name=trace_name, source=str(source),
                         source_sha256=digest, forced_collection_kwargs=dict(with_stack=True, record_shapes=True),
                         unchanged_factory_settings=['activities','with_modules','profile_memory','experimental_config','trace_handler'],
                         object_attributes={name:_scalar(getattr(profiler, name, None)) for name in ['with_stack','record_shapes','with_modules','profile_memory']},
                         max_iterations=profiler_config.max_iterations))
        return profiler

    observed_factory._wio_host_profile = True
    cls._create_profiler = staticmethod(observed_factory)

def install(module):
    cls = module.NPUModelRunner
    if getattr(cls.execute_model, '_wio_host_profile', False):
        return
    source = Path(module.__file__)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != SOURCE_SHA:
        raise RuntimeError('Target worker source differs from pinned diagnostic contract')
    root = Path(os.environ['WIO_HOST_PROFILE_ROOT'])
    root.mkdir(parents=True, exist_ok=True)
    original = cls.execute_model
    _emit(root, dict(event='installed', source=str(source), source_sha256=digest, max_calls_per_process=MAX_CALLS_PER_PROCESS,
                     max_process_bytes=MAX_PROCESS_BYTES, no_tensor_values_or_device_reads=True))

    @functools.wraps(original)
    def observed(self, *args, **kwargs):
        global _ordinal
        if not (root/'ARMED.json').is_file():
            return original(self, *args, **kwargs)
        with _lock:
            _ordinal += 1
            call = _ordinal
        if call > MAX_CALLS_PER_PROCESS:
            raise RuntimeError('Finite diagnostic model-call budget exceeded')
        scheduler = kwargs.get('scheduler_output', args[0] if args else None)
        schedule = getattr(scheduler, 'num_scheduled_tokens', {})
        scheduled = {str(k): int(v) for k,v in schedule.items()} if isinstance(schedule, dict) else None
        config = getattr(self, 'parallel_config', None)
        spec = getattr(self, 'speculative_config', None)
        cached = getattr(scheduler, 'scheduled_cached_reqs', None)
        cached_state = {key:_scalar_sequence(getattr(cached, key, None))
                        for key in ['req_ids','resumed_req_ids','num_computed_tokens','num_output_tokens']}
        new_state = []
        for request in getattr(scheduler, 'scheduled_new_reqs', ()):
            new_state.append({key:_scalar(getattr(request, key, None))
                              for key in ['req_id','num_computed_tokens']})
        label = 'wio_hostdiag::execute_model/p'+str(os.getpid())+'/c'+str(call)
        _emit(root, dict(event='execute_enter', call_id=call, label=label,
                         scheduled_tokens=scheduled, total_scheduled_tokens=sum(scheduled.values()) if scheduled is not None else None,
                         cached_request_state=cached_state, new_request_state=new_state,
                         scheduled_spec_tokens=_scalar(getattr(scheduler, 'num_spec_tokens_to_schedule', None)),
                         rank=_scalar(getattr(config, 'rank', None)), data_parallel_rank=_scalar(getattr(config, 'data_parallel_rank', None)),
                         tensor_parallel_size=_scalar(getattr(config, 'tensor_parallel_size', None)),
                         speculative_method=_scalar(getattr(spec, 'method', None)),
                         use_aux_hidden_state_outputs=_scalar(getattr(self, 'use_aux_hidden_state_outputs', None))))
        begin_wall, begin_cpu = time.monotonic_ns(), time.thread_time_ns()
        error = None
        try:
            with module.torch.profiler.record_function(label):
                return original(self, *args, **kwargs)
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            end_wall, end_cpu = time.monotonic_ns(), time.thread_time_ns()
            _emit(root, dict(event='execute_exit', call_id=call, label=label, error_type=error,
                             wrapper_wall_ns=end_wall-begin_wall, wrapper_thread_cpu_ns=end_cpu-begin_cpu,
                             timing_is_instrumented_and_not_per_function_cost=True))

    observed._wio_host_profile = True
    cls.execute_model = observed
