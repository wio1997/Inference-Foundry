"""Opt-in per-process diagnostic hooks; never load model modules at startup.

Set WIO_DIAG_DIR and create its PROFILE_ACTIVE only around the capture.  Set
WIO_DIAG_HMAC_KEY to the same private secret in all relevant processes if
request correlation is required; raw IDs and tensor data are never written.
This is an experimental diagnostic overlay, not framework code.
"""
import functools
import hashlib
import hmac
import importlib.abc
import importlib.machinery
import itertools
import json
import os
import re
import sys
import threading
import time

_DIR = os.environ.get('WIO_DIAG_DIR', '')
_GATE = os.path.join(_DIR, 'PROFILE_ACTIVE') if _DIR else ''
_KEY = os.environ.get('WIO_DIAG_HMAC_KEY', '').encode()
_CAP = 10000
_counter = itertools.count(1)
_lock = threading.Lock()
_enabled = False
_written = 0
_errors = 0
_TARGETS = {
    'vllm_ascend.profiler.torch_npu_profiler': [('TorchNPUProfilerWrapper', '_profiler')],
    'vllm.v1.core.sched.scheduler': [('Scheduler', 'schedule')],
    'vllm_ascend.worker.model_runner_v1': [('NPUModelRunner', 'execute_model')],
    'vllm_ascend.models.deepseek_v41.model': [('DeepseekV41DecoderLayer', 'forward')],
    'vllm_ascend.models.deepseek_v41.dspark': [('DeepseekV41DSparkModel', 'forward')],
}

def _emit(kind, **fields):
    global _written, _errors
    if not _DIR:
        return
    try:
        with _lock:
            if kind != 'instrumentation_error' and _written >= _CAP:
                return
            if kind == 'instrumentation_error':
                _errors += 1
                if _errors > 100:
                    return
            else:
                _written += 1
            obj = {'kind': kind, 'pid': os.getpid(), 'tid': threading.get_native_id(),
                   'perf_counter_ns': time.perf_counter_ns(), 'time_ns': time.time_ns(), **fields}
            line = (json.dumps(obj, separators=(',', ':'), ensure_ascii=True) + '\n').encode()
            fd = os.open(os.path.join(_DIR, 'events-%d.jsonl' % os.getpid()),
                         os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)
    except Exception as exc:
        # A failed sidecar must not interrupt model execution. If the event
        # file cannot be written, surface it on stderr rather than hide it.
        try:
            sys.stderr.write('WIO_INSTRUMENTATION_ERROR log_write %s\n' % type(exc).__name__)
        except Exception:
            pass

def _error(where, exc):
    _emit('instrumentation_error', where=where, error_type=type(exc).__name__)

def _active(scheduler=False):
    try:
        torch = sys.modules.get('torch')
        compiler = getattr(torch, 'compiler', None) if torch else None
        if compiler is not None and compiler.is_compiling():
            return False
        dynamo = sys.modules.get('torch._dynamo')
        if dynamo is not None and dynamo.is_compiling():
            return False
        if not _GATE or not (scheduler or _enabled):
            return False
        return os.path.isfile(_GATE)
    except Exception as exc:
        _error('active_check', exc)
        return False

def _request_stats(obj, scheduler=None):
    if obj is None:
        return None
    counts = getattr(obj, 'num_scheduled_tokens', None)
    if not isinstance(counts, dict):
        return None
    def digest_id(req_id):
        if not isinstance(req_id, str) or not _KEY:
            return None
        return hmac.new(_KEY, req_id.encode(), hashlib.sha256).hexdigest()
    def integer(v):
        return v if type(v) is int else None
    new = {}
    for req in getattr(obj, 'scheduled_new_reqs', ()):
        rid = getattr(req, 'req_id', None)
        if isinstance(rid, str):
            new[rid] = integer(getattr(req, 'num_computed_tokens', None))
    cached = {}
    cached_reqs = getattr(obj, 'scheduled_cached_reqs', None)
    if cached_reqs is not None:
        ids = getattr(cached_reqs, 'req_ids', ())
        nums = getattr(cached_reqs, 'num_computed_tokens', ())
        if isinstance(ids, (list, tuple)) and isinstance(nums, (list, tuple)):
            for rid, n in zip(ids, nums):
                if isinstance(rid, str):
                    cached[rid] = integer(n)
    specs = getattr(obj, 'scheduled_spec_decode_tokens', {})
    if not isinstance(specs, dict):
        specs = {}
    out = {'scheduled_total': integer(getattr(obj, 'total_num_scheduled_tokens', None)), 'requests': []}
    for req_id, count in counts.items():
        if not isinstance(req_id, str) or type(count) is not int:
            continue
        spec = specs.get(req_id)
        row = {'request_hmac_sha256': digest_id(req_id), 'scheduled_tokens': count,
               'new_computed_tokens': new.get(req_id), 'cached_computed_tokens': cached.get(req_id),
               'spec_token_count': len(spec) if isinstance(spec, (list, tuple)) else None}
        out['requests'].append(row)
    out['requests'].sort(key=lambda r: (r['request_hmac_sha256'] or '', r['scheduled_tokens']))
    if _KEY:
        canonical = json.dumps(out, sort_keys=True, separators=(',', ':')).encode()
        out['schedule_digest_sha256'] = hashlib.sha256(canonical).hexdigest()
    else:
        out['request_ids_omitted_no_hmac_key'] = True
        out['schedule_digest_sha256'] = None
    # Scheduler.schedule may update state before return; do not treat these as
    # pre-step values or incorporate them into the cross-process digest.
    if scheduler is not None:
        requests = getattr(scheduler, 'requests', None)
        if isinstance(requests, dict):
            states = []
            for req_id in counts:
                req = requests.get(req_id)
                if req is not None:
                    states.append({'request_hmac_sha256': digest_id(req_id),
                                   'num_computed_tokens_post_schedule': integer(getattr(req, 'num_computed_tokens', None)),
                                   'num_tokens': integer(getattr(req, 'num_tokens', None)),
                                   'num_prompt_tokens': integer(getattr(req, 'num_prompt_tokens', None))})
            out['scheduler_state_after_schedule'] = states
    return out

def _tensor_metadata(args):
    torch = sys.modules.get('torch')
    tensor_type = getattr(torch, 'Tensor', None) if torch else None
    if tensor_type is None:
        return []
    result = []
    for value in args[:8]:
        if isinstance(value, tensor_type):
            result.append({'shape': list(value.shape), 'dtype': str(value.dtype), 'device_type': value.device.type})
    return result

def _wrap_method(cls, method, scheduler=False):
    original = cls.__dict__.get(method)
    if not callable(original) or getattr(original, '_wio_diagnostic', False):
        return
    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        if not _active(scheduler):
            return original(self, *args, **kwargs)
        call_id = '%d-%d' % (os.getpid(), next(_counter))
        label = cls.__name__ + '.' + method
        extra = {}
        try:
            layer_idx = getattr(self, 'layer_idx', None)
            if type(layer_idx) is int:
                extra['layer_index'] = layer_idx
            prefix = getattr(self, 'layer_name', None) or getattr(self, 'prefix', None)
            if isinstance(prefix, str):
                match = re.search(r'(?:^|\.)layers\.(\d+)(?:\.|$)', prefix)
                if match:
                    extra['layer_index'] = int(match.group(1))
            if method == 'execute_model':
                candidate = args[0] if args else kwargs.get('scheduler_output')
                extra['schedule'] = _request_stats(candidate)
            extra['tensor_metadata'] = _tensor_metadata(args)
        except Exception as exc:
            _error('metadata_' + label, exc)
        _emit('call_begin', call_id=call_id, label=label, **extra)
        context = None
        entered = False
        try:
            try:
                torch = sys.modules.get('torch')
                if torch is not None:
                    context = torch.profiler.record_function('wio_diag:' + label + ':' + call_id)
                    context.__enter__()
                    entered = True
            except Exception as exc:
                _error('record_function_enter', exc)
            try:
                result = original(self, *args, **kwargs)
            except BaseException as exc:
                _emit('call_end', call_id=call_id, label=label, outcome='exception', exception_type=type(exc).__name__)
                raise
            if scheduler:
                try:
                    extra = {'schedule': _request_stats(result, scheduler=self)}
                except Exception as exc:
                    extra = {}
                    _error('scheduler_metadata', exc)
            else:
                extra = {}
            _emit('call_end', call_id=call_id, label=label, outcome='ok', **extra)
            return result
        finally:
            if entered:
                try:
                    context.__exit__(None, None, None)
                except Exception as exc:
                    _error('record_function_exit', exc)
    wrapped._wio_diagnostic = True
    setattr(cls, method, wrapped)

def _patch_profiler(module):
    cls = module.TorchNPUProfilerWrapper
    original_factory = cls._create_profiler
    def factory(config, trace_name):
        try:
            # Preserve the upstream profiler-type, directory and daemon guard.
            if config.profiler != 'torch' or not config.torch_profiler_dir:
                return original_factory(config, trace_name)
            try:
                use_daemon = module.get_ascend_config().msmonitor_use_daemon
            except RuntimeError:
                use_daemon = False
            if use_daemon:
                return original_factory(config, trace_name)
            npu = module.torch_npu
            experimental = npu.profiler._ExperimentalConfig(
                export_type=npu.profiler.ExportType.Text,
                profiler_level=npu.profiler.ProfilerLevel.Level1,
                msprof_tx=True, aic_metrics=npu.profiler.AiCMetrics.PipeUtilization,
                l2_cache=False, op_attr=True, data_simplification=False,
                record_op_args=True, gc_detect_threshold=None)
            p = npu.profiler.profile(
                activities=[npu.profiler.ProfilerActivity.CPU, npu.profiler.ProfilerActivity.NPU],
                record_shapes=True, with_stack=True, with_modules=True,
                profile_memory=config.torch_profiler_with_memory,
                experimental_config=experimental,
                on_trace_ready=npu.profiler.tensorboard_trace_handler(config.torch_profiler_dir, worker_name=trace_name))
            _emit('profiler_factory', configuration='diagnostic_requested', note='confirm effective profiler_info and export')
            return p
        except Exception as exc:
            _error('profiler_factory_fallback_original', exc)
            return original_factory(config, trace_name)
    cls._create_profiler = staticmethod(factory)
    for name in ('_start', '_stop'):
        original = cls.__dict__.get(name)
        if not callable(original):
            continue
        def make_hook(fn, is_start):
            @functools.wraps(fn)
            def hook(self, *args, **kwargs):
                global _enabled
                if is_start:
                    result = fn(self, *args, **kwargs)
                    _enabled = True
                    _emit('profiler_start_anchor')
                    return result
                try:
                    _emit('profiler_stop_anchor')
                    return fn(self, *args, **kwargs)
                finally:
                    _enabled = False
            return hook
        setattr(cls, name, make_hook(original, name == '_start'))

def _patch_module(module):
    try:
        entries = _TARGETS[module.__name__]
        for class_name, method in entries:
            cls = getattr(module, class_name, None)
            if cls is None:
                _emit('instrumentation_error', where=module.__name__, error_type='MissingClass:' + class_name)
                continue
            if method == '_profiler':
                _patch_profiler(module)
            elif method not in cls.__dict__:
                _emit('instrumentation_error', where=module.__name__ + '.' + class_name, error_type='MethodNotDefined:' + method)
            else:
                _wrap_method(cls, method, scheduler=class_name == 'Scheduler')
    except Exception as exc:
        _error('patch_module:' + module.__name__, exc)

class _Loader(importlib.abc.Loader):
    def __init__(self, inner):
        self.inner = inner
    def create_module(self, spec):
        fn = getattr(self.inner, 'create_module', None)
        return fn(spec) if fn else None
    def exec_module(self, module):
        self.inner.exec_module(module)
        _patch_module(module)
    def __getattr__(self, name):
        return getattr(self.inner, name)

class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec and spec.loader and hasattr(spec.loader, 'exec_module'):
            spec.loader = _Loader(spec.loader)
        return spec

if _DIR:
    sys.meta_path.insert(0, _Finder())
    _emit('instrumentation_loaded', gate=_GATE, request_hmac_key_present=bool(_KEY), max_events=_CAP)
