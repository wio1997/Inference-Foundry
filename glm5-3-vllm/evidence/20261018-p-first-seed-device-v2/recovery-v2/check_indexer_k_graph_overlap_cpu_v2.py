"""Guard/ownership protocol checks; actual numerical proof is the NPU fragment."""
from contextlib import contextmanager
import importlib.util
import itertools
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS

os.environ['GLM53_P_FIRST_DRAFT_EXPORT'] = '0'
r = Path(__file__).resolve().parent
repo = r.parents[3]
spec = importlib.util.spec_from_file_location('helper', repo/'glm5-3-vllm/runtime-patches/p-first-seed-v2/indexer_k_graph_overlap.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
ptrs = itertools.count(100)


class Device:
    type = 'npu'

    def __str__(self):
        return 'npu:0'


device = Device()


class Tensor:
    def __init__(self, shape, value=0, dtype='bf16'):
        self.shape, self.value, self.dtype = tuple(shape), value, dtype
        self.device, self.ptr = device, next(ptrs)
        self.ndim = len(self.shape)

    def stride(self):
        result, acc = [], 1
        for n in reversed(self.shape):
            result.append(acc)
            acc *= n
        return tuple(reversed(result))

    def data_ptr(self):
        return self.ptr

    def is_contiguous(self):
        return True

    def numel(self):
        acc = 1
        for n in self.shape:
            acc *= n
        return acc

    def copy_(self, source, non_blocking=False):
        self.value = source.value
        return self


def fixture(root):
    events = []

    class Stream:
        def __init__(self, ident=None):
            self.npu_stream = next(ptrs) if ident is None else ident

        def wait_stream(self, other):
            events.append(('wait', self.npu_stream, other.npu_stream))

        def synchronize(self):
            events.append(('sync', self.npu_stream))

    class Graph:
        def replay(self):
            events.append(('replay',))
            self.run()

    npu = NS(Stream=Stream, NPUGraph=Graph, current=Stream(0), capture=None,
             graph_pool_handle=lambda: object())

    @contextmanager
    def stream(selected):
        before, npu.current = npu.current, selected
        try:
            yield
        finally:
            npu.current = before

    @contextmanager
    def graph(selected, pool, stream):
        before_stream, before_capture = npu.current, npu.capture
        npu.current, npu.capture = stream, selected
        try:
            yield
        finally:
            npu.current, npu.capture = before_stream, before_capture

    npu.stream, npu.graph, npu.current_stream = stream, graph, lambda: npu.current
    torch = NS(npu=npu, bfloat16='bf16', float32='fp32',
               empty_like=lambda t: Tensor(t.shape, dtype=t.dtype),
               equal=lambda a, b: (a.shape, a.dtype, a.value) == (b.shape, b.dtype, b.value))
    ctx = NS(capturing=False, cudagraph_runtime_mode='NONE')
    extra = NS(draft=True, capturing=False, sp=False)

    class Quant:
        pass

    class Indexer:
        def __init__(self):
            self.wk_weights_proj = NS(weight=Tensor((160,6144)))
            self.k_norm = NS(weight=Tensor((128,), dtype='fp32'), bias=Tensor((128,), dtype='fp32'))
            self.q_hadamard = Tensor((128,128))
            self.head_dim = 128
            self._pcp_active = self._dsa_cp_active = self.enable_sparse_li_c4 = False
            self.enable_sparse_li_c8 = True
            self.fail = False

        def forward(self, hidden, cos, sin):
            return self.forward_k(hidden, cos, sin)

        def forward_k(self, hidden, cos, sin):
            if self.fail:
                raise ValueError('native K warmup failure')
            outputs = (Tensor((hidden.shape[0],128)), Tensor((hidden.shape[0],1)), Tensor((hidden.shape[0],32)))

            def run():
                outputs[0].value = hidden.value*2 + cos.value + sin.value
                outputs[1].value = hidden.value/8
                outputs[2].value = hidden.value*3
            run()
            if npu.capture is not None:
                npu.capture.run = run
            events.append(('native-K',))
            return outputs

    class Wrapper:
        def __init__(self, impl):
            object.__setattr__(self, 'impl', impl)
        def forward(self, hidden, cos, sin):
            return self.impl.forward(hidden, cos, sin)
        def __getattr__(self, name):
            return getattr(self.impl, name)
        def __setattr__(self, name, value):
            if name == 'impl':
                object.__setattr__(self, name, value)
            else:
                setattr(self.impl, name, value)

    class SFA:
        def __init__(self):
            self.indexer = Wrapper(Indexer())
            self.hidden = Tensor((6,6144), 1)
            self.metadata = NS(cos=Tensor((6,1,1,64), 2), sin=Tensor((6,1,1,64), 3))
            self.preprocess_type, self._quant_type = 'PROLOG', Quant
            self.enable_sparse_sfa_c8 = self.runtime_has_indexer = self.has_indexer = True
            self.enable_sparse_sfa_turboquant = False
            self.local_num_heads, self.kv_lora_rank, self.qk_rope_head_dim = 4,512,64
            self.fail_after_prolog = self.skip_consumer = False

        def _get_indexer_attn_metadata(self):
            return self.metadata

        def _sfa_preprocess_prolog_v3(self, hidden_states, kv_cache, cos, sin, slot_mapping, *, attn_metadata=None):
            events.append(('native-Prolog',))
            return hidden_states, hidden_states.value*7, None, None

        def forward(self):
            q = self._sfa_preprocess_prolog_v3(self.hidden, (), self.metadata.cos, self.metadata.sin, None)
            if self.fail_after_prolog:
                raise ValueError('native after-Prolog failure')
            if self.skip_consumer:
                return q
            k = self.indexer.forward(self.hidden, self.metadata.cos, self.metadata.sin)
            events.append(('native-cache-write', k[0].value))
            return q[0], q[1], tuple(t.value for t in k)

    class Runner:
        def __init__(self):
            self.sfa = SFA()
            self.parallel_config = NS(tensor_parallel_size=16, data_parallel_size=1,
                                      pipeline_parallel_size=1, decode_context_parallel_size=1,
                                      prefill_context_parallel_size=1)
            self.input_batch = NS(req_ids=['r'])
            self.requests = {'r':NS(sampling_params=NS(max_tokens=64))}
            self.is_kv_consumer, self.is_kv_producer, self.use_async_scheduling = True, False, True
            self.num_spec_tokens, self.dynamic_eplb, self.tp_rank = 5, False, 0
            self.model_config = NS(use_mla=True, is_hybrid=False)

        def sample_tokens(self):
            return self.sfa.forward()

    state = helper.install(dict(torch=torch, NPUModelRunner=Runner,
                                CUDAGraphMode=NS(FULL='FULL'), enable_sp=lambda: extra.sp), root=root,
                           deps=dict(SFA=SFA, Indexer=Indexer, Wrapper=Wrapper, quant=Quant, prolog_type='PROLOG',
                                     ctx=lambda: ctx, available=lambda: True,
                                     is_draft=lambda: extra.draft, capturing=lambda: extra.capturing))
    return NS(runner=Runner(), state=state, ctx=ctx, extra=extra, events=events, root=root)


checks = []
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    f = fixture(root/'first')
    original = f.runner.sample_tokens()
    assert original[0] is f.runner.sfa.hidden and not f.state['entries'] and not f.state['pending']
    checks.append('marker-off retains original hidden identity and no graph/copy')
    (f.root/'enable').touch()
    selected = f.runner.sample_tokens()
    assert selected == original and len(f.state['entries']) == 1 and not f.state['pending']
    checks.append('model-facing Wrapper.impl binding reaches one backend consumer with current-native equality')
    for value in [5, -7, 11]:
        f.runner.sfa.hidden = Tensor((6,6144), value)
        f.runner.sfa.metadata = NS(cos=Tensor((6,1,1,64), value+2), sin=Tensor((6,1,1,64), value-3))
        out = f.runner.sample_tokens()
        assert out[0] is f.runner.sfa.hidden and out[1] == value*7
        assert out[2][0] == value*4-1 and len(f.state['entries']) == 1
    checks.append('same-shape replaced source buffers and changing values refresh owned inputs')
    assert any(e[0]=='wait' and e[1]==0 for e in f.events)
    last = max(i for i,e in enumerate(f.events) if e[0]=='native-cache-write')
    assert any(e[0]=='wait' and e[1]==0 for e in f.events[:last])
    checks.append('main waits for auxiliary producer before original cache consumer')

    mutations = [
        ('target FULL graph', lambda z:setattr(z.ctx,'cudagraph_runtime_mode','FULL')),
        ('capture context', lambda z:setattr(z.ctx,'capturing',True)),
        ('extra capture context', lambda z:setattr(z.extra,'capturing',True)),
        ('target/non-draft', lambda z:setattr(z.extra,'draft',False)),
        ('sequence parallel', lambda z:setattr(z.extra,'sp',True)),
        ('PCP', lambda z:setattr(z.runner.sfa.indexer,'_pcp_active',True)),
        ('DSA CP', lambda z:setattr(z.runner.sfa.indexer,'_dsa_cp_active',True)),
        ('single token request', lambda z:setattr(z.runner.requests['r'].sampling_params,'max_tokens',1)),
        ('unsupported input N', lambda z:setattr(z.runner.sfa,'hidden',Tensor((1,6144),3))),
    ]
    for i,(name,mutate) in enumerate(mutations):
        z = fixture(root/('guard'+str(i)))
        (z.root/'enable').touch()
        mutate(z)
        z.runner.sample_tokens()
        assert not z.state['entries'] and not z.state['pending'], name
        checks.append(name+' stays native')
    for field in ['fail_after_prolog','skip_consumer']:
        z=fixture(root/field)
        (z.root/'enable').touch()
        setattr(z.runner.sfa,field,True)
        try:
            z.runner.sample_tokens()
        except (ValueError,RuntimeError):
            pass
        else:
            raise AssertionError(field)
        assert not z.state['pending'] and any(e[0]=='wait' and e[1]==0 for e in z.events)
        checks.append(field+' joins producer and clears scope')
    z=fixture(root/'coldfail')
    (z.root/'enable').touch()
    z.runner.sfa.indexer.fail=True
    try:
        z.runner.sample_tokens()
    except ValueError:
        pass
    else:
        raise AssertionError('cold capture failure swallowed')
    assert not z.state['pending'] and any(e[0]=='wait' and e[1]==0 for e in z.events)
    checks.append('cold warmup failure joins auxiliary before releasing source scope')

result = dict(passed=len(checks), checks=checks, scope='CPU fake execution/ownership protocol only. Native executable AST and numerical correctness are independently checked on NPU. No model quality or performance claim.')
(r/'indexer-k-graph-overlap-cpu-check-v2.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
