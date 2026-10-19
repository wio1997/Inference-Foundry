"""Host-control checks, not NPU correctness or graph-capture compatibility."""
from pathlib import Path
from types import SimpleNamespace as S
from contextlib import contextmanager
import importlib.util
import json
import os
import tempfile

repo = Path(__file__).resolve().parents[4]
source = repo / 'glm5-3-vllm/runtime-patches/p-first-seed-v2/mtp_ffn_graph.py'
spec = importlib.util.spec_from_file_location('mtp_ffn_test', source)
helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
os.environ['GLM53_P_FIRST_DRAFT_EXPORT'] = '0'
passed = []

def test(name, mutate=None, expected_graph=True, fail=False):
    recording = [None]
    calls = []
    context = S(capturing=False, cudagraph_runtime_mode='NONE', device_metadata_executor=None)
    extra = S(moe_comm_type=S(name='ALLGATHER'), is_draft_model=True, in_profile_run=False,
              num_tokens=6, num_actual_tokens=6, num_accept_tokens=1)

    class Tensor:
        def __init__(self, value, n=6):
            self.value = value; self.shape = (n, 6144); self.dtype = 'bf16'; self.device = S(type='npu'); self.ndim = 2
        def is_contiguous(self): return True
        def copy_(self, other): self.value = other.value; calls.append(('copy', id(self), other.value)); return self

    class Graph:
        def __init__(self): self.fn = None
        def replay(self): calls.append(('replay',)); self.fn()

    @contextmanager
    def graph(g):
        assert context.capturing
        recording[0] = g
        try: yield
        finally: recording[0] = None

    class MoE:
        is_sequence_parallel = False; enable_eplb = False
        def forward(self, x, already_sequence_parallel=False):
            calls.append(('native', x.value))
            if fail and recording[0] is not None: raise RuntimeError('capture refusal')
            y = Tensor(x.value * 3, x.shape[0])
            if recording[0] is not None:
                recording[0].fn = lambda: setattr(y, 'value', x.value * 3)
            return y

    module = MoE(); target = MoE(); original_moe_forward = MoE.forward
    class Model:
        def named_modules(self): return [('model.layers.78.mtp_block.mlp', module)]
    class Runner:
        def sample_tokens(self):
            values = []
            for i in range(6):
                values.append(module.forward(Tensor(i)).value)
                assert target.forward(Tensor(i)).value == i * 3
            return values

    runner = Runner()
    runner.input_batch = S(req_ids=['ordinary'])
    runner.requests = {'ordinary': S(sampling_params=S(max_tokens=64))}
    runner.parallel_config = S(tensor_parallel_size=16, data_parallel_size=1, pipeline_parallel_size=1,
                               decode_context_parallel_size=1, prefill_context_parallel_size=1)
    runner.drafter = S(method='mtp', use_cuda_graph=False, model=Model())
    runner.is_kv_consumer=True; runner.is_kv_producer=False; runner.use_async_scheduling=True
    runner.num_spec_tokens=5; runner.model_config=S(use_mla=True, is_hybrid=False)
    runner.dynamic_eplb=False; runner.tp_rank=0
    torch = S(bfloat16='bf16', empty_like=lambda x: Tensor(-1, x.shape[0]), npu=S(NPUGraph=Graph, graph=graph))
    sp = [False]
    ns = {'NPUModelRunner': Runner, 'CUDAGraphMode': S(NONE='NONE'), 'torch': torch, 'enable_sp': lambda: sp[0]}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory); (root/'enable').touch()
        helper.install(ns, root=root, deps={'MoE': MoE, 'ctx': lambda: context, 'extra': extra})
        assert MoE.forward is original_moe_forward, 'Global target method was changed before model compilation'
        if mutate: mutate(runner, context, extra, module, root, sp)
        if fail:
            try: runner.sample_tokens()
            except RuntimeError as exc: assert str(exc) == 'capture refusal'
            else: raise AssertionError('Expected capture failure')
            assert context.capturing is False
            before = len(calls); assert module.forward(Tensor(17)).value == 51
            assert len(calls) == before + 1, 'ContextVar leaked after error'
        else:
            assert runner.sample_tokens() == [i*3 for i in range(6)]
            assert any(x[0] == 'replay' for x in calls) is expected_graph
            assert context.capturing is False
            before = len(calls); assert module.forward(Tensor(17)).value == 51
            assert len(calls) == before + 1, 'ContextVar leaked after normal completion'
        passed.append(name)
        assert MoE.forward is original_moe_forward and target.forward.__func__ is original_moe_forward

test('owned buffer copied on each replay and target MoE remains native')
test('marker absent', lambda r,c,e,m,p,s:(p/'enable').unlink(), False)
test('single output protected', lambda r,c,e,m,p,s:setattr(r.requests['ordinary'].sampling_params,'max_tokens',1), False)
test('producer protected', lambda r,c,e,m,p,s:setattr(r,'is_kv_producer',True), False)
test('SP protected', lambda r,c,e,m,p,s:s.__setitem__(0,True), False)
test('DP protected', lambda r,c,e,m,p,s:setattr(r.parallel_config,'data_parallel_size',2), False)
test('whole draft graph protected', lambda r,c,e,m,p,s:setattr(r.drafter,'use_cuda_graph',True), False)
test('device metadata protected', lambda r,c,e,m,p,s:setattr(c,'device_metadata_executor',object()), False)
test('attention capture protected', lambda r,c,e,m,p,s:setattr(c,'cudagraph_runtime_mode','FULL'), False)
test('other communication protected', lambda r,c,e,m,p,s:setattr(e.moe_comm_type,'name','MC2'), False)
test('profile protected', lambda r,c,e,m,p,s:setattr(e,'in_profile_run',True), False)
test('non scalar context protected', lambda r,c,e,m,p,s:setattr(e,'num_accept_tokens',object()), False)
test('EPLB protected', lambda r,c,e,m,p,s:setattr(m,'enable_eplb',True), False)
test('capture exception restores context and invokes native outside sample', fail=True)
result = {'passed': True, 'checks': len(passed), 'cases': passed,
          'scope': 'CPU host-control and buffer-copy/guard/context checks only. No NPU graph or model correctness evidence.'}
(Path(__file__).parent/'mtp-ffn-graph-cpu-check-v1.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
