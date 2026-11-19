"""NPU hardware boundary for the native V4.1 low-ratio backend.

Preserves native source ownership, candidate publication, metadata and DSPARK
control. This is an integration candidate until full model/PD gates pass.
"""
from pathlib import Path
import ast, hashlib, inspect, json, textwrap
import torch
from fp4_compat import fake_quant_compressed_kv
from lowratio_sparse_attention import prepare_pool, _sparse_shared_kv, BANK_ROWS


def install_native_metadata_scope(root):
    import sglang.srt.layers.attention.deepseek_v4_backend as native
    if getattr(native, '_ds41_npu_metadata_g268', False):
        return
    pins = json.loads((root / 'native_backend_sha256.json').read_text())
    assert hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() == pins['backend']
    original = native.DeepseekV4AttnBackend.__init__
    function = ast.parse(textwrap.dedent(inspect.getsource(original))).body[0]
    changed = 0
    for node in ast.walk(function):
        if isinstance(node, ast.Assert) and ast.unparse(node.test) == 'self.page_size == 256':
            node.test = ast.parse('_ds41_npu_page_guard_g268(self)', mode='eval').body
            changed += 1
    assert changed == 1
    # Keep the native __class__ closure used by zero-argument super().
    unit = ast.fix_missing_locations(ast.Module(body=[
        ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
        ast.ClassDef(name='_ConstructorCode', bases=[], keywords=[],
                     body=[function], decorator_list=[])], type_ignores=[]))
    namespace = dict(native.__dict__)
    exec(compile(unit, str(root/'native_backend_constructor_g268.py'), 'exec'), namespace)
    code = namespace['_ConstructorCode'].__init__.__code__
    assert code.co_freevars == original.__code__.co_freevars == ('__class__',)
    def page_guard(self):
        return self.page_size == 256 or (
            self.page_size == 128 and self.device.type == 'npu'
            and self.model_runner.model_config.hf_config.model_type == 'deepseek_v41')
    native.__dict__['_ds41_npu_page_guard_g268'] = page_guard
    original.__code__ = code
    prior = native._create_flashmla_metadata
    def schedule():
        from sglang.srt.utils import is_npu
        return None if is_npu() else prior()
    native._create_flashmla_metadata = schedule
    import sglang.srt.layers.attention.dsv4.dsv41_sparse as sparse
    prior_quant = sparse.fake_quant_compressed_kv
    # A2 stock-native032 stores post-RoPE compressed values as BF16.
    # Change only attention compressed-KV rounding; FP4 index keys/queries,
    # W4A8 checkpoint, pooling, RoPE and BF16 physical cache stay unchanged.
    seen = False
    def compressed_kv_bf16(x):
        nonlocal seen
        if x.device.type != 'npu':
            return prior_quant(x)
        assert x.dtype == torch.bfloat16 and x.shape[-1] == 512
        if not seen:
            import os
            receipt = {'pid': os.getpid(), 'policy': 'A2_BF16_compressed_KV_no_fake_FP4',
                       'shape': list(x.shape), 'dtype': str(x.dtype), 'device': str(x.device),
                       'identity_return_same_storage': True, 'FP4_index_quantization_unchanged': True,
                       'W4A8_checkpoint_unchanged': True,
                       'source_SHA256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       'scope': 'First actual compressed-KV dispatch; quality/performance not yet accepted'}
            (root / ('compressed_KV_BF16_first_call.pid' + str(os.getpid()) + '.json')).write_text(
                json.dumps(receipt, indent=2) + '\n')
            seen = True
        return x
    sparse.fake_quant_compressed_kv = compressed_kv_bf16
    native._ds41_npu_metadata_g268 = True


def install_bf16_request_window():
    import sglang.srt.mem_cache.dsv41_request_window as native
    if getattr(native.RequestWindow, '_npu_bf16_g268', False):
        return
    class NPUBF16RequestWindow(native.RequestWindow):
        _npu_bf16_g268 = True
        def buffer(self, layer):
            from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode
            in_capture = get_is_capture_mode() or getattr(
                torch.npu, 'is_current_stream_capturing', lambda: False)()
            key = (layer, in_capture)
            if self.prepared != key:
                layout = self.layout
                assert layout is not None, 'Request-window metadata not active'
                if not in_capture and native.resolve_level() >= native.InvariantCheckLevel.WARN:
                    native.expect(native._SWA_HISTORY_PRESENT,
                        (self.tags[layer,layout.history_src] == layout.history_pos)
                        | ~layout.history_valid,
                        msg='replay or window ownership is invalid')
                source = self.state.kv_buffer[layer]
                target = self.workspace.kv_buffer[0]
                assert source.dtype == target.dtype == torch.bfloat16
                assert source.shape[-1] == target.shape[-1] == 512
                target.view(-1,512)[layout.history_loc.long()] = source.view(-1,512)[layout.history_src.long()]
                self.prepared = key
            return self.workspace.kv_buffer[0]
        def commit(self, layer):
            layout = self.layout
            source = self.buffer(layer).view(-1,512)
            target = self.state.kv_buffer[layer].view(-1,512)
            loc = self.row_loc(layout.req, layout.pos)
            # Valid rows retain the native unique last-capacity write set.
            # Padding writes use an unobservable dedicated sink, never zero_row.
            loc = torch.where(layout.commit_mask, loc,
                              torch.full_like(loc, self.sink_row)).long()
            target[loc] = source[layout.write_loc.long()]
            self.tags[layer, loc] = layout.pos
    native.RequestWindow = NPUBF16RequestWindow


from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend


class DeepseekV41NPUBackend(DeepseekV4AttnBackend):
    def __init__(self, model_runner, *args, **kwargs):
        assert str(model_runner.device).startswith('npu')
        super().__init__(model_runner, *args, **kwargs)
        from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
        assert self.dsa_topk_backend is DSATopKBackend.TORCH
        assert self.present_ratios in ((0,), (1,2), (0,1,2)) or set(self.present_ratios) <= {0,1,2}
        self._bf16_views = {}
        pool = self.token_to_kv_pool
        stores = list(pool.kv_pools.values())
        if pool.request_window is not None:
            stores.extend([pool.request_window.state, pool.request_window.workspace])
        elif pool.swa_kv_pool is not None:
            stores.append(pool.swa_kv_pool)
        for store in stores:
            if store is None:
                continue
            for buffer in store.kv_buffer:
                view = buffer.view(-1,512)
                prepare_pool(view)
                self._bf16_views[buffer.data_ptr()] = view
        self._empty_kv = torch.zeros((1,512),dtype=torch.bfloat16,device=self.device)
        prepare_pool(self._empty_kv)
        self._no_compressed = torch.full((1,512),-1,dtype=torch.int32,device=self.device)
        self._no_sink = torch.full((64,),-torch.inf,dtype=torch.float32,device=self.device)

    def init_forward_metadata_indexer(self, *args, **kwargs):
        from paged_indexer_scope_g314 import active_v41_npu_indexer
        token = active_v41_npu_indexer.set(True)
        try:
            return super().init_forward_metadata_indexer(*args, **kwargs)
        finally:
            active_v41_npu_indexer.reset(token)

    def _view(self, buffer):
        view = self._bf16_views[buffer.data_ptr()]
        assert view.numel() == buffer.numel()
        return view

    def store_cache(self, layer_id, swa_k, forward_batch):
        loc = self.get_swa_out_cache_loc(forward_batch).long()
        buffer = self.token_to_kv_pool.get_swa_raw_buffer(layer_id)
        self._view(buffer)[loc] = swa_k.reshape(-1,512).to(torch.bfloat16)

    def _low_ratio_prefill_reads_page_indices(self, forward_batch):
        # The native torch prefill indexer also writes its raw selection; this
        # backend additionally consumes physical KV slots from the same picks.
        return True

    def _forward_attention(self,q,k,v,layer,forward_batch,compress_ratio=0,
                           save_kv_cache=True,attn_sink=None,**kwargs):
        if forward_batch.forward_mode.is_idle():
            return torch.zeros_like(q)
        assert compress_ratio in (0,1,2)
        if save_kv_cache:
            assert k is not None
            self.store_cache(layer.layer_id,k,forward_batch)
        core = self.forward_metadata.core_metadata
        pool = self.token_to_kv_pool
        original = self._view(pool.get_swa_raw_buffer(layer.layer_id))
        oi = core.swa_page_indices.reshape(q.shape[0],-1).to(torch.int32).contiguous()
        if compress_ratio:
            item=pool.layer_mapping[layer.layer_id]
            compressed=self._view(item.compress_kv_pool.kv_buffer[item.compress_layer_id])
            ci=core.sparse_page_indices(compress_ratio).reshape(q.shape[0],-1).to(torch.int32).contiguous()
        else:
            compressed=self._empty_kv
            ci=self._no_compressed.expand(q.shape[0],512).contiguous()
        assert oi.shape[1] >= 128 and ci.shape[1] == 512
        q=q.contiguous()
        sink=(self._no_sink[:q.shape[1]] if attn_sink is None else attn_sink).float().contiguous()
        assert sink.numel()==q.shape[1]
        result=torch.empty_like(q)
        ob=original._ds41_attention_banks[1]
        cb=compressed._ds41_attention_banks[1]
        if q.shape[0]:
            _sparse_shared_kv[(q.shape[0],q.shape[1])](
                q,ob,cb,oi,ci,sink,result,q.stride(0),original.stride(0),compressed.stride(0),
                oi.stride(0),ci.stride(0),result.stride(0),ob.numel(),cb.numel(),BANK_ROWS,
                512,oi.shape[1],512,float(layer.scaling),512,8)
        return result
