"""Retain native metadata while admitting the V4.1 NPU 128/64 page contract.

The context flag is set only by DeepseekV41NPUBackend's indexer factory.
CUDA/other model callers keep the original page-256 guard.
"""
import ast
from contextvars import ContextVar
import hashlib
import inspect
import json
from pathlib import Path
import textwrap

active_v41_npu_indexer = ContextVar('active_v41_npu_indexer_g314', default=False)


def permitted(metadata, torch_path):
    return metadata.page_size == 256 or (
        active_v41_npu_indexer.get()
        and metadata.page_size == 128
        and metadata.compress_ratio in (1, 2)
        and metadata.compressed_page_size == 64
        and metadata.page_table.device.type == 'npu'
        and metadata.compressed_seq_lens.device.type == 'npu'
        and torch_path
        and not metadata.force_deep_gemm_metadata
        and not metadata.use_topk_v2
    )


def install(root):
    import sglang.srt.layers.attention.dsv4.metadata as native
    if getattr(native, '_v41_npu_page_g314', False):
        return
    expected = json.loads((root/'native_backend_sha256.json').read_text())['metadata']
    assert hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() == expected
    method = native.PagedIndexerMetadata.__post_init__
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert) and ast.unparse(node.test) == 'self.page_size == 256':
            node.test = ast.parse('_v41_npu_page_guard_g314(self, envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get())', mode='eval').body
            count += 1
    assert count == 1
    native.__dict__['_v41_npu_page_guard_g314'] = permitted
    namespace = dict(native.__dict__)
    exec(compile(ast.fix_missing_locations(tree), str(root/'paged_indexer_scope_g314.py'), 'exec'), namespace)
    assert not method.__code__.co_freevars
    method.__code__ = namespace['__post_init__'].__code__
    native._v41_npu_page_g314 = True
