"""Opt-in CPU trace span for the fixed V4.1 Prefill index score path."""
from functools import wraps
from hashlib import sha256
from pathlib import Path

SPARSE_SOURCE_SHA256 = '829c867262dfeaf1284f3fd47ff9268f4e3bc01ffce62255e92d9293d27f4197'
SPAN_SOURCE_SHA256 = 'aef800c2f362094103ff57f04231267c27d2f3d6ae6be00af348d3a1d6651340'


def annotate(original, enabled, record):
    @wraps(original)
    def scoped(self, q, k, weights):
        if not enabled() or q.device.type != 'npu':
            return original(self,q,k,weights)
        rows, heads, depth = q.shape
        name = f'ds41.profile.index_score q={rows} h={heads} d={depth} k={k.shape[0]}'
        with record(name):
            return original(self,q,k,weights)
    scoped._ds41_prefill_index_score_trace_marker = True
    return scoped


def install(*, role):
    """Call from a separately gated Prefill profiling process."""
    assert role == 'prefill'
    import torch
    from sglang.srt.layers.attention.dsv4 import dsv41_sparse
    from sglang.srt.model_executor import step_span_utils

    assert sha256(Path(dsv41_sparse.__file__).read_bytes()).hexdigest() == SPARSE_SOURCE_SHA256
    assert sha256(Path(step_span_utils.__file__).read_bytes()).hexdigest() == SPAN_SOURCE_SHA256
    indexer = dsv41_sparse.DeepseekV41Indexer
    original = indexer.scores
    if not getattr(original,'_ds41_prefill_index_score_trace_marker',False):
        indexer.scores = annotate(original,step_span_utils.detailed_annotations_enabled,torch.profiler.record_function)
