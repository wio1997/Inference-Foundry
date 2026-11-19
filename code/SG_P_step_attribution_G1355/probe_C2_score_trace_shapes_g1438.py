"""Observe fixed-image CPU operand shapes for the unannotated Prefill scorer."""

import ast
from contextlib import closing
from hashlib import sha256
import json
from pathlib import Path
import platform
import sqlite3

import torch


SOURCE_SHA = '829c867262dfeaf1284f3fd47ff9268f4e3bc01ffce62255e92d9293d27f4197'
SOURCE = Path('/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py')


def load_scores():
    raw = SOURCE.read_bytes()
    assert sha256(raw).hexdigest() == SOURCE_SHA
    ast_tree = ast.parse(raw)
    indexer = next(node for node in ast_tree.body if isinstance(node, ast.ClassDef)
                   and node.name == 'DeepseekV41Indexer')
    method = next(node for node in indexer.body if isinstance(node, ast.FunctionDef) and node.name == 'scores')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<pinned-C2-scores>', 'exec'), namespace)
    return namespace['scores']


def cpu_shapes(scores, query_rows, keys):
    q = torch.zeros((query_rows, 32, 128), dtype=torch.bfloat16)
    k = torch.zeros((keys, 128), dtype=torch.bfloat16)
    weights = torch.ones((query_rows, 32), dtype=torch.bfloat16)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as events:
        output = scores(None, q, k, weights)
        torch.einsum('bhd,bd->bh', q, q[:, 0, :])
    bmm = [event for event in events.events() if event.name == 'aten::bmm']
    assert len(bmm) == 2 and all(event.cpu_parent is not None for event in bmm)
    known, control = bmm
    stock = [[1, query_rows*32, 128], [1, 128, keys]]
    assert known.input_shapes == stock and known.cpu_parent.name == 'aten::einsum'
    assert control.input_shapes != stock and control.cpu_parent.name == 'aten::einsum'
    assert tuple(output.shape) == (query_rows, keys) and output.dtype is torch.float32
    return {'query_rows': query_rows, 'compressed_K_rows': keys, 'score_bmm_input_shapes': known.input_shapes,
            'unrelated_einsum_bmm_input_shapes': control.input_shapes, 'both_CPU_parents': 'aten::einsum'}


def ascend_CPU_export(scores):
    import torch_npu

    profiler = torch_npu.profiler
    output = Path('/scratch') / 'ascend_cpu_shape_control'
    q = torch.zeros((7, 32, 128), dtype=torch.bfloat16)
    k = torch.zeros((11, 128), dtype=torch.bfloat16)
    weights = torch.ones((7, 32), dtype=torch.bfloat16)
    try:
        config = profiler._ExperimentalConfig(export_type=[profiler.ExportType.Db])
        with profiler.profile(activities=[profiler.ProfilerActivity.CPU], record_shapes=True,
                               experimental_config=config,
                               on_trace_ready=profiler.tensorboard_trace_handler(str(output))) as trace:
            scores(None, q, k, weights)
            trace.step()
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return {'parsed_bmm_shapes_available': False, 'CPU_export_error_type': type(exc).__name__}
    db_files = sorted(output.rglob('*.db'))
    shapes = []
    for path in db_files:
        with closing(sqlite3.connect(path.as_uri()+'?mode=ro', uri=True)) as db:
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not {'PYTORCH_API', 'STRING_IDS'} <= tables:
                continue
            shapes.extend(db.execute("""
                SELECT shape.value FROM PYTORCH_API api JOIN STRING_IDS s ON s.id=api.name
                LEFT JOIN STRING_IDS shape ON shape.id=api.inputShapes WHERE s.value='aten::bmm'
            """).fetchall())
    return {'parsed_bmm_shapes_available': any(row[0] for row in shapes),
            'ascend_CPU_exported_DB_files': len(db_files), 'bmm_exported_input_shapes': shapes}


def main():
    scores = load_scores()
    torch.set_num_threads(1)
    result = [cpu_shapes(scores, query, keys) for query, keys in ((7, 11), (19, 29))]
    export = ascend_CPU_export(scores)
    assert not torch.npu.is_initialized()
    print('G1438_FIXED_SCORE_TRACE_SHAPES_JSON '+json.dumps({
        'architecture': platform.machine(), 'torch': torch.__version__, 'source_SHA256': SOURCE_SHA,
        'cpu_operand_checks': result, 'ascend_CPU_export': export, 'NPU_initialized': False,
    }, sort_keys=True))


if __name__ == '__main__':
    main()
