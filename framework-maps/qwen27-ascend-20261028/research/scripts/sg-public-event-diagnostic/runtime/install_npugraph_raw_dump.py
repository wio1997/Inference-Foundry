"""Opt-in post-capture raw NPUGraph dump observer.

Set Q27_NPUGRAPH_DUMP_DIR before process startup, then call install() after
torch_npu.npu.graphs is imported. This wraps the existing Python NPUGraph
capture_end method; it never creates, mutates, or replays a graph.
"""
from __future__ import annotations

import functools
import json
import os
import threading
import time
from pathlib import Path

_ENV = "Q27_NPUGRAPH_DUMP_DIR"
_LOCK = threading.Lock()
_COUNTER = 0
_INSTALLED = set()
_CAPTURE_CONTEXT = {}

def _next_seq():
    global _COUNTER
    with _LOCK:
        _COUNTER += 1
        return _COUNTER

def _atomic_json(path: Path, payload):
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}.{threading.get_ident()}")
    tmp.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)

def install(npugraph_cls=None, dump_dir=None):
    """Install once; accepts a class for tests or imports torch_npu lazily.

    Returns a small status mapping. With no env override and no dump_dir, this
    is a no-op. Failures during dump are captured in receipts and do not change
    capture_end's successful return value. The original capture_end exception
    propagates unchanged and causes no dump attempt.
    """
    target = dump_dir if dump_dir is not None else os.environ.get(_ENV)
    if not target:
        return {"installed": False, "reason": f"set {_ENV} or pass dump_dir"}
    if npugraph_cls is None:
        from torch_npu.npu.graphs import NPUGraph as npugraph_cls
    key = (npugraph_cls, os.path.abspath(os.fspath(target)))
    if key in _INSTALLED:
        return {"installed": True, "already_installed": True, "dump_dir": key[1]}
    original = npugraph_cls.capture_end

    @functools.wraps(original)
    def capture_end_with_raw_dump(self, *args, **kwargs):
        seq = _next_seq()
        pid = os.getpid()
        graph_python_id = id(self)
        root = Path(key[1]) / f"pid-{pid}"
        stem = f"graph-{graph_python_id}-capture-{seq}"
        raw = root / "raw" / f"{stem}.json"
        derived = root / "derived" / f"{stem}.aligned.json"
        receipt = root / "receipts" / f"{stem}.receipt.json"
        rec = {
            "schema_version": 1, "pid": pid, "thread_id": threading.get_ident(),
            "graph_python_id": graph_python_id, "capture_sequence": seq,
            "raw_path": str(raw), "derived_path": str(derived),
            "derived_semantics": "visualization-only rebuilt/aligned timestamps; never measured timing",
            "started_unix_ns": time.time_ns(),
        }
        try:
            result = original(self, *args, **kwargs)
        except BaseException as exc:
            rec.update({"capture_end_status": "failure", "dump_status": "skipped_capture_end_failed",
                        "error_type": type(exc).__name__, "error": str(exc)[:1000]})
            rec["finished_unix_ns"] = time.time_ns()
            _CAPTURE_CONTEXT[graph_python_id]=dict(rec)
            try:
                receipt.parent.mkdir(parents=True, exist_ok=True)
                _atomic_json(receipt, rec)
            except BaseException:
                pass
            raise
        rec.update({"capture_end_status": "success", "capture_end_return_type": type(result).__name__,
                    "capture_end_return_repr": repr(result)[:256]})
        try:
            raw.parent.mkdir(parents=True, exist_ok=True)
            derived.parent.mkdir(parents=True, exist_ok=True)
            receipt.parent.mkdir(parents=True, exist_ok=True)
            dump_result = self.debug_dump(str(raw))
            runtime_aligned = raw.with_suffix(".aligned.json")
            if runtime_aligned.is_file():
                os.replace(runtime_aligned, derived)
            from torch import distributed as dist
            actual_rank=int(dist.get_rank()) if dist.is_initialized() else None
            graph_rows=json.loads(raw.read_text()) if raw.is_file() else []
            graph_nodes=graph_rows if isinstance(graph_rows,list) else graph_rows.get('traceEvents',[])
            models=set()
            for node in graph_nodes:
                if not isinstance(node,dict):continue
                for box in [node,node.get('args',{})]:
                    if not isinstance(box,dict):continue
                    for name,value in box.items():
                        if ''.join(c.lower()for c in str(name)if c.isalnum())=='modelid' and value is not None:models.add(str(value))
            rec['graph_model_ids']=sorted(models)
            rec.update(rank_id=actual_rank, rank_source='torch.distributed.get_rank' if actual_rank is not None else 'distributed_not_initialized',capture_id=os.environ.get('Q27_LINEAGE_GENERATION','initial'),generation=os.environ.get('Q27_LINEAGE_GENERATION','initial'),graph_node_pids=sorted({str(node['pid']) for node in graph_nodes if isinstance(node,dict) and node.get('pid') is not None}))
            rec.update({
                "dump_status": "success", "debug_dump_return_type": type(dump_result).__name__,
                "raw_exists": raw.is_file(), "raw_size_bytes": raw.stat().st_size if raw.is_file() else None,
                "derived_exists": derived.is_file(),
                "debug_dump_return_repr": repr(dump_result)[:256],
            })
        except Exception as exc:
            # Observer must not turn an already successful capture_end into a failure.
            rec.update({"dump_status": "failure", "error_type": type(exc).__name__, "error": str(exc)[:1000]})
        finally:
            rec["finished_unix_ns"] = time.time_ns()
            _CAPTURE_CONTEXT[graph_python_id]=dict(rec)
            try:
                receipt.parent.mkdir(parents=True, exist_ok=True)
                _atomic_json(receipt, rec)
            except BaseException:
                # Keep observation strictly non-fatal, including receipt I/O.
                pass
        return result

    npugraph_cls.capture_end = capture_end_with_raw_dump
    def emit_replay(row):
        path=Path(key[1])/('pid-'+str(os.getpid()))/'replay-events.jsonl'
        path.parent.mkdir(parents=True,exist_ok=True)
        data=(json.dumps(row,separators=(',',':'))+'\n').encode()
        fd=os.open(path,os.O_CREAT|os.O_WRONLY|os.O_APPEND,0o600)
        try:os.write(fd,data)
        finally:os.close(fd)
    from q27_graph_replay_identity import install as observe_replay
    observe_replay(npugraph_cls,lambda graph:_CAPTURE_CONTEXT.get(id(graph)),emit_replay)
    _INSTALLED.add(key)
    return {"installed": True, "dump_dir": key[1], "target": f"{npugraph_cls.__module__}.{npugraph_cls.__qualname__}.capture_end"}
