#!/usr/bin/env python3
"""Reversible Run605 dispatch/frontier observer; diagnostic, not a timing Bound."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path

import loop081_product_patch_run602 as product

ASC = Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend')
SOURCES = dict(product.SOURCES)
SOURCES.update({
    'dflash': ASC / 'spec_decode/dflash_proposer.py',
    'acl_graph': ASC / 'compilation/acl_graph.py',
})
ORIGINAL = dict(product.ORIGINAL)
ORIGINAL.update({
    'dflash': '6be46559f9f869861c676efb4531b0c01eef8a0445521ecadf5c1fec1c4412f8',
    'acl_graph': '6396ca409d633ee61c4433a762982cf901b893f252c864c0ff05ac17bdbdb3f6',
})
MARK = '# EXTREME_LOOP081_RUN605'


def one(src, old, new):
    if src.count(old) != 1:
        raise ValueError(f'anchor count {src.count(old)}: {old[:100]!r}')
    return src.replace(old, new, 1)


META_HELPER = '''
# EXTREME_LOOP081_RUN605_META_HELPER
def _p605_metadata_summary(obj):
    rows, seen = [], set()
    def visit(value, depth):
        relevant = isinstance(value, (tuple, list, dict)) or 'DSA' in type(value).__name__ or 'SFA' in type(value).__name__
        if depth > 4 and relevant:
            raise RuntimeError('Run605 metadata depth overflow')
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, (tuple, list)):
            for x in value:
                visit(x, depth + 1)
        elif isinstance(value, dict):
            for x in value.values():
                visit(x, depth + 1)
        elif 'DSA' in type(value).__name__ or 'SFA' in type(value).__name__:
            if len(rows) >= 16:
                raise RuntimeError('Run605 metadata group overflow')
            row = {'type': type(value).__name__}
            for key in ('num_decodes', 'num_prefills', 'num_decode_tokens',
                        'num_prefill_tokens', 'num_actual_tokens', 'num_input_tokens',
                        'num_reqs', 'max_query_len'):
                scalar = getattr(value, key, None)
                if isinstance(scalar, (int, bool, np.integer)):
                    row[key] = int(scalar)
            state = getattr(value, 'attn_state', None)
            row['attn_state'] = getattr(state, 'name', None)
            rows.append(row)
    visit(obj, 0)
    return rows
'''


def patch_runner(src):
    src = product.patch('runner', src)
    src = src.replace('EXTREME_RUN602_DIR', 'EXTREME_RUN605_DIR')
    old = '''        _p602_start.record()
        try:
            return _p602_original_forward(self, num_tokens_padded, *args, **kwargs)
        finally:
            _p602_end.record()
            self._p602_calls = getattr(self, "_p602_calls", [])
'''
    new = '''        _p602_start.record()
        from vllm_ascend.compilation.acl_graph import _p605_forward_ctx
        self._p605_graph_records = getattr(self, "_p605_graph_records", [])
        _p605_ordinal = len(getattr(self, "_p602_marks", ()))
        _p605_ctx_token = _p605_forward_ctx.set(
            {"records": self._p605_graph_records, "ordinal": _p605_ordinal})
        try:
            return _p602_original_forward(self, num_tokens_padded, *args, **kwargs)
        finally:
            _p602_end.record()
            _p605_forward_ctx.reset(_p605_ctx_token)
            _p605_ctx = get_forward_context()
            _p605_desc = _p605_ctx.batch_descriptor
            self._p605_forward_records = getattr(self, "_p605_forward_records", [])
            self._p605_forward_records.append({
                "ordinal": _p605_ordinal,
                "mode": _p605_ctx.cudagraph_runtime_mode.name,
                "batch_num_tokens": getattr(_p605_desc, "num_tokens", None),
                "batch_num_reqs": getattr(_p605_desc, "num_reqs", None),
                "use_compress": bool(self.use_compress),
                "use_sparse": bool(self.use_sparse),
                "enable_enpu": bool(self.enable_enpu),
                "attention": _p605_metadata_summary(_p605_ctx.attn_metadata),
            })
            self._p602_calls = getattr(self, "_p602_calls", [])
'''
    src = one(src, old, new)
    src += META_HELPER
    src = one(src,
        '            # EXTREME_LOOP081_RUN602_BUILT\n',
        '            # EXTREME_LOOP081_RUN605_RUNTIME_ARM\n'
        '            _extreme_runtime._p605_armed = bool(_p602_enabled)\n'
        '            # EXTREME_LOOP081_RUN602_BUILT\n')
    src = one(src,
        '            self._draft_token_ids = self.propose_draft_token_ids(\n',
        '            # EXTREME_LOOP081_RUN605_DRAFT_START\n'
        '            if _p602_enabled:\n'
        '                self._p605_draft_records = getattr(self, "_p605_draft_records", [])\n'
        '            self._draft_token_ids = self.propose_draft_token_ids(\n')
    src = one(src,
        '                    _p602_payload = {"rank": _p602_rank, "cohort": _cohort_index,\n',
        '''                    # EXTREME_LOOP081_RUN605_FRONTIER_FLUSH
                    _p605_target_event = getattr(_extreme_runtime, "_p605_first_target_event", None)
                    _p605_draft_event = getattr(_extreme_runtime, "_p605_first_draft_event", None)
                    _p605_context = []
                    for _p605_row, _p605_event in getattr(self, "_p605_context_events", []):
                        _p605_row = dict(_p605_row)
                        _p605_row["to_first_target_ms"] = (
                            float(_p605_event.elapsed_time(_p605_target_event))
                            if _p605_target_event is not None else None)
                        _p605_row["to_first_draft_ms"] = (
                            float(_p605_event.elapsed_time(_p605_draft_event))
                            if _p605_draft_event is not None else None)
                        _p605_context.append(_p605_row)
                    _p605_frontier = {
                        "forward": getattr(self, "_p605_forward_records", []),
                        "graphs": getattr(self, "_p605_graph_records", []),
                        "draft_inputs": getattr(self, "_p605_draft_records", []),
                        "context_stores": _p605_context,
                        "first_target_entry_event_present": _p605_target_event is not None,
                        "first_draft_entry_event_present": _p605_draft_event is not None,
                        "first_target_stream": getattr(_extreme_runtime, "_p605_first_target_stream", None),
                        "first_draft_stream": getattr(_extreme_runtime, "_p605_first_draft_stream", None),
                        "scope": "current-stream events and Host branch marks; no side-stream completion proof",
                    }
                    self._p605_forward_records = []
                    self._p605_graph_records = []
                    self._p605_draft_records = []
                    self._p605_context_events = []
                    _p602_payload = {"rank": _p602_rank, "cohort": _cohort_index,
                                     "frontier": _p605_frontier,
''')
    src = one(src,
        '                return None\n            _initial_num_computed = (\n',
        '''                # EXTREME_LOOP081_RUN605_OUTPUT_RETURN
                if _p602_enabled:
                    _p605_out = os.path.join(
                        os.path.dirname(_p602_dir), "returns",
                        f"return_rank{_p602_rank}_cohort{_cohort_index}.json")
                    with open(_p605_out, "w") as _p605_f:
                        json.dump({"rank": _p602_rank, "cohort": _cohort_index,
                                   "run_ts": os.getenv("RUN_TS"),
                                   "req_ids": list(_extreme_req_ids),
                                   "model_runner_output_built_ns": time.monotonic_ns()},
                                  _p605_f, separators=(",", ":"))
                return None
            _initial_num_computed = (
''')
    return src


def patch_dspark(src):
    src = product.patch('draft', src)
    src = one(src, 'from typing import Any\n', 'from typing import Any\nimport os\nimport time\n')
    anchor = '        self._dflash_hidden_states[: self._dflash_num_context] = target_hidden_states[: self._dflash_num_context]\n'
    addition = '''        # EXTREME_LOOP081_RUN605_DRAFT_INPUTS
        _p605_dir = os.getenv("EXTREME_RUN605_DIR")
        if (_p605_dir and os.path.exists(os.path.join(_p605_dir, "arm"))
                and self.runner is not None
                and hasattr(self.runner, "input_batch")
                and hasattr(self.runner, "_p602_marks")):
            self.runner._p605_draft_records = getattr(self.runner, "_p605_draft_records", [])
            self.runner._p605_draft_records.append({
                "ordinal": len(getattr(self.runner, "_p602_marks", ())),
                "t_ns": time.monotonic_ns(),
                "req_ids": list(self.runner.input_batch.req_ids),
                "batch_size": int(batch_size),
                "context_rows": self._dflash_num_context,
                "query_rows": num_query_total,
                "sample_rows": num_sample_total,
                "num_query_per_req": int(self.num_query_per_req),
                "sample_from_anchor": bool(self.sample_from_anchor),
                "has_num_rejected": bool(has_num_rejected),
                "group_count": len(self.draft_attn_groups),
                "use_cuda_graph": bool(self.use_cuda_graph),
            })
'''
    return one(src, anchor, anchor + addition)


def patch_dflash(src):
    src = one(src, 'from typing import Any\n', 'from typing import Any\nimport os\nimport time\n')
    anchor = '''        self.model.precompute_and_store_context_kv(
            self._dflash_hidden_states[:num_context],
            self._context_positions_buffer[:num_context],
            _context_slots,
        )
'''
    addition = '''        # EXTREME_LOOP081_RUN605_CONTEXT_STORE
        _p605_dir = os.getenv("EXTREME_RUN605_DIR")
        if (_p605_dir and os.path.exists(os.path.join(_p605_dir, "arm"))
                and self.runner is not None
                and hasattr(self.runner, "input_batch")
                and hasattr(self.runner, "_p602_marks")):
            _p605_stream = torch.npu.current_stream()
            _p605_event = torch.npu.Event(enable_timing=True)
            _p605_event.record()
            self.runner._p605_context_events = getattr(self.runner, "_p605_context_events", [])
            self.runner._p605_context_events.append(({
                "ordinal": len(getattr(self.runner, "_p602_marks", ())),
                "req_ids": list(self.runner.input_batch.req_ids),
                "context_rows": int(num_context),
                "slot_list_present": _context_slots is not None,
                "stream": getattr(_p605_stream, "npu_stream", None),
                "host_enqueue_end_ns": time.monotonic_ns(),
            }, _p605_event))
'''
    return one(src, anchor, anchor + addition)


def patch_runtime(src):
    src = product.patch('runtime', src)
    src = one(src,
        '                    target_output = self.target.execute(self.state)\n',
        '''                    # EXTREME_LOOP081_RUN605_FIRST_TARGET
                    if self.state.cycle_index == 0 and getattr(self, "_p605_armed", False):
                        self._p605_first_target_event = torch.npu.Event(enable_timing=True)
                        self._p605_first_target_event.record()
                        self._p605_first_target_stream = getattr(torch.npu.current_stream(), "npu_stream", None)
                    target_output = self.target.execute(self.state)
''')
    src = one(src,
        '                next_draft = self.proposer.execute(\n',
        '''                # EXTREME_LOOP081_RUN605_FIRST_DRAFT
                if self.state.cycle_index == 0 and getattr(self, "_p605_armed", False):
                    self._p605_first_draft_event = torch.npu.Event(enable_timing=True)
                    self._p605_first_draft_event.record()
                    self._p605_first_draft_stream = getattr(torch.npu.current_stream(), "npu_stream", None)
                next_draft = self.proposer.execute(
''')
    return src


def patch_acl_graph(src):
    src = one(src, 'import dataclasses\n',
              'import dataclasses\nimport contextvars\nimport time\n')
    src = one(src, '_acl_graph_wrappers: weakref.WeakSet[Any] = weakref.WeakSet()\n',
              '_acl_graph_wrappers: weakref.WeakSet[Any] = weakref.WeakSet()\n'
              '_p605_forward_ctx = contextvars.ContextVar("extreme_run605_forward", default=None)\n')
    src = one(src,
        '        forward_context = get_forward_context()\n        batch_descriptor = forward_context.batch_descriptor\n',
        '''        forward_context = get_forward_context()
        # EXTREME_LOOP081_RUN605_GRAPH_ENTRY
        _p605_parent = _p605_forward_ctx.get()
        _p605_row = None
        if (_p605_parent is not None and
                self.runtime_mode in (CUDAGraphMode.FULL, CUDAGraphMode.FULL_DECODE_ONLY)):
            if len(_p605_parent["records"]) >= 512:
                raise RuntimeError("Run605 graph record pool overflow")
            _p605_row = {"ordinal": _p605_parent["ordinal"],
                         "owner": type(self.runnable).__qualname__,
                         "wrapper_mode": self.runtime_mode.name,
                         "incoming_mode": forward_context.cudagraph_runtime_mode.name,
                         "enable_enpu": bool(self.enable_enpu),
                         "use_eagle": bool(self.use_eagle),
                         "is_draft_model": bool(_EXTRA_CTX.is_draft_model),
                         "host_entry_ns": time.monotonic_ns()}
            _p605_parent["records"].append(_p605_row)
        batch_descriptor = forward_context.batch_descriptor
''')
    src = one(src,
        '            return self.runnable(*args, **kwargs)\n',
        '''            if _p605_row is not None:
                _p605_row["branch"] = "fallthrough"
                _p605_row["fallthrough_enter_ns"] = time.monotonic_ns()
            _p605_output = self.runnable(*args, **kwargs)
            if _p605_row is not None:
                _p605_row["host_return_ns"] = time.monotonic_ns()
            return _p605_output
''')
    src = one(src,
        '            return output\n\n        if self.is_debugging_mode:',
        '''            if _p605_row is not None:
                _p605_row["branch"] = "capture"
                _p605_row["entry_id"] = id(entry)
                _p605_row["host_return_ns"] = time.monotonic_ns()
            return output

        if self.is_debugging_mode:''')
    src = one(src,
        '''        if not self.enable_enpu and need_sync:
            torch.npu.current_stream().synchronize()
        entry.aclgraph.replay()
        return entry.output
''',
        '''        if _p605_row is not None:
            _p605_row["branch"] = "replay"
            _p605_row["entry_id"] = id(entry)
            _p605_row["runtime_mode"] = aclgraph_runtime_mode.name
            _p605_row["need_sync"] = bool(need_sync)
            _p605_row["sync_taken"] = bool(not self.enable_enpu and need_sync)
        if not self.enable_enpu and need_sync:
            if _p605_row is not None:
                _p605_row["sync_enter_ns"] = time.monotonic_ns()
            torch.npu.current_stream().synchronize()
            if _p605_row is not None:
                _p605_row["sync_return_ns"] = time.monotonic_ns()
        if _p605_row is not None:
            _p605_row["replay_enter_ns"] = time.monotonic_ns()
        entry.aclgraph.replay()
        if _p605_row is not None:
            _p605_row["replay_return_ns"] = time.monotonic_ns()
        return entry.output
''')
    return src


def patch(key, src):
    if MARK in src:
        raise ValueError('Run605 already patched')
    if key == 'runner':
        return patch_runner(src)
    if key == 'draft':
        return patch_dspark(src)
    if key == 'dflash':
        return patch_dflash(src)
    if key == 'runtime':
        return patch_runtime(src)
    if key == 'acl_graph':
        return patch_acl_graph(src)
    return product.patch(key, src).replace('EXTREME_RUN602_DIR', 'EXTREME_RUN605_DIR')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def atomic(path, data):
    tmp = path.with_name(path.name + '.run605.tmp')
    if tmp.exists():
        raise ValueError('stale temporary file')
    with tmp.open('xb') as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())
    if path.exists():
        os.chmod(tmp, stat.S_IMODE(path.stat().st_mode))
    os.replace(tmp, path)


def prepared():
    paths = [path.resolve() for path in SOURCES.values()]
    if len(paths) != len(set(paths)):
        raise ValueError('duplicate source path')
    helper = product.base.HELPER.read_bytes()
    compile(helper, str(product.base.HELPER), 'exec')
    manifest = {'helper_sha256': sha(helper), 'sources': {}}
    old, new = {}, {}
    for key, path in SOURCES.items():
        data = path.read_bytes()
        if sha(data) != ORIGINAL[key]:
            raise ValueError(f'{key} original SHA drift: {sha(data)}')
        edited = patch(key, data.decode()).encode()
        compile(edited, str(path), 'exec')
        old[key], new[key] = data, edited
        manifest['sources'][key] = {'path': str(path), 'original_sha256': sha(data),
                                    'patched_sha256': sha(edited)}
    return manifest, old, new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('check', 'install', 'restore'))
    parser.add_argument('--state-dir', type=Path)
    parser.add_argument('--record', required=True, type=Path)
    parser.add_argument('--offline-confirmed', action='store_true')
    args = parser.parse_args()
    if args.action != 'check' and not args.offline_confirmed:
        raise ValueError('offline confirmation required')
    if args.action in ('check', 'install'):
        manifest, old, new = prepared()
        if args.action == 'install':
            if args.state_dir is None or args.state_dir.exists():
                raise ValueError('fresh state directory required')
            args.state_dir.mkdir(parents=True)
            for key, data in old.items():
                atomic(args.state_dir / f'{key}.orig', data)
            atomic(args.state_dir / 'manifest.json', (json.dumps(manifest, indent=2) + '\n').encode())
            try:
                for key, path in SOURCES.items():
                    if path.read_bytes() != old[key]:
                        raise ValueError('install race ' + key)
                    atomic(path, new[key])
            except BaseException:
                for key, path in SOURCES.items():
                    if sha(path.read_bytes()) == manifest['sources'][key]['patched_sha256']:
                        atomic(path, old[key])
                raise
    else:
        if args.state_dir is None:
            raise ValueError('state dir required')
        manifest = json.loads((args.state_dir / 'manifest.json').read_text())
        if set(manifest['sources']) != set(SOURCES):
            raise ValueError('source set drift')
        for key, path in SOURCES.items():
            row = manifest['sources'][key]
            if row['path'] != str(path) or row['original_sha256'] != ORIGINAL[key]:
                raise ValueError('source manifest drift ' + key)
            backup = (args.state_dir / f'{key}.orig').read_bytes()
            if sha(backup) != ORIGINAL[key]:
                raise ValueError('backup drift ' + key)
            if sha(path.read_bytes()) not in (row['original_sha256'], row['patched_sha256']):
                raise ValueError('installed source drift ' + key)
        for key, path in SOURCES.items():
            if sha(path.read_bytes()) != ORIGINAL[key]:
                atomic(path, (args.state_dir / f'{key}.orig').read_bytes())
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps({'action': args.action, **manifest}, indent=2) + '\n')
    print(json.dumps({'action': args.action, 'files': len(SOURCES)}))


if __name__ == '__main__':
    main()
