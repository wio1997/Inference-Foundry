"""R4 cache contract probe. Run inside the authorized container.

CPU torch contract checks do not establish NPU speed or real-service correctness.
The two rejection tests exercise the retained baseline validation fragment below;
they must be followed by actual candidate integration tests and model requests.
"""
import argparse
import importlib.util
import json
import pathlib
import sys
from types import SimpleNamespace

import torch


class Mode:
    def __init__(self, name): self.name = name
    def is_extend(self): return self.name == "extend"
    def is_decode(self): return self.name == "decode"
    def is_target_verify(self): return self.name == "verify"
    def is_draft_extend_v2(self): return self.name == "draft"


def state(name="extend"):
    backend = SimpleNamespace(graph_mode=False, forward_metadata=SimpleNamespace())
    batch = SimpleNamespace(forward_mode=Mode(name), spec_info=None,
                            can_run_tbo=False, forward_metadata_ready=False,
                            seq_lens=torch.tensor([2], dtype=torch.int32),
                            req_pool_indices=torch.tensor([3], dtype=torch.int32),
                            extend_seq_lens=torch.tensor([2], dtype=torch.int32))
    return backend, batch, torch.tensor([0, 1], dtype=torch.int64)


def retained_validation(values, count, request_capacity=16):
    # Fixed engine flash_kpool_indexer_npu.py:66-76 equivalent fragment.
    # Actual production retained code still needs source/integration verification.
    lengths, sequences, request_slots, actual_positions = values
    if sum(lengths) != count or len(lengths) != len(sequences) or len(lengths) != len(request_slots):
        raise ValueError("Flash KPool actual-token/request boundaries mismatch")
    active_slots = [slot for slot, length in zip(request_slots, lengths) if length]
    if len(set(active_slots)) != len(active_slots):
        raise ValueError("Duplicate active request tail slots")
    offset = 0
    for length, sequence, slot in zip(lengths, sequences, request_slots):
        if actual_positions[offset:offset+length] != list(range(sequence-length, sequence)):
            raise ValueError("Flash KPool positions must be contiguous at the actual prefix")
        if length and not 0 <= slot < request_capacity:
            raise ValueError("Invalid actual Flash request tail slot")
        offset += length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--helper", default=str(pathlib.Path(__file__).with_name("flash_kpool_metadata_cache_npu.py")))
    parser.add_argument("--output")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("r4_kpool_cache_probe_target", args.helper)
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    calls = []
    original = helper._read
    def counted_read(*xs):
        calls.append(1)
        return original(*xs)
    helper._read = counted_read
    results = []
    def observe(name, fn):
        start = len(calls)
        try:
            detail = fn()
            results.append(dict(name=name, status="PASS", cpu_read_sets=len(calls)-start, detail=detail))
        except Exception as exc:
            results.append(dict(name=name, status="FAIL", cpu_read_sets=len(calls)-start,
                                exception=type(exc).__name__, message=str(exc)))
    def invoke(st, enabled=True, count=None):
        backend, batch, positions = st
        query = (batch.extend_seq_lens.to(torch.int32) if batch.forward_mode.is_extend()
                 else torch.ones_like(batch.seq_lens, dtype=torch.int32))
        if count is None: count = positions.numel()
        return helper.read_kpool_cpu_metadata(
            backend, batch, positions, query, batch.seq_lens.to(torch.int32),
            batch.req_pool_indices.to(torch.int32), count, enabled=enabled)
    def reuse():
        st = state()
        first, event1 = invoke(st)
        second, event2 = invoke(st)
        assert (event1, event2) == ("miss", "hit") and first == second
        first[0][0] = 99
        third, event3 = invoke(st)
        assert third[0] == [2] and event3 == "hit"
        return dict(events=[event1, event2, event3], isolated_return_lists=True)
    observe("first_read_then_repeat_reuse", reuse)
    def new_metadata():
        st = state(); invoke(st)
        st[0].forward_metadata = SimpleNamespace()
        _, event = invoke(st); assert event == "miss"
        return dict(event=event)
    observe("metadata_generation_change", new_metadata)
    def replacement():
        st = state(); invoke(st)
        st[1].req_pool_indices = st[1].req_pool_indices.clone()
        _, event = invoke(st); assert event == "miss"
        return dict(event=event)
    observe("new_tensor_same_values_invalidates", replacement)
    def inplace():
        st = state(); invoke(st)
        st[1].seq_lens.add_(1)
        values, event = invoke(st)
        assert event == "miss" and values[1] == [3]
        return dict(event=event, observed_sequence=values[1])
    observe("normal_inplace_version_invalidates", inplace)
    def layout_count():
        st = state(); invoke(st)
        st[2].resize_(3); st[2][2] = 2
        _, event = invoke(st, count=3); assert event == "miss"
        return dict(event=event)
    observe("shape_count_layout_change_invalidates", layout_count)
    def disabled():
        st = state()
        assert invoke(st, False)[1] == "bypass" and invoke(st, False)[1] == "bypass"
        return dict(default_off=True)
    observe("disabled_retains_original_reads", disabled)
    def guards():
        scenarios = []
        for field, value in [("can_run_tbo", True), ("spec_info", object()),
                             ("forward_metadata_ready", True)]:
            st = state(); setattr(st[1], field, value)
            assert invoke(st)[1] == "bypass"; scenarios.append(field)
        st = state(); st[0].graph_mode = True
        assert invoke(st)[1] == "bypass"; scenarios.append("graph_mode")
        st = state("verify"); assert invoke(st)[1] == "bypass"
        scenarios.append("target_verify")
        return dict(bypassed=scenarios)
    observe("unsupported_lifetimes_bypass", guards)
    def inference():
        with torch.inference_mode():
            st = state()
            _, event1 = invoke(st); _, event2 = invoke(st)
            assert (event1, event2) == ("miss", "hit")
            st[0].forward_metadata = SimpleNamespace()
            _, event3 = invoke(st); assert event3 == "miss"
        return dict(events=[event1,event2,event3], limitation="same-object native/inference writes are not generally detected; cache requires proven read-only eager generation")
    observe("inference_tensor_readonly_generation", inference)
    def rejections():
        checked = []
        for kind in ["positions", "duplicate_slots"]:
            st = state()
            if kind == "positions": st[2][1] = 7
            else:
                st[1].seq_lens = torch.tensor([1,1],dtype=torch.int32)
                st[1].extend_seq_lens = torch.tensor([1,1],dtype=torch.int32)
                st[1].req_pool_indices = torch.tensor([3,3],dtype=torch.int32)
                st = (st[0],st[1],torch.tensor([0,0],dtype=torch.int64))
            for expected_event in ["miss", "hit"]:
                values,event = invoke(st)
                assert event == expected_event
                try: retained_validation(values, 2)
                except ValueError as exc: checked.append(dict(kind=kind,event=event,message=str(exc)))
                else: raise AssertionError("retained validation failed to reject bad metadata")
        return dict(rejections=checked, scope="retained validation fragment; real integration rejection still required")
    observe("bad_positions_and_duplicate_slots_still_rejected", rejections)
    output = dict(schema_version=1, scope="CPU torch draft-helper contract; no NPU benchmark/model validation", results=results,
                  passed=sum(r["status"]=="PASS" for r in results), failed=sum(r["status"]=="FAIL" for r in results))
    rendered = json.dumps(output,indent=2)+"\n"
    if args.output: pathlib.Path(args.output).write_text(rendered,encoding="utf-8")
    print(rendered,end="")
    return int(output["failed"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
