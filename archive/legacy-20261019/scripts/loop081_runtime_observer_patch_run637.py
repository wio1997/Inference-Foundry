#!/usr/bin/env python3
"""Run637 reversible observer candidate generator; --check only, never installs."""
from __future__ import annotations
import ast, difflib, hashlib, json
from pathlib import Path

ROOT=Path('/data/wio/Inference_Foundry')
ASC=Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend')
OUT=ROOT/'evidence/20260928_loop081_bound/run637'
SOURCES={
 'cycle':(ROOT/'runtime/extreme_decode.py','eb4b5142cc475a51e06c6eec8238e9cfeb918b568e8a5f27b0a4bc7d26963499'),
 'target':(ROOT/'runtime/target_adapter.py','c46d60318098077064c0ff57eae98bc937dcf943bda531918388be4e9128acd5'),
 'serving':(ROOT/'runtime/fixed_serving.py','137f1cc6a460700e6f26e5837aa062022eb621198e33099b85988f8b7ef6743a'),
 'draft':(ROOT/'bootstrap/vllm_dspark_handoff.py','fe7039361894acd70666f72c3089efd635404f1fe9cbc685db21272d946a317e'),
 'runner':(ASC/'worker/model_runner_v1.py','004dbd0d5b1a5c3f9533fa1d12327bd0ae421fe24bb2674544b2c4a25d74aaba'),
}
def digest(data): return hashlib.sha256(data).hexdigest()
def one(src,old,new):
    if src.count(old)!=1: raise AssertionError(f'anchor count {src.count(old)} {old[:70]!r}')
    return src.replace(old,new,1)
def cycle(src):
    src=one(src,'        self.diagnostic_events = []\n','''        self.diagnostic_events = []
        # EXTREME_RUN637_BOUND_PROBE: optional; events warm-recorded before wall timer.
        self._bound_probe = None
        _bp_ledger_dir = os.getenv("EXTREME_BOUND_LEDGER_DIR")
        if os.getenv("EXTREME_BOUND_PACKET_DIR") and not _bp_ledger_dir:
            raise RuntimeError("bound packet requires common OFF/ON ledger")
        _bp_armed = bool(_bp_ledger_dir and os.path.isfile(
            os.path.join(_bp_ledger_dir, "arm")))
        self._bound_ledger_armed = _bp_armed
        if _bp_armed and os.getenv("EXTREME_BOUND_PACKET_DIR"):
            if self._profile_dag or self._diagnose or self._cycle_profile_dir:
                raise RuntimeError("bound event packet incompatible with active profiler")
            if target_page_audit is not None or kv_slot_audit is not None:
                raise RuntimeError("bound event packet excludes debug page/KV audits")
            from .bound_observer import BoundEventRecorder
            self._bound_probe = BoundEventRecorder(
                (63, 64, 65),
                current_stream=torch.npu.current_stream(state.target_positions.device),
                copy_stream=getattr(proposer, "_host_copy_stream", None))
            target._bound_probe = self._bound_probe
            proposer._bound_probe = self._bound_probe
''')
    src=one(src,'''                markers.append((label, event))
        with self._scope("extreme::cycle"):
''','''                markers.append((label, event))
            if self._bound_probe is not None:
                _bound_names = {
                    "begin": "cycle_begin", "prepare_target": "prepare_target",
                    "derived_target_metadata": "derived_target_metadata",
                    "target": "target_after", "acceptance": "acceptance_after",
                    "state_advance": "state_advance_after",
                    "proposer": "proposer_after", "draft_commit": "draft_commit_after",
                }
                self._bound_probe.mark(_bound_cycle, _bound_names[label])
        _bound_cycle = self.state.cycle_index
        with self._scope("extreme::cycle"):
''')
    src=one(src,'''                with self._scope("extreme::target"):
                    if self._cycle_profiler''','''                with self._scope("extreme::target"):
                    if self._bound_probe is not None:
                        self._bound_probe.mark(_bound_cycle, "target_before")
                    if self._cycle_profiler''')
    src=one(src,'''            with self._scope("extreme::proposer"):
                next_draft =''','''            with self._scope("extreme::proposer"):
                if self._bound_probe is not None:
                    self._bound_probe.mark(_bound_cycle, "proposer_before")
                next_draft =''')
    return src
def target(src):
    src=one(src,'''        model_output = self.binding.forward(
            state.target_input_ids,
            state.target_positions,
        )
''','''        model_output = self.binding.forward(
            state.target_input_ids,
            state.target_positions,
        )
        if getattr(self, "_bound_probe", None) is not None:
            self._bound_probe.mark(state.cycle_index, "target_forward_return")
''')
    return one(src,'''        logits = self.binding.compute_logits(sample_hidden)
        return TargetOutput(''','''        logits = self.binding.compute_logits(sample_hidden)
        if getattr(self, "_bound_probe", None) is not None:
            self._bound_probe.mark(state.cycle_index, "target_logits_return")
        return TargetOutput(''')
def serving(src):
    src=one(src,'''            count_history[index].copy_(self.runtime.state.num_sampled)
            cycles = index + 1
''','''            count_history[index].copy_(self.runtime.state.num_sampled)
            if self.runtime._bound_probe is not None:
                self.runtime._bound_probe.mark(result.cycle - 1, "serve_stage_after")
            cycles = index + 1
''')
    src=one(src,'''            self._park_completed(progress)
            if all(
''','''            self._park_completed(progress)
            if self.runtime._bound_probe is not None:
                self.runtime._bound_probe.mark(result.cycle - 1, "serve_park_after")
            if all(
''')
    return one(src,'''        counts_cpu = count_history[:cycles].cpu()
        output: list[list[int]] =''','''        counts_cpu = count_history[:cycles].cpu()
        if self.runtime._bound_ledger_armed:
            self.runtime._bound_cpu_history = (tokens_cpu, counts_cpu)
        output: list[list[int]] =''')
def draft(src):
    src=one(src,'''        self._host_copy_event.synchronize()
        batch = self.config.batch_size
''','''        _bp = getattr(self, "_bound_probe", None)
        if _bp is not None:
            _bp.wait_start(self._bound_current_cycle,
                           getattr(self, "_bound_previous_copy_cycle", -1))
        self._host_copy_event.synchronize()
        if _bp is not None:
            _bp.wait_end(self._bound_current_cycle)
        batch = self.config.batch_size
''')
    src=one(src,'''        with torch.npu.stream(self._host_copy_stream):
            self._host_count_copy.copy_(counts, non_blocking=True)
            self._host_copy_event.record()
        self._host_copy_pending = True
''','''        with torch.npu.stream(self._host_copy_stream):
            _bp = getattr(self, "_bound_probe", None)
            if _bp is not None:
                _bp.mark(self._bound_current_cycle, "host_copy_before", stream="copy")
            self._host_count_copy.copy_(counts, non_blocking=True)
            self._host_copy_event.record()
            if _bp is not None:
                _bp.mark(self._bound_current_cycle, "host_copy_after", stream="copy")
        if _bp is not None:
            self._bound_previous_copy_cycle = self._bound_current_cycle
        self._host_copy_pending = True
''')
    return one(src,'''        mark("begin")
        # Consume the previous cycle's count copy''','''        mark("begin")
        if getattr(self, "_bound_probe", None) is not None:
            self._bound_current_cycle = state.cycle_index
        # Consume the previous cycle's count copy''')
def runner(src):
    return one(src,'''                _extreme_wall_seconds = (
                    time.perf_counter() - _extreme_wall_start
                )
                _host_mirror_exact =''','''                _extreme_wall_seconds = (
                    time.perf_counter() - _extreme_wall_start
                )
                # Existing terminal synchronize above precedes extraction.
                # torch_npu elapsed_time itself calls sync APIs; ON Product
                # publication is perturbed and requires OFF/ON/OFF admission.
                if _extreme_runtime._bound_ledger_armed:
                    import hashlib
                    _bp_ledger_dir = os.environ["EXTREME_BOUND_LEDGER_DIR"]
                    os.makedirs(_bp_ledger_dir, exist_ok=True)
                    _bp_rank = int(get_tp_group().rank_in_group)
                    _bp_cohort = len(self._extreme_served_cohorts)
                    _bp_tokens, _bp_counts = _extreme_runtime._bound_cpu_history
                    _bp_hash = hashlib.sha256()
                    for _bp_i in range(_cohort_output.cycles):
                        for _bp_slot in range(12):
                            _bp_count = int(_bp_counts[_bp_i, _bp_slot])
                            if not 0 <= _bp_count <= 8:
                                raise RuntimeError("bound ledger acceptance count outside fixed contract")
                            _bp_hash.update(bytes((_bp_count,)))
                            for _bp_token in _bp_tokens[_bp_i, _bp_slot, :_bp_count].tolist():
                                _bp_hash.update(int(_bp_token).to_bytes(4, "little", signed=True))
                    _bp_basis = {
                        "cycles": _cohort_output.cycles,
                        "canonical_effective_staged_trajectory_sha256": _bp_hash.hexdigest(),
                        "accepted_count_history_sha256": hashlib.sha256(
                            _bp_counts.contiguous().numpy().tobytes()).hexdigest(),
                        "raw_padded_token_history_sha256": hashlib.sha256(
                            _bp_tokens.contiguous().numpy().tobytes()).hexdigest(),
                    }
                    _bp_identity = {
                        "rank": _bp_rank, "cohort": _bp_cohort,
                        "req_ids": list(_extreme_req_ids),
                        "run_ts": os.getenv("RUN_TS"),
                        "time_namespace": os.readlink("/proc/self/ns/time"),
                        "runtime_scoped_wall_s": _extreme_wall_seconds,
                        "runtime_basis": _bp_basis,
                        "generated_output_counts": _cohort_output.generated_output_counts,
                        "product_output_sha256": hashlib.sha256(json.dumps(
                            _cohort_output.token_ids, separators=(",", ":")
                        ).encode()).hexdigest(),
                    }
                    with open(os.path.join(_bp_ledger_dir, f"rank{_bp_rank}_cohort{_bp_cohort}.json"),
                              "w", encoding="utf-8") as _bp_file:
                        json.dump(_bp_identity, _bp_file, separators=(",", ":"))
                    if _extreme_runtime._bound_probe is not None:
                        _bp_dir = os.environ["EXTREME_BOUND_PACKET_DIR"]
                        os.makedirs(_bp_dir, exist_ok=True)
                        _bp_packet = _extreme_runtime._bound_probe.export_after_existing_drain(
                            _bp_identity)
                        with open(os.path.join(_bp_dir, f"rank{_bp_rank}_cohort{_bp_cohort}.json"),
                                  "w", encoding="utf-8") as _bp_file:
                            json.dump(_bp_packet, _bp_file, separators=(",", ":"))
                _host_mirror_exact =''')
PATCH={'cycle':cycle,'target':target,'serving':serving,'draft':draft,'runner':runner}
def main():
    observer=ROOT/'runtime/bound_observer.py'
    ast.parse(observer.read_text())
    manifest={'status':'candidate_only_NOT_INSTALLED_NOT_LIVE_READY',
              'observer_module_sha256':digest(observer.read_bytes()),'sources':{}}
    dest=Path('/tmp/loop081_run637_candidate'); dest.mkdir(parents=True,exist_ok=True)
    diffs=OUT/'candidate_diff'; diffs.mkdir(parents=True,exist_ok=True)
    for key,(path,expected) in SOURCES.items():
        raw=path.read_bytes()
        if digest(raw)!=expected: raise AssertionError(f'source drift {key}')
        before=raw.decode(); after=PATCH[key](before); ast.parse(after)
        for needle in ('.synchronize(','.wait_stream(','.wait_event('):
            if after.count(needle)!=before.count(needle):
                raise AssertionError(f'new {needle} in {key}')
        candidate=dest/f'{key}.py'; candidate.write_text(after)
        diff=diffs/f'{key}.diff'
        diff.write_text(''.join(difflib.unified_diff(
            before.splitlines(keepends=True),after.splitlines(keepends=True),
            fromfile=f'live/{key}.py',tofile=f'candidate/{key}.py')))
        manifest['sources'][key]={'live_path':str(path),'live_sha256':expected,
                                  'candidate_sha256':digest(candidate.read_bytes()),
                                  'diff_path':str(diff),'diff_sha256':digest(diff.read_bytes())}
    output=OUT/'observer_patch_preflight.json'
    output.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'status':manifest['status'],'sha256':digest(output.read_bytes())}))
if __name__=='__main__': main()
