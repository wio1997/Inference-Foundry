#!/usr/bin/env python3
"""SHA-pinned source preview for a no-publication pause/resume A0/B/A1.

`check` never changes runtime sources. Install/restore require an offline
controller and a fresh state directory; no service launch is performed here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import loop079_formal_ledger_patch as base


ROOT = Path("/data/wio/Inference_Foundry")
RUNNER = Path("/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/worker/model_runner_v1.py")
SEGMENTED = ROOT / "runtime/segmented_serving.py"
CANDIDATE = ROOT / "evidence/20260928_loop080_bound/run562/segmented_serving_candidate.py"
ORIGINAL = {
    "runner": "004dbd0d5b1a5c3f9533fa1d12327bd0ae421fe24bb2674544b2c4a25d74aaba",
    "segmented": "389e600bbda01b667323ee51622c39fcccfcabf2b54a954febade1d5d46bea7e",
}
CANDIDATE_SHA = "169a6ad7f77698c4c5172c2aa755130fe15e7426f610458532624e2efef87e49"
HELPER_SHA = "3dd36594271405bfb852327ea0e0200d9ddabda0eab4454d7241794fab31f64f"
SOURCES = {"runner": RUNNER, "segmented": SEGMENTED}
MARK = "EXTREME_LOOP080_PAUSE_CONTROL_RUN564"


def patch_runner(src: str) -> str:
    if MARK in src:
        raise ValueError("pause control already patched")
    src = base.once(
        src,
        "                _cohort_output = FixedCohortServing(\n"
        "                    _extreme_runtime,\n"
        "                    initial_output_counts=[0] * _cfg.batch_size,\n"
        "                    max_output_tokens=1024,\n"
        "                ).run()\n",
        "                # EXTREME_LOOP080_PAUSE_CONTROL_RUN564\n"
        "                _pause_mode = os.getenv('EXTREME_PAUSE_MODE', 'off')\n"
        "                _pause_cohort = int(os.getenv('EXTREME_PAUSE_COHORT', '5'))\n"
        "                _pause_generation = len(self._extreme_served_cohorts)\n"
        "                if _pause_mode not in ('off', 'on'):\n"
        "                    raise ValueError('invalid pause control mode')\n"
        "                if _pause_mode == 'on' and _pause_generation == _pause_cohort:\n"
        "                    from runtime.segmented_serving import SegmentedCohortServing\n"
        "                    _pause_dir = os.getenv('EXTREME_PAUSE_TRACE_DIR')\n"
        "                    if not _pause_dir:\n"
        "                        raise ValueError('pause control requires trace dir')\n"
        "                    _paused = SegmentedCohortServing(\n"
        "                        _extreme_runtime, generation=_pause_generation,\n"
        "                        request_ids=_extreme_req_ids,\n"
        "                        initial_output_counts=[0] * _cfg.batch_size,\n"
        "                        max_output_tokens=1024,\n"
        "                    )\n"
        "                    _segment = _paused.pause_after_first()\n"
        "                    _cohort_output = _paused.finish_after_pause(\n"
        "                        expected_generation=_pause_generation)\n"
        "                    os.makedirs(_pause_dir, exist_ok=True)\n"
        "                    _pause_rank = int(get_tp_group().rank_in_group)\n"
        "                    from dataclasses import asdict as _pause_asdict\n"
        "                    with open(os.path.join(_pause_dir,\n"
        "                        f'pause_rank{_pause_rank}_cohort{_pause_generation}.json'),\n"
        "                        'w', encoding='utf-8') as _pause_file:\n"
        "                        json.dump({'segment': _pause_asdict(_segment),\n"
        "                                   'trajectory': _paused.trajectory,\n"
        "                                   'resume_pause_ms': _paused.resume_pause_ms},\n"
        "                                  _pause_file, separators=(',', ':'))\n"
        "                else:\n"
        "                    _cohort_output = FixedCohortServing(\n"
        "                        _extreme_runtime,\n"
        "                        initial_output_counts=[0] * _cfg.batch_size,\n"
        "                        max_output_tokens=1024,\n"
        "                    ).run()\n",
    )
    src = base.once(
        src,
        "                if _rank == 0 and os.getenv(\"EXTREME_RUNTIME_DIAGNOSE\") == \"1\":\n",
        "                # EXTREME_LOOP080_PAUSE_CONTROL_RUN564\n"
        "                if (os.getenv('EXTREME_PAUSE_TRACE_DIR')\n"
        "                    and 5 <= _cohort_index <= 8\n"
        "                    and os.getenv(\"EXTREME_RUNTIME_DIAGNOSE\") == \"1\"):\n",
    )
    src = base.once(
        src,
        "                        os.path.join(_extreme_run_dir, \"trace_rank0.json\"),\n",
        "                        os.path.join(os.getenv('EXTREME_PAUSE_TRACE_DIR'),\n"
        "                            f'trace_rank{_rank}_cohort{_cohort_index}.json'),\n",
    )
    src = base.once(
        src,
        "                    os.makedirs(_extreme_run_dir, exist_ok=True)\n"
        "                    with open(\n"
        "                        os.path.join(os.getenv('EXTREME_PAUSE_TRACE_DIR'),\n",
        "                    os.makedirs(os.getenv('EXTREME_PAUSE_TRACE_DIR'), exist_ok=True)\n"
        "                    with open(\n"
        "                        os.path.join(os.getenv('EXTREME_PAUSE_TRACE_DIR'),\n",
    )
    return src


def preview():
    if base.sha(Path(base.__file__).read_bytes()) != HELPER_SHA:
        raise ValueError("atomic patch helper SHA drift")
    originals = {key: path.read_bytes() for key, path in SOURCES.items()}
    for key, raw in originals.items():
        if base.sha(raw) != ORIGINAL[key]:
            raise ValueError(f"{key} source SHA drift: {base.sha(raw)}")
    candidate = CANDIDATE.read_bytes()
    if base.sha(candidate) != CANDIDATE_SHA:
        raise ValueError("Run562 candidate SHA drift")
    patched = {"runner": patch_runner(originals["runner"].decode()).encode(),
               "segmented": candidate}
    for key, data in patched.items():
        compile(data, str(SOURCES[key]), "exec")
    files = {
        key: dict(path=str(SOURCES[key]), original=base.sha(originals[key]),
                  patched=base.sha(patched[key]), bytes=len(patched[key]))
        for key in SOURCES
    }
    return files, originals, patched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "install", "restore"))
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--offline-confirmed", action="store_true")
    args = parser.parse_args()
    if args.action in ("install", "restore") and not args.offline_confirmed:
        raise ValueError("live source mutation requires guarded offline confirmation")
    if base.sha(Path(base.__file__).read_bytes()) != HELPER_SHA:
        raise ValueError("atomic patch helper SHA drift")
    if args.action in ("check", "install"):
        files, originals, patched = preview()
        record = dict(action=args.action, marker=MARK, files=files,
                      candidate_sha256=CANDIDATE_SHA,
                      helper_sha256=HELPER_SHA,
                      patcher_sha256=base.sha(Path(__file__).read_bytes()))
        if args.action == "install":
            if args.state_dir is None or args.state_dir.exists():
                raise ValueError("install requires a fresh state-dir")
            args.state_dir.mkdir(parents=True)
            for key in SOURCES:
                base.atomic_replace(args.state_dir / f"{key}.original.py", originals[key])
            base.write_json(args.state_dir / "manifest.json", record)
            for key, path in SOURCES.items():
                if base.sha(path.read_bytes()) != files[key]["original"]:
                    raise ValueError(f"{key} changed before install")
                base.atomic_replace(path, patched[key])
                if base.sha(path.read_bytes()) != files[key]["patched"]:
                    raise ValueError(f"{key} patched bytes differ")
    else:
        if args.state_dir is None:
            raise ValueError("restore requires state-dir")
        manifest = json.loads((args.state_dir / "manifest.json").read_text())
        if manifest["marker"] != MARK or set(manifest["files"]) != set(SOURCES):
            raise ValueError("state manifest mismatch")
        if (manifest["candidate_sha256"] != CANDIDATE_SHA
                or manifest["helper_sha256"] != HELPER_SHA
                or manifest["patcher_sha256"] != base.sha(Path(__file__).read_bytes())):
            raise ValueError("restore tool provenance mismatch")
        if base.sha(CANDIDATE.read_bytes()) != CANDIDATE_SHA:
            raise ValueError("candidate drift before restore")
        for key, path in SOURCES.items():
            saved = (args.state_dir / f"{key}.original.py").read_bytes()
            expected = manifest["files"][key]
            known_patched = (base.sha(patch_runner(saved.decode()).encode())
                             if key == "runner" else CANDIDATE_SHA)
            if (expected["path"] != str(path)
                    or expected["original"] != ORIGINAL[key]
                    or expected["patched"] != known_patched
                    or base.sha(saved) != ORIGINAL[key]):
                raise ValueError(f"{key} backup changed")
            if base.sha(path.read_bytes()) not in (expected["original"], expected["patched"]):
                raise ValueError(f"{key} has unknown source bytes")
        for key, path in SOURCES.items():
            saved = (args.state_dir / f"{key}.original.py").read_bytes()
            if base.sha(path.read_bytes()) != manifest["files"][key]["original"]:
                base.atomic_replace(path, saved)
            if base.sha(path.read_bytes()) != ORIGINAL[key]:
                raise ValueError(f"{key} restore verification failed")
        record = dict(manifest, action="restore", restored=True)
    base.write_json(args.record, record)
    print(json.dumps(dict(action=args.action, marker=MARK,
                          files=record["files"], record=str(args.record)),
                     separators=(",", ":")))


if __name__ == "__main__":
    main()
