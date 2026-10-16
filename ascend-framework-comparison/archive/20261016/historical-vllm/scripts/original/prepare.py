#!/usr/bin/env python3
"""Prepare private AISBench data/configs; never sends inference requests."""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path


ROOT = Path(os.environ.get("ROOT", "/data/tiankuan/aisbench/warm_w2/codex_4p2d_20261008"))
TOKENIZER = Path(os.environ.get("TOKENIZER_PATH", "/data/tiankuan/aisbench/GLM-5.3-w8a8"))
GENERATOR_DIR = Path(os.environ.get("GENERATOR_DIR", "/data/tiankuan/aisbench/aisbench_auto_tools_prefix-main"))
SOURCE_DATA = Path(os.environ.get("GSM8K_SOURCE", GENERATOR_DIR / "GSM8K.jsonl"))
MODEL_HOST = os.environ.get("API_HOST", "172.16.10.168")
MODEL_PORT = int(os.environ.get("API_PORT", "9300"))
MODEL_NAME = os.environ.get("API_MODEL", "glm-53")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_jsonl(path: Path, questions: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps({"question": q, "answer": "none"}, ensure_ascii=False) + "\n")


def load_generator():
    # This preparation is CPU-only. Prevent torch from probing/loading the
    # container's Ascend backend extension (which is unavailable in some
    # client containers and is irrelevant to tokenization).
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    sys.path.insert(0, str(GENERATOR_DIR))
    work = ROOT / "generator_work"
    work.mkdir(parents=True, exist_ok=True)
    corpus_link = work / "GSM8K.jsonl"
    if not corpus_link.exists():
        corpus_link.symlink_to(SOURCE_DATA)
    os.chdir(work)
    from transformers import AutoTokenizer  # type: ignore
    from generate_dataset import create_dataset
    return AutoTokenizer, create_dataset


def make_salt_ids(tokenizer, seed: int, row: int) -> list[int]:
    """Mirror generate_unique_tokens' per-row RNG but retain token IDs."""
    rng = random.Random(seed + row)
    vocab_size = len(tokenizer)
    specials = set(getattr(tokenizer, "all_special_ids", []))
    ids: list[int] = []
    seen: set[int] = set()
    while len(ids) < 3:
        token_id = rng.randint(0, vocab_size - 1)
        if token_id not in seen and token_id not in specials:
            ids.append(token_id)
            seen.add(token_id)
    return ids


def stable_fill_id(tokenizer) -> int:
    """Find a non-special ID that survives one-token decode/encode."""
    specials = set(getattr(tokenizer, "all_special_ids", []))
    for candidate in range(len(tokenizer)):
        if candidate in specials:
            continue
        piece = tokenizer.decode([candidate], skip_special_tokens=False,
                                 clean_up_tokenization_spaces=False)
        if piece and tokenizer.encode(piece, add_special_tokens=False) == [candidate]:
            return candidate
    raise SystemExit("Could not find a stable non-special token for boundary correction")


def decode_exact_prompt(tokenizer, prefix: str, prefix_ids: list[int], salt_ids: list[int],
                        suffix_ids: list[int], fill_id: int,
                        row_label: str) -> tuple[str, int]:
    """Decode a prompt and correct only its suffix tail for exact tokenizer length."""
    tail = list(suffix_ids)
    adjustments = 0
    seen_lengths: set[int] = set()
    for _ in range(32):
        suffix_text = tokenizer.decode(salt_ids + tail, skip_special_tokens=False,
                                       clean_up_tokenization_spaces=False)
        # A literal newline separates the established common prefix from each
        # run-specific salt. The probe showed this keeps all 74,400 prefix IDs
        # intact; then adjust only the suffix tail to retain exact 80K length.
        text = prefix + "\n" + suffix_text
        encoded = tokenizer.encode(text, add_special_tokens=False)
        n_tokens = len(encoded)
        if encoded[:len(prefix_ids)] != prefix_ids:
            raise SystemExit(f"{row_label}: suffix boundary altered the shared prefix token IDs")
        if n_tokens == 80000:
            return text, adjustments
        delta = 80000 - n_tokens
        new_len = len(tail) + delta
        if new_len <= 0 or new_len in seen_lengths:
            delta = 1 if delta >= 0 else -1
            new_len = len(tail) + delta
        seen_lengths.add(len(tail))
        if new_len < 1:
            raise SystemExit(f"{row_label}: cannot correct token boundary without altering prefix")
        if new_len < len(tail):
            tail = tail[:new_len]
        else:
            tail.extend([fill_id] * (new_len - len(tail)))
        adjustments += abs(delta)
    raise SystemExit(f"{row_label}: could not produce exact 80,000-token prompt")


def config_model(alias: str, rate: float, cap: int, out_len: int) -> str:
    return f'''from ais_bench.benchmark.models import VLLMCustomAPIChatStream\nmodels = [dict(\n    attr="service", type=VLLMCustomAPIChatStream,\n    abbr="vllm-api-stream-chat", path={str(TOKENIZER)!r}, model={MODEL_NAME!r},\n    api_key="", request_rate={rate!r}, retry=2,\n    host_ip={MODEL_HOST!r}, host_port={MODEL_PORT}, url="",\n    max_out_len={out_len}, batch_size={cap},\n    generation_kwargs=dict(temperature=0, ignore_eos=True),\n)]\n'''


def config_dataset(alias: str, data_dir: Path) -> str:
    return f'''from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate\nfrom ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever\nfrom ais_bench.benchmark.openicl.icl_inferencer import GenInferencer\nfrom ais_bench.benchmark.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator\ngsm8k_reader_cfg = dict(input_columns=["question"], output_column="answer")\ngsm8k_infer_cfg = dict(prompt_template=dict(type=PromptTemplate, template="{{question}}"), retriever=dict(type=ZeroRetriever), inferencer=dict(type=GenInferencer))\ngsm8k_eval_cfg = dict(evaluator=dict(type=Gsm8kEvaluator), pred_role="BOT", pred_postprocessor=dict(type=gsm8k_postprocess), dataset_postprocessor=dict(type=gsm8k_dataset_postprocess))\ngsm8k_datasets = [dict(abbr="gsm8k", type=GSM8KDataset, path={str(data_dir)!r}, reader_cfg=gsm8k_reader_cfg, infer_cfg=gsm8k_infer_cfg, eval_cfg=gsm8k_eval_cfg)]\n'''


def make_run(name: str, rate: float, cap: int, n: int, seed: int,
             prefix: str, questions: list[str], kind: str,
             boundary_adjustments: list[int]) -> dict:
    run = ROOT / "runs" / name
    data = run / "data" / "gsm8k"
    cfg = run / "configs"
    write_jsonl(data / "test.jsonl", questions)
    (data / "train.jsonl").write_text("", encoding="utf-8")
    (cfg / "models").mkdir(parents=True, exist_ok=True)
    (cfg / "datasets").mkdir(parents=True, exist_ok=True)
    (cfg / "models" / f"{name}.py").write_text(config_model(name, rate, cap, 600), encoding="utf-8")
    (cfg / "datasets" / f"{name}_dataset.py").write_text(config_dataset(name, data), encoding="utf-8")
    return {"name": name, "kind": kind, "N": n, "C": cap, "request_rate": rate,
            "input_len": 80000, "output_len": 600, "repeat_rate": 0.93,
            "seed": seed, "dataset": str(data / "test.jsonl"),
            "dataset_sha256": sha(data / "test.jsonl"), "dataset_rows": len(questions),
            "shared_prefix_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
            "matching_distribution_not_byteidentical": True,
            "boundary_correction_tokens_min": min(boundary_adjustments),
            "boundary_correction_tokens_max": max(boundary_adjustments),
            "config_dir": str(cfg), "work_dir": str(run / "outputs"),
            "summarizer": "default_perf"}


def main() -> None:
    # The only allowed pre-existing entry is this copied preparation bundle.
    # Any prior generated data, configs, manifest, or outputs makes this a hard
    # stop so a retry cannot overwrite benchmark evidence.
    if ROOT.exists():
        allowed = {"benchmark", "scripts"}
        unexpected = [p.name for p in ROOT.iterdir() if p.name not in allowed]
        if unexpected:
            raise SystemExit(f"Refusing to overwrite existing ROOT content: {unexpected}")
    if not SOURCE_DATA.is_file() or not TOKENIZER.exists():
        raise SystemExit(f"Missing tokenizer or source dataset: {TOKENIZER} / {SOURCE_DATA}")
    ROOT.mkdir(parents=True, exist_ok=True)
    AutoTokenizer, create_dataset = load_generator()
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER))
    # Use the established generator for the common prefix, then reuse that exact
    # prefix across all runs. Picked IDs are isolated in this private generator cwd.
    random.seed(1)
    prefix_rows = create_dataset(str(TOKENIZER), int(80000 * .93), 1, 1)
    if not prefix_rows or len(prefix_rows) != 1:
        raise SystemExit("Could not generate one common 74,400-token prefix")
    prefix = prefix_rows[0]
    if len(tokenizer.encode(prefix, add_special_tokens=False)) != 74400:
        raise SystemExit("Common prefix does not encode to 74,400 tokens")

    # Sixteen warmup rows, one shared prefix, used per run. Generation does not
    # contact the inference service.
    write_jsonl(ROOT / "shared" / "prefix_warmup" / "gsm8k" / "test.jsonl", [prefix] * 16)
    (ROOT / "shared" / "prefix_warmup" / "gsm8k" / "train.jsonl").write_text("", encoding="utf-8")

    plans: list[tuple[str, float, int, int, int, str]] = []
    for cap in (16, 32, 64, 96):
        plans.append((f"sat_c{cap}_pilot", 0.0, cap, 384, 10000 + cap, "saturation"))
    for cap in (16, 32):
        plans.append((f"rate093_c{cap}_pilot", 0.93, cap, 48,
                      20000 + cap, "arrival_limited"))

    manifests = []
    fill_id = stable_fill_id(tokenizer)
    for name, rate, cap, n, seed, kind in plans:
        # AISBench's existing generator chooses suffix examples from GSM8K and
        # creates independent 3-token salts. Reset picked IDs per run to keep
        # suffix generation valid; use matched prefix/length/rate distribution.
        random.seed(seed)
        random.seed(seed)
        suffixes = create_dataset(str(TOKENIZER), 80000 - 74400 - 3, n, 0)
        if suffixes is None or len(suffixes) != n:
            raise SystemExit(f"Failed suffix generation for {name}")
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        questions = []
        boundary_adjustments = []
        for i, suffix in enumerate(suffixes):
            suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
            if not 5500 <= len(suffix_ids) <= 5700:
                raise SystemExit(f"{name}: suffix row {i} tokenized to implausible length {len(suffix_ids)}")
            q, adjustment = decode_exact_prompt(tokenizer, prefix, prefix_ids,
                                                make_salt_ids(tokenizer, seed, i),
                                                suffix_ids, fill_id, f"{name} row {i}")
            questions.append(q)
            boundary_adjustments.append(adjustment)
        manifests.append(make_run(name, rate, cap, n, seed, prefix, questions, kind,
                                  boundary_adjustments))

    # Separate 4-row prefix warmup config, requested output=1 and rate=0 to
    # finish warmup quickly. It is deliberately excluded from SLA outputs.
    warm_dir = ROOT / "shared" / "prefix_warmup" / "configs"
    (warm_dir / "models").mkdir(parents=True, exist_ok=True)
    (warm_dir / "datasets").mkdir(parents=True, exist_ok=True)
    (warm_dir / "models" / "prefix_warmup.py").write_text(config_model("prefix_warmup", 0.0, 16, 1), encoding="utf-8")
    (warm_dir / "datasets" / "prefix_warmup_dataset.py").write_text(
        config_dataset("prefix_warmup", ROOT / "shared" / "prefix_warmup" / "gsm8k"), encoding="utf-8")
    # Load every private config with the installed mmengine parser. This is
    # config validation only; it never constructs a client or sends requests.
    from mmengine.config import Config
    for item in manifests:
        model_cfg = Config.fromfile(str(Path(item["config_dir"]) / "models" / f'{item["name"]}.py'))
        data_cfg = Config.fromfile(str(Path(item["config_dir"]) / "datasets" / f'{item["name"]}_dataset.py'))
        assert len(model_cfg.models) == 1 and model_cfg.models[0]["batch_size"] == item["C"]
        assert model_cfg.models[0]["request_rate"] == item["request_rate"]
        assert model_cfg.models[0]["max_out_len"] == 600
        assert len(data_cfg.gsm8k_datasets) == 1
        assert Path(data_cfg.gsm8k_datasets[0]["path"]).resolve() == Path(item["dataset"]).parent.resolve()
        assert Path(item["dataset"]).is_file() and Path(item["dataset"]).stat().st_size > 0
    warm_m = Config.fromfile(str(warm_dir / "models" / "prefix_warmup.py"))
    warm_d = Config.fromfile(str(warm_dir / "datasets" / "prefix_warmup_dataset.py"))
    assert warm_m.models[0]["max_out_len"] == 1 and warm_m.models[0]["batch_size"] == 16
    assert len(warm_d.gsm8k_datasets) == 1
    manifest = {"root": str(ROOT), "tokenizer": str(TOKENIZER), "source_dataset_sha256": sha(SOURCE_DATA),
                "input_len": 80000, "output_len": 600, "repeat_rate": 0.93,
                "input_length_basis": "raw question tokenizer IDs before AISBench/API chat template; actual served input_tokens must be recorded from run details",
                "prefix_len": 74400, "prefix_num": 1, "prefix_warmup_rows": 16,
                "pilot_N": 384, "pilot_warning": "Report this run's P99 against the gates, but treat it as a pilot sample rather than a production long-run stability claim.",
                "privacy": "No prompt or prediction text is included in this manifest.", "runs": manifests}
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared private data/configs only: {ROOT}")
    print(f"Runs: {len(manifests)}; private mmengine configs loaded/checked; no AISBench CLI was invoked.")


if __name__ == "__main__":
    main()
