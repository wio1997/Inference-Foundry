#!/usr/bin/env python3
"""Generate E27 32K frozen token-ID workloads and static AISBench configs.

The generator creates one new output directory and eight new files under
configs/e27/. Existing output or config paths are rejected, never replaced.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

HOST_ROOT = Path(__file__).resolve().parent.parent
CONTAINER_ROOT = "/work/performance-20261021"
SOURCE_REL = "data/workloads/E26-L32K-c16.jsonl"
SUPPORTED_LENGTH = 32768
SUPPORTED_OUTPUT = 128
SUPPORTED_CONCURRENCY = (32, 64, 128)
PHASES = (
    ("SHAPE-W1", "shape", 128),
    ("NORMAL-R1", "normal", 128),
    ("HOT-W1", "hot", 1),
    ("HOT-R1", "hot", 128),
)
FRAMEWORKS = (
    ("DI", "deepinfer", 18121),
    ("V", "vllm", 18122),
)
WARM_SEED_OFFSET = 100000
SENTINEL_SEED_OFFSET = 900000
WORKLOAD_FILES = (
    "corpus.jsonl",
    "shape-warm.jsonl",
    "di-prefix.jsonl",
    "v-prefix.jsonl",
    "questions.jsonl",
)
MODEL_NAME = "DeepSeek-V4.1-Flash-1017"
MODEL_PATH = "/data/wio/DeepSeek-V4.1-Flash-1017"


class PrepareError(RuntimeError):
    pass


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Prepare frozen E27 32K token-ID workloads and AISBench configs."
    )
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--output", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--requests", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026102232)
    parser.add_argument("--cached-tokens", type=int, default=None)
    parser.add_argument("--v-warm-prefix", type=int, default=None)
    return parser.parse_args(argv)


def validate_cli(args):
    if args.length == 81920:
        raise PrepareError(
            "length=81920 (80K input) is outside this subtask; "
            "only length=32768 is supported, so no 80K data was generated"
        )
    if args.length != SUPPORTED_LENGTH:
        raise PrepareError(
            f"unsupported --length {args.length}; this stage supports only "
            f"{SUPPORTED_LENGTH}"
        )
    if args.concurrency not in SUPPORTED_CONCURRENCY:
        raise PrepareError(
            f"unsupported --concurrency {args.concurrency}; choose 32, 64 or 128"
        )
    if args.output != SUPPORTED_OUTPUT:
        raise PrepareError(
            f"unsupported --output {args.output}; this stage requires 128"
        )

    length = args.length
    concurrency = args.concurrency
    requests = 2 * concurrency if args.requests is None else args.requests
    if requests < concurrency:
        raise PrepareError(
            f"--requests ({requests}) must be at least --concurrency ({concurrency})"
        )
    if requests % 8 != 0:
        raise PrepareError(
            f"--requests ({requests}) must be divisible by 8 for 8 DI ranks"
        )
    if requests > 256:
        raise PrepareError(
            f"--requests ({requests}) exceeds 256, the DI limit of 32 parents per rank"
        )

    cached_tokens = (
        (length * 9 // 10) // 128 * 128
        if args.cached_tokens is None
        else args.cached_tokens
    )
    if cached_tokens <= 0 or cached_tokens >= length:
        raise PrepareError(
            f"--cached-tokens must be between 1 and {length - 1}, got {cached_tokens}"
        )
    if cached_tokens % 128 != 0:
        raise PrepareError(
            f"--cached-tokens ({cached_tokens}) must be a multiple of 128"
        )

    v_warm_prefix = (
        cached_tokens + 128
        if args.v_warm_prefix is None
        else args.v_warm_prefix
    )
    if v_warm_prefix <= 0 or v_warm_prefix >= length:
        raise PrepareError(
            f"--v-warm-prefix must be between 1 and {length - 1}, "
            f"got {v_warm_prefix}"
        )

    return (
        length,
        concurrency,
        requests,
        args.output,
        cached_tokens,
        v_warm_prefix,
        args.seed,
    )


def resolve_out_dir(raw):
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = HOST_ROOT / candidate
    root = HOST_ROOT.resolve()
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        raise PrepareError(
            f"--out-dir must be a subdirectory of host project ROOT {root}: {resolved}"
        )
    if relative == Path(".") or not relative.parts:
        raise PrepareError("--out-dir must be below ROOT, not ROOT itself")
    return resolved, relative.as_posix()


def path_entry_exists(path):
    return path.exists() or path.is_symlink()


def build_config_specs(concurrency):
    specs = []
    for framework_tag, framework, port in FRAMEWORKS:
        for phase, namespace_group, max_out_len in PHASES:
            if phase == "SHAPE-W1":
                corpus = "shape-warm.jsonl"
            elif phase in ("NORMAL-R1", "HOT-R1"):
                corpus = "corpus.jsonl"
            else:
                corpus = (
                    "di-prefix.jsonl"
                    if framework == "deepinfer"
                    else "v-prefix.jsonl"
                )

            if framework == "deepinfer" and phase == "HOT-W1":
                cache_action = "write"
            elif framework == "deepinfer" and phase == "HOT-R1":
                cache_action = "read"
            else:
                cache_action = ""

            run_id = f"E27-{framework_tag}-L32K-C{concurrency}-{phase}"
            specs.append(
                {
                    "filename": run_id + ".py",
                    "run_id": run_id,
                    "framework": framework,
                    "framework_tag": framework_tag,
                    "phase": phase,
                    "corpus": corpus,
                    "output": max_out_len,
                    "cache_action": cache_action,
                    "cache_namespace": (
                        f"e27-{framework_tag.lower()}-l32k-c{concurrency}-"
                        f"{namespace_group}"
                    ),
                    "port": port,
                }
            )
    return specs


def ensure_targets_free(out_path, config_dir, specs):
    if path_entry_exists(out_path):
        raise PrepareError(f"refusing to overwrite existing out-dir: {out_path}")
    if path_entry_exists(config_dir):
        if config_dir.is_symlink() or not config_dir.is_dir():
            raise PrepareError(
                f"refusing to use non-directory config path: {config_dir}"
            )
    for spec in specs:
        config_path = config_dir / spec["filename"]
        if path_entry_exists(config_path):
            raise PrepareError(f"refusing to overwrite existing config: {config_path}")


def load_token_pool(source_path):
    digest = hashlib.sha256()
    candidates = set()
    rows = 0
    with source_path.open("rb") as source_file:
        for line_number, raw_line in enumerate(source_file, 1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise PrepareError(
                    f"invalid JSON in {source_path} at line {line_number}: {exc}"
                )
            token_ids = row.get("input_token_ids") if isinstance(row, dict) else None
            if not isinstance(token_ids, list) or not token_ids:
                raise PrepareError(
                    f"missing non-empty input_token_ids in {source_path} "
                    f"at line {line_number}"
                )
            for token_id in token_ids:
                if type(token_id) is not int or token_id < 0:
                    raise PrepareError(
                        f"invalid token ID {token_id!r} in {source_path} "
                        f"at line {line_number}"
                    )
                candidates.add(token_id)
            rows += 1
    if rows == 0:
        raise PrepareError(f"no workload rows found in {source_path}")
    if len(candidates) < 2:
        raise PrepareError(
            f"source token pool in {source_path} has fewer than two distinct IDs"
        )
    return sorted(candidates), digest.hexdigest(), rows


def row_for(request_index, token_ids):
    return {
        "request_index": request_index,
        "logical_id": f"E27-L32K-q{request_index}",
        "input_token_ids": token_ids,
    }


def write_json_line(file_handle, digest, obj):
    line = json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"
    file_handle.write(line)
    digest.update(line.encode("utf-8"))


def make_shape_warm_ids(request_index, corpus_ids, candidates, length, base_seed):
    rng = random.Random(base_seed + WARM_SEED_OFFSET + request_index)
    prefix_len = min(128, length)
    token_ids = rng.choices(candidates, k=prefix_len)
    for position in range(prefix_len):
        while token_ids[position] == corpus_ids[position]:
            token_ids[position] = rng.choice(candidates)
    token_ids.extend(rng.choices(candidates, k=length - prefix_len))
    return token_ids


def make_v_sentinel(request_index, forbidden, candidates, base_seed):
    rng = random.Random(base_seed + SENTINEL_SEED_OFFSET + request_index)
    sentinel = rng.choice(candidates)
    while sentinel == forbidden:
        sentinel = rng.choice(candidates)
    return sentinel


def generate_workloads(
    out_path,
    candidates,
    length,
    requests,
    cached_tokens,
    v_warm_prefix,
    seed,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_path.mkdir()
    except FileExistsError:
        raise PrepareError(f"refusing to overwrite existing out-dir: {out_path}")

    digests = {name: hashlib.sha256() for name in WORKLOAD_FILES}
    with contextlib.ExitStack() as stack:
        corpus_file = stack.enter_context(
            (out_path / "corpus.jsonl").open("x", encoding="utf-8", newline="\n")
        )
        shape_file = stack.enter_context(
            (out_path / "shape-warm.jsonl").open("x", encoding="utf-8", newline="\n")
        )
        di_file = stack.enter_context(
            (out_path / "di-prefix.jsonl").open("x", encoding="utf-8", newline="\n")
        )
        v_file = stack.enter_context(
            (out_path / "v-prefix.jsonl").open("x", encoding="utf-8", newline="\n")
        )
        questions_file = stack.enter_context(
            (out_path / "questions.jsonl").open("x", encoding="utf-8", newline="\n")
        )

        for request_index in range(requests):
            corpus_ids = random.Random(seed + request_index).choices(
                candidates, k=length
            )
            shape_ids = make_shape_warm_ids(
                request_index, corpus_ids, candidates, length, seed
            )
            sentinel = make_v_sentinel(
                request_index, corpus_ids[v_warm_prefix], candidates, seed
            )

            write_json_line(
                corpus_file,
                digests["corpus.jsonl"],
                row_for(request_index, corpus_ids),
            )
            write_json_line(
                shape_file,
                digests["shape-warm.jsonl"],
                row_for(request_index, shape_ids),
            )
            write_json_line(
                di_file,
                digests["di-prefix.jsonl"],
                row_for(request_index, corpus_ids[:cached_tokens]),
            )
            write_json_line(
                v_file,
                digests["v-prefix.jsonl"],
                row_for(
                    request_index,
                    corpus_ids[:v_warm_prefix] + [sentinel],
                ),
            )
            write_json_line(
                questions_file,
                digests["questions.jsonl"],
                {"question": str(request_index), "answer": ""},
            )
    return digests


def render_config(spec, out_relative, concurrency):
    def quoted(value):
        return json.dumps(value, ensure_ascii=True)

    questions_path = "/" + out_relative + "/questions.jsonl"
    corpus_path = "/" + out_relative + "/" + spec["corpus"]
    inference_url = f"http://127.0.0.1:{spec['port']}/v1/completions"
    abbr = f"E27-L32K-C{concurrency}"
    return f'''from e27_ais_model import E27MappedAPI
from ais_bench.benchmark.datasets import CustomDataset
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local import LocalRunner
from ais_bench.benchmark.tasks import OpenICLApiInferTask
root = {quoted(CONTAINER_ROOT)}
framework = {quoted(spec["framework"])}
run_id = {quoted(spec["run_id"])}
datasets = [dict(abbr={quoted(abbr)},type=CustomDataset,path=root+{quoted(questions_path)},
 reader_cfg=dict(input_columns=["question"],output_column="answer"),
 infer_cfg=dict(prompt_template=dict(type=PromptTemplate,template="{{question}}"),
 retriever=dict(type=ZeroRetriever),inferencer=dict(type=GenInferencer)),
 eval_cfg=dict(evaluator=dict(type=AccEvaluator),pred_role="BOT"),meta_path="")]
models = [dict(attr="service",type=E27MappedAPI,abbr=run_id,framework=framework,run_id=run_id,
 cache_namespace={quoted(spec["cache_namespace"])},
 cache_action={quoted(spec["cache_action"])},
 corpus=root+{quoted(corpus_path)},
 model={quoted(MODEL_NAME)},path={quoted(MODEL_PATH)},
 inference_url={quoted(inference_url)},
 stream=True,request_rate=0,retry=1,max_out_len={spec["output"]},batch_size={concurrency},
 generation_kwargs=dict(temperature=0.0,ignore_eos=True,seed=3000))]
infer = dict(partitioner=dict(type=NaivePartitioner),runner=dict(type=LocalRunner,max_num_workers=1,
 task=dict(type=OpenICLApiInferTask)))
work_dir = root+"/runs/e27-ais/"+run_id
'''


def write_configs(config_dir, specs, out_relative, concurrency):
    config_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for spec in specs:
        config_path = config_dir / spec["filename"]
        config_bytes = render_config(spec, out_relative, concurrency).encode("utf-8")
        with config_path.open("xb") as config_file:
            config_file.write(config_bytes)
        records.append(
            {
                "filename": spec["filename"],
                "location": f"configs/e27/{spec['filename']}",
                "host_path": str(config_path),
                "container_path": (
                    f"{CONTAINER_ROOT}/configs/e27/{spec['filename']}"
                ),
                "sha256": hashlib.sha256(config_bytes).hexdigest(),
                "run_id": spec["run_id"],
                "framework": spec["framework"],
                "framework_tag": spec["framework_tag"],
                "phase": spec["phase"],
                "corpus": spec["corpus"],
                "corpus_container_path": (
                    f"{CONTAINER_ROOT}/{out_relative}/{spec['corpus']}"
                ),
                "output": spec["output"],
                "cache_action": spec["cache_action"],
                "cache_namespace": spec["cache_namespace"],
            }
        )
    return records


def file_records(out_path, out_relative, digests, requests, lengths):
    records = {}
    for name in WORKLOAD_FILES:
        path = out_path / name
        record = {
            "host_path": str(path),
            "container_path": f"{CONTAINER_ROOT}/{out_relative}/{name}",
            "sha256": digests[name].hexdigest(),
            "bytes": path.stat().st_size,
            "lines": requests,
        }
        if name in lengths:
            record["tokens_per_line"] = lengths[name]
        records[name] = record
    return records


def build_manifest(
    out_path,
    out_relative,
    source_path,
    source_sha256,
    source_rows,
    candidate_count,
    length,
    concurrency,
    requests,
    output,
    cached_tokens,
    v_warm_prefix,
    seed,
    digests,
    config_records,
):
    lengths = {
        "corpus.jsonl": length,
        "shape-warm.jsonl": length,
        "di-prefix.jsonl": cached_tokens,
        "v-prefix.jsonl": v_warm_prefix + 1,
    }
    files = file_records(out_path, out_relative, digests, requests, lengths)
    fraction = cached_tokens / length
    percent = round(fraction * 100.0, 8)
    return {
        "schema": "e27-prepare-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "length": length,
        "concurrency": concurrency,
        "requests": requests,
        "output": output,
        "cached_tokens": cached_tokens,
        "v_warm_prefix": v_warm_prefix,
        "seed": seed,
        "seeds": {
            "corpus_per_request": "seed + request_index",
            "shape_warm_per_request": f"seed + {WARM_SEED_OFFSET} + request_index",
            "v_sentinel_per_request": (
                f"seed + {SENTINEL_SEED_OFFSET} + request_index"
            ),
        },
        "planned_cache": {
            "cached_tokens": cached_tokens,
            "input_tokens": length,
            "fraction": fraction,
            "percent": percent,
            "cache_hit_measured": False,
            "statement": (
                "cached_tokens is a planned prefix length only; "
                "no cache hit has been measured"
            ),
        },
        "root": {
            "host_path": str(HOST_ROOT),
            "container_path": CONTAINER_ROOT,
        },
        "out_dir": {
            "host_path": str(out_path),
            "relative_path": out_relative,
            "container_path": f"{CONTAINER_ROOT}/{out_relative}",
        },
        "source": {
            "relative_path": SOURCE_REL,
            "host_path": str(source_path),
            "container_path": f"{CONTAINER_ROOT}/{SOURCE_REL}",
            "sha256": source_sha256,
            "rows": source_rows,
            "candidate_token_count": candidate_count,
        },
        "source_sha256": source_sha256,
        "files": files,
        "configs": config_records,
        "sha256": {
            "source": source_sha256,
            "files": {name: record["sha256"] for name, record in files.items()},
            "configs": {
                record["filename"]: record["sha256"]
                for record in config_records
            },
        },
        "notes": [
            (
                "Synthetic frozen valid token-ID payloads are sampled from the "
                "sorted unique input_token_ids of E26-L32K-c16; payload quality "
                "has not been evaluated."
            ),
            (
                "shape-warm uses an independent RNG stream and forces each of "
                "the first 128 IDs to differ from the corresponding corpus ID."
            ),
            (
                "v-prefix contains the first v_warm_prefix corpus IDs followed "
                "by a sentinel that differs from corpus[v_warm_prefix]."
            ),
            (
                "Default K for length 32768 is 29440, i.e. 89.84375% of the "
                "input length as a planned prefix fraction, not a measured "
                "cache hit."
            ),
            (
                "All configs are static Python dictionaries; no environment "
                "variables are read."
            ),
        ],
    }


def main(argv=None):
    args = parse_args(argv)
    try:
        (
            length,
            concurrency,
            requests,
            output,
            cached_tokens,
            v_warm_prefix,
            seed,
        ) = validate_cli(args)
        out_path, out_relative = resolve_out_dir(args.out_dir)
        config_dir = HOST_ROOT / "configs" / "e27"
        specs = build_config_specs(concurrency)
        ensure_targets_free(out_path, config_dir, specs)

        source_path = HOST_ROOT / SOURCE_REL
        candidates, source_sha256, source_rows = load_token_pool(source_path)
        digests = generate_workloads(
            out_path,
            candidates,
            length,
            requests,
            cached_tokens,
            v_warm_prefix,
            seed,
        )
        config_records = write_configs(
            config_dir, specs, out_relative, concurrency
        )

        manifest = build_manifest(
            out_path=out_path,
            out_relative=out_relative,
            source_path=source_path,
            source_sha256=source_sha256,
            source_rows=source_rows,
            candidate_count=len(candidates),
            length=length,
            concurrency=concurrency,
            requests=requests,
            output=output,
            cached_tokens=cached_tokens,
            v_warm_prefix=v_warm_prefix,
            seed=seed,
            digests=digests,
            config_records=config_records,
        )
        manifest_path = out_path / "manifest.json"
        manifest_bytes = (
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        with manifest_path.open("xb") as manifest_file:
            manifest_file.write(manifest_bytes)
    except PrepareError as exc:
        print(f"e27_prepare: error: {exc}", file=sys.stderr)
        return 2
    except FileExistsError as exc:
        print(
            f"e27_prepare: error: refusing to overwrite existing path: "
            f"{exc.filename}",
            file=sys.stderr,
        )
        return 2

    summary = {
        "status": "ok",
        "out_dir": str(out_path),
        "manifest": str(manifest_path),
        "length": length,
        "concurrency": concurrency,
        "requests": requests,
        "output": output,
        "cached_tokens": cached_tokens,
        "v_warm_prefix": v_warm_prefix,
        "files": list(WORKLOAD_FILES),
        "configs": len(config_records),
        "source_sha256": source_sha256,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
