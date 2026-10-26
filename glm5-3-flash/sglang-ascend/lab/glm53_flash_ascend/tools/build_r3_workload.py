"""Local-tokenizer CPU construction of fixed R3 16K/1K inputs."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys

SEED, INPUT_TOKENS, OUTPUT_TOKENS = 20261022, 16384, 1024
TARGETS, TOLERANCE = (512, 8192, 15360), 64
WORDS = "apple bird river stone cloud garden orange quiet window tree ocean flower book green silver morning pencil water field gentle forest bright music little yellow simple table mountain circle paper summer blue candle happy road".split()
VALUES = "cobalt violet amber ivory bronze teal coral scarlet indigo crimson ochre lilac".split()
TAILS = (" a", " x", ".", "!", "\n", " z", " 0", " blue")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_identity(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def render(tokenizer, content, tokenize):
    result = tokenizer.apply_chat_template([{"role": "user", "content": content}],
        tokenize=tokenize, add_generation_prompt=True, **({"return_dict": False} if tokenize else {}))
    if tokenize and (not isinstance(result, list) or any(type(x) is not int for x in result)):
        raise ValueError("expected one list of rendered chat token IDs")
    if not tokenize and not isinstance(result, str):
        raise ValueError("expected one rendered chat string")
    return result


def encode(tokenizer, text):
    result = tokenizer(text, add_special_tokens=False, truncation=False, return_attention_mask=False)["input_ids"]
    if not isinstance(result, list) or any(type(x) is not int for x in result):
        raise ValueError("expected one list of tokenizer token IDs")
    return result


def bounded_search(count, maximum, limit):
    lo, hi = 0, maximum
    if count(0) > limit:
        raise ValueError("fixed prefix exceeds construction target")
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(mid) <= limit:
            lo = mid
        else:
            hi = mid - 1
    # Small local search also checks nearby BPE boundary effects.
    valid = [(count(n), n) for n in range(max(0, lo - 16), min(maximum, lo + 16) + 1)]
    valid = [(size, n) for size, n in valid if size <= limit]
    return max(valid, key=lambda value: (value[0], value[1]))[1]


def fact_positions(tokenizer, rendered, ids, facts):
    if encode(tokenizer, rendered) != ids:
        raise ValueError("rendered-string encoding differs from canonical chat token IDs")
    offsets, fallback = None, None
    try:
        encoded = tokenizer(rendered, add_special_tokens=False, truncation=False, return_attention_mask=False,
                            return_offsets_mapping=True)
        if encoded["input_ids"] != ids or len(encoded["offset_mapping"]) != len(ids):
            raise ValueError("offset mapping is not aligned with canonical chat token IDs")
        offsets = encoded["offset_mapping"]
    except (NotImplementedError, TypeError, ValueError) as error:
        fallback = type(error).__name__
    result = []
    for fact, target in zip(facts, TARGETS):
        marker = fact["statement"]
        if rendered.count(marker) != 1:
            raise ValueError("fact statement must occur exactly once in rendered chat")
        start, end = rendered.index(marker), rendered.index(marker) + len(marker)
        if offsets is not None:
            hit = [i for i, (left, right) in enumerate(offsets)
                   if right > left and left < end and right > start]
            if not hit:
                raise ValueError("fact has no mapped canonical token span")
            first, last = min(hit), max(hit) + 1
            method = "tokenizer.return_offsets_mapping"
        else:
            before, through = encode(tokenizer, rendered[:start]), encode(tokenizer, rendered[:end])
            # Accept only verified stable boundaries; never invent an error bound.
            if ids[:len(before)] != before or ids[:len(through)] != through:
                raise ValueError("offsets unavailable and exact prefix token boundaries are unstable")
            first, last = len(before), len(through)
            method = "exact_rendered_prefix_count_verified_against_full_ids"
        if not target - TOLERANCE <= first < last <= target + TOLERANCE + 1:
            raise ValueError("whole fact token span falls outside declared position window")
        result.append({**fact, "rendered_char_start": start, "rendered_char_end": end,
            "token_start": first, "token_end_exclusive": last, "target_start_token": target,
            "allowed_token_window_inclusive": [target - TOLERANCE, target + TOLERANCE],
            "position_method": method, "boundary_error_tokens": 0,
            "offset_fallback_reason": fallback,
            "statement_sha256": digest(marker.encode())})
    return result


def prompt_row(tokenizer, name, category, content, facts=None):
    ids, rendered = render(tokenizer, content, True), render(tokenizer, content, False)
    if facts is not None and len(ids) != INPUT_TOKENS:
        raise ValueError("long prompt is not exactly 16384 rendered chat tokens")
    positions = fact_positions(tokenizer, rendered, ids, facts) if facts is not None else []
    return {"id": name, "category": category, "prompt": content, "rendered_prompt": rendered,
        "input_token_ids": ids, "rendered_chat_tokens": len(ids),
        "user_text_sha256": digest(content.encode()), "rendered_text_sha256": digest(rendered.encode()),
        "input_ids_sha256": digest(canonical(ids)), "facts": positions,
        "expected_final": "; ".join(f["id"] + "=" + f["value"] for f in facts) if facts else "5",
        "expected_facts": {f["id"]: f["value"] for f in facts} if facts else {"arithmetic": "5"}}


def make_long_prompt(tokenizer, name, category, index):
    rng = random.Random(SEED + index)
    words = [rng.choice(WORDS) for _ in range(32000)]
    values = rng.sample(VALUES, 3)
    facts = [{"id": f"R3P{index:03d}_{part}", "value": value}
             for part, value in zip(("HEAD", "MIDDLE", "TAIL"), values)]
    for fact in facts:
        fact["statement"] = "[" + fact["id"] + "] = " + fact["value"] + "."
    content = (f"Synthetic reading passage {index:03d}. The named factual entries are exact. "
               "The other words are neutral filler. Read all entries and answer the final question.\n")
    cursor = 0
    for fact, target in zip(facts, TARGETS):
        def construct(n):
            filler = " ".join(words[cursor:cursor + n])
            return content + filler + "\n" + fact["statement"] + "\n"
        def position(n):
            rendered = render(tokenizer, construct(n), False)
            return len(encode(tokenizer, rendered[:rendered.index(fact["statement"])]))
        selected = bounded_search(position, len(words) - cursor, target)
        content = construct(selected)
        cursor += selected
    names = [fact["id"] for fact in facts]
    question = ("\nFinal question: Report the assigned words for fact IDs " + ", ".join(names)
                + ". Output one final line in this exact order and format: "
                + "; ".join(name + "=value" for name in names) + ". Copy each assigned word exactly.")
    def construct_tail(n, small=""):
        return content + " ".join(words[cursor:cursor + n]) + small + "\n" + question
    selected = bounded_search(lambda n: len(render(tokenizer, construct_tail(n), True)),
                              len(words) - cursor, INPUT_TOKENS)
    small = ""
    for _ in range(32):
        count = len(render(tokenizer, construct_tail(selected, small), True))
        if count == INPUT_TOKENS:
            break
        options = [(len(render(tokenizer, construct_tail(selected, small + tail), True)), tail)
                   for tail in TAILS]
        options = [(size, tail) for size, tail in options if count < size <= INPUT_TOKENS]
        if not options:
            raise ValueError("bounded filler/tail construction cannot reach exact 16384")
        small += min(options, key=lambda value: (value[0], value[1]))[1]
    return prompt_row(tokenizer, name, category, construct_tail(selected, small), facts)


def request(row, request_id=None, exact=True):
    return {"id": request_id or row["id"], "prompt_id": row["id"], "prompt": row["prompt"],
            "max_tokens": OUTPUT_TOKENS, "ignore_eos": exact}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("output directory must be new; no overwrite")
    from transformers import AutoTokenizer, __version__ as transformers_version
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    template = tokenizer.get_chat_template()
    if not isinstance(template, str) or not template:
        raise ValueError("checkpoint default single-user chat template is required")
    spec = [(f"r{r}-{i}", "performance") for r in (1, 2) for i in range(4)]
    spec += [(f"warmup-{i}", "performance_warmup") for i in range(4)]
    spec += [("quality-first", "quality")] + [(f"quality-concurrent-{i}", "quality") for i in range(4)]
    rows = [make_long_prompt(tokenizer, name, category, i) for i, (name, category) in enumerate(spec)]
    smoke = prompt_row(tokenizer, "quality-arith-smoke", "short_smoke",
                       "What is 2 + 3? Give only the final numeric answer.")
    if len({row["prompt"] for row in rows}) != 17:
        raise ValueError("the 17 constructed long user prompts must be distinct")
    payload = b"".join(canonical(row) + b"\n" for row in [*rows, smoke])
    refs = [file_identity(args.model_dir / name) for name in ("config.json", "tokenizer_config.json",
        "tokenizer.json", "special_tokens_map.json", "generation_config.json", "chat_template.jinja",
        "tokenizer.model", "vocab.json", "merges.txt") if (args.model_dir / name).is_file()]
    config = json.loads((args.model_dir / "config.json").read_text())
    config = config.get("text_config", config)
    common = {"schema_version": 1, "data_seed": SEED, "input_tokens": INPUT_TOKENS,
        "sampling": {"seed": SEED, "temperature": 0, "top_p": 1},
        "prefix_cache_policy": "server radix cache disabled; no prefix reuse",
        "thinking_policy": "checkpoint default; no reasoning_effort/thinking/enable_thinking override",
        "transport": "one user message containing prompt; never re-wrap rendered_prompt as user content",
        "construction": "own synthetic filler only; bounded search/tails; no request-time truncation",
        "position_contract": "whole fact span inside target +/-64; offsets preferred; fallback only with exact canonical prefix token boundaries, otherwise fail",
        "tokenizer": {"class": type(tokenizer).__module__ + "." + type(tokenizer).__name__,
            "transformers_version": transformers_version,
            "tokenizers_version": importlib.metadata.version("tokenizers"), "python_version": sys.version,
            "chat_template_sha256": digest(template.encode()), "model_files": refs,
            "declared_checkpoint_dtype": config.get("dtype", config.get("torch_dtype")),
            "dtype_scope": "checkpoint metadata only; no model tensors/weights or NPU executed"},
        "generator": file_identity(Path(__file__)), "prompts_jsonl_sha256": digest(payload)}
    perf = {**common, "kind": "benchmark", "output_tokens": OUTPUT_TOKENS,
        "exact_output_tokens": OUTPUT_TOKENS, "ignore_eos": True, "concurrency_points": [1, 4],
        "requests": [request(row) for row in rows[:8]],
        "warmup_requests": [request(row) for row in rows[8:12]],
        "measurement_blocks": [{"id": f"round-{r}", "request_ids": [f"r{r}-{i}" for i in range(4)]}
                               for r in (1, 2)],
        "reuse_contract": "identical eight prompt IDs/payloads for A/B and C1/C4; 32 formal requests total",
        "server_verification_required": "Each formal usage must be 16384 input/1024 output and length finish; retain every mismatch."}
    quality_rows = rows[12:]
    by_prompt = {row["id"]: row for row in quality_rows}
    quality_requests = [request(row, exact=False) for row in quality_rows]
    quality_requests.append(request(quality_rows[0], "quality-first-repeat", exact=False))
    quality = {**common, "kind": "diagnostic", "output_tokens_cap": OUTPUT_TOKENS,
        "exact_output_tokens": None, "ignore_eos": False, "requests": quality_requests,
        "execution_stages": [{"id": "serial-first", "concurrency": 1, "request_ids": ["quality-first"]},
            {"id": "four-different", "concurrency": 4, "request_ids": [row["id"] for row in quality_rows[1:]]},
            {"id": "serial-repeat", "concurrency": 1, "request_ids": ["quality-first-repeat"]}],
        "quality_reference": [{"request_id": item["id"], "prompt_id": item["prompt_id"],
            "expected_final": by_prompt[item["prompt_id"]]["expected_final"],
            "expected_facts": by_prompt[item["prompt_id"]]["expected_facts"]} for item in quality_requests],
        "short_smoke_request": request(smoke, exact=False), "short_smoke_expected_final": "5",
        "server_verification_required": "Six long requests need input=16384; natural stop required for final-answer qualification. Short smoke is separate and not 16K. Length finish without final answer is not a quality pass.",
        "quality_scope": "head/middle/tail retrieval and limited repeat/concurrency check; no full accuracy/state certification"}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "prompts.jsonl").write_bytes(payload)
    hashes = {}
    for name, value in (("workload.json", perf), ("quality-workload.json", quality)):
        data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode() + b"\n"
        (args.output_dir / name).write_bytes(data)
        hashes[name] = digest(data)
    print(json.dumps({"status": "constructed_not_measured", "long_prompts": 17,
        "formal_prompts": 8, "performance_warmup_prompts": 4, "quality_unique_prompts": 5,
        "quality_long_requests": 6, "short_smoke_requests": 1, "input_tokens_long": INPUT_TOKENS,
        "performance_output_tokens": OUTPUT_TOKENS, "file_sha256": hashes,
        "prompts_jsonl_sha256": digest(payload)}))


if __name__ == "__main__":
    main()
