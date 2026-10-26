"""Proposal: build fixed local-tokenizer chat inputs; execute only in target container."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

SEED, INPUT_TOKENS, OUTPUT_TOKENS = 20261022, 512, 128
WORDS = "apple bird river stone cloud garden orange quiet window tree ocean flower book green silver morning pencil water field gentle forest bright music little yellow simple table mountain circle paper summer blue candle happy road".split()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_identity(p):
    h = hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return {"path": p.name, "bytes": p.stat().st_size, "sha256": h.hexdigest()}


def make_prompt(tokenizer, index):
    rng = random.Random(SEED + index)
    words = [rng.choice(WORDS) for _ in range(2048)]
    prefix = f"Synthetic reading passage {index:02d}. Read these neutral words: "
    suffix = ". Write one short paragraph about their general theme."
    build = lambda n: prefix + " ".join(words[:n]) + suffix
    def render(content):
        messages = [{"role": "user", "content": content}]
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
        if not isinstance(ids, list) or any(type(x) is not int for x in ids):
            raise ValueError("expected a single list of rendered chat token IDs")
        return ids
    lo, hi = 0, len(words)
    if len(render(build(lo))) > INPUT_TOKENS:
        raise ValueError("template and synthetic prefix exceed declared input length")
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(render(build(mid))) <= INPUT_TOKENS:
            lo = mid
        else:
            hi = mid - 1
    content = build(lo)
    for _ in range(32):
        count = len(render(content))
        if count == INPUT_TOKENS:
            break
        options = [(len(render(content + tail)), content + tail)
                   for tail in (" a", " x", ".", "!", "\n", " z", " 0", " blue")]
        options = [(n, text) for n, text in options if count < n <= INPUT_TOKENS]
        if not options:
            raise ValueError("cannot construct exact declared chat length without changing template")
        _, content = min(options, key=lambda item: (item[0], item[1]))
    ids = render(content)
    if len(ids) != INPUT_TOKENS:
        raise ValueError("bounded synthetic construction did not reach exact input length")
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": content}],
                                               tokenize=False, add_generation_prompt=True)
    return {"prompt": content, "rendered_prompt": rendered, "input_token_ids": ids,
            "rendered_chat_tokens": len(ids), "user_text_sha256": digest(content.encode()),
            "rendered_text_sha256": digest(rendered.encode()), "input_ids_sha256": digest(canonical(ids))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("output directory must be new")
    from transformers import AutoTokenizer, __version__ as transformers_version
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    template = tokenizer.get_chat_template()
    if not isinstance(template, str) or not template:
        raise ValueError("checkpoint must provide a default single-user chat template")
    names = [f"r{r}-{i}" for r in range(1, 4) for i in range(8)] + [f"warmup-{i}" for i in range(4)]
    rows = [{"id": name, **make_prompt(tokenizer, i)} for i, name in enumerate(names)]
    if len({row["prompt"] for row in rows}) != 28:
        raise ValueError("synthetic user prompts must be unique")
    refs = [file_identity(args.model_dir / n) for n in ["config.json", "tokenizer_config.json",
            "tokenizer.json", "special_tokens_map.json", "generation_config.json",
            "chat_template.jinja", "tokenizer.model", "vocab.json", "merges.txt"] if (args.model_dir / n).is_file()]
    request = lambda row: {"id": row["id"], "prompt": row["prompt"], "max_tokens": OUTPUT_TOKENS}
    metadata = {"schema_version": 1, "data_seed": SEED, "input_tokens": INPUT_TOKENS,
        "output_tokens": OUTPUT_TOKENS, "concurrency_points": [1, 4], "sampling": {"seed": SEED, "temperature": 0, "top_p": 1},
        "exact_output_tokens": OUTPUT_TOKENS, "ignore_eos": True, "prefix_cache_policy": "server radix cache disabled; no prefix reuse",
        "thinking_policy": "checkpoint default; no reasoning_effort/thinking/enable_thinking override",
        "transport": "one user message whose content is prompt; never re-wrap rendered_prompt as user content",
        "server_verification_required": "Every measured usage.prompt_tokens must equal 512 and completion_tokens 128; otherwise retain mismatch and do not call it the declared baseline.",
        "tokenizer": {"class": type(tokenizer).__module__ + "." + type(tokenizer).__name__, "transformers_version": transformers_version,
            "chat_template_sha256": digest(template.encode()), "model_files": refs},
        "generator": file_identity(Path(__file__)), "requests": [request(row) for row in rows[:24]],
        "warmup_requests": [request(row) for row in rows[24:]],
        "measurement_blocks": [{"id": f"round-{r}", "request_ids": [f"r{r}-{i}" for i in range(8)]} for r in range(1, 4)]}
    payload = b"".join(canonical(row) + b"\n" for row in rows)
    metadata["prompts_jsonl_sha256"] = digest(payload)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "prompts.jsonl").write_bytes(payload)
    data = json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2).encode() + b"\n"
    (args.output_dir / "workload.json").write_bytes(data)
    print(json.dumps({"status": "constructed_not_measured", "formal_prompts": 24, "warmup_prompts": 4,
        "all_rendered_input_tokens": INPUT_TOKENS, "workload_sha256": digest(data), "prompts_sha256": digest(payload)}))


if __name__ == "__main__":
    main()
