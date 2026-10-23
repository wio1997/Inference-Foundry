#!/usr/bin/env python3
"""Prepare the authorized 80K/600 cases locally; never starts a service."""
import argparse
import contextlib
import hashlib
import json
import pathlib
import random

import e27_prepare as shared

ROOT = pathlib.Path(__file__).resolve().parents[1]
LENGTH, OUTPUT, CACHED = 81920, 600, 73728
SEED = 2026102380


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--concurrency", type=int, choices=[16, 32], required=True)
    args = ap.parse_args()
    c, n = args.concurrency, args.concurrency * 2
    out = ROOT / f"data/workloads/E29-L80K-C{c}"
    config_dir = ROOT / "configs/e29"
    phases = [("SHAPE-W1", "shape-warm.jsonl", OUTPUT, "", "shape"),
              ("NORMAL-R1", "corpus.jsonl", OUTPUT, "", "normal"),
              ("HOT-W1", None, 1, "write", "firstuse"),
              ("HOT-R1", "corpus.jsonl", OUTPUT, "read", "firstuse"),
              ("HOT-W3", None, 1, "write", "fixed-r2"),
              ("HOT-R2", "corpus.jsonl", OUTPUT, "read", "fixed-r2")]
    specs = []
    for tag, frame, port in shared.FRAMEWORKS:
        for phase, corpus, output, action, namespace in phases:
            rid = f"E29-{tag}-L80K-C{c}-{phase}"
            specs.append(dict(filename=rid + ".py", run_id=rid, phase=phase,
                              framework=frame, framework_tag=tag, port=port,
                              corpus=corpus or ("di-prefix.jsonl" if tag == "DI" else "v-prefix.jsonl"),
                              output=output, cache_action=action if tag == "DI" else "",
                              cache_namespace=f"e29-{tag.lower()}-l80k-c{c}-{namespace}"))
    shared.ensure_targets_free(out, config_dir, specs)
    pool, source_sha, source_rows = shared.load_token_pool(ROOT / shared.SOURCE_REL)
    if source_sha != "b1994e95663861b1a18308ea4894e916d2ad2e32b2250ed986b7a359474dec66":
        raise ValueError("Frozen source token pool changed")
    out.mkdir(parents=True)
    digests = {name: hashlib.sha256() for name in shared.WORKLOAD_FILES}
    with contextlib.ExitStack() as stack:
        files = {name: stack.enter_context((out / name).open("x"))
                 for name in shared.WORKLOAD_FILES}
        for q in range(n):
            ids = random.Random(SEED + q).choices(pool, k=LENGTH)
            warm = shared.make_shape_warm_ids(q, ids, pool, LENGTH, SEED)
            sentinel = shared.make_v_sentinel(q, ids[CACHED + 128], pool, SEED)
            payloads = {"corpus.jsonl": ids, "shape-warm.jsonl": warm,
                        "di-prefix.jsonl": ids[:CACHED],
                        "v-prefix.jsonl": ids[:CACHED + 128] + [sentinel]}
            for name, tokens in payloads.items():
                shared.write_json_line(files[name], digests[name],
                                       dict(request_index=q, logical_id=f"E29-L80K-q{q}",
                                            input_token_ids=tokens))
            shared.write_json_line(files["questions.jsonl"], digests["questions.jsonl"],
                                   dict(question=str(q), answer=""))
    config_dir.mkdir(parents=True, exist_ok=True)
    config_records = []
    for spec in specs:
        # Retain the already validated mapper and its explicit native cache
        # protocol; E29 changes only the workload, output cap and run paths.
        text = shared.render_config(spec, str(out.relative_to(ROOT)), c)
        text = text.replace(f'"E27-L32K-C{c}"', f'"E29-L80K-C{c}"')
        text = text.replace('/runs/e27-ais/', '/runs/e29-ais/')
        compile(text, spec["filename"], "exec")
        path = config_dir / spec["filename"]
        with path.open("x") as f:
            f.write(text)
        config_records.append(dict(path=str(path.relative_to(ROOT)),
                                   sha256=hashlib.sha256(text.encode()).hexdigest(), **spec))
    lengths = {"corpus.jsonl": LENGTH, "shape-warm.jsonl": LENGTH,
               "di-prefix.jsonl": CACHED, "v-prefix.jsonl": CACHED + 129}
    manifest = dict(experiment="E29", status="prepared locally; no measured cache hit or performance",
                    input_tokens=LENGTH, actual_output_target=OUTPUT, concurrency=c,
                    total_requests=n, planned_cached_tokens=CACHED,
                    planned_hit_fraction=CACHED / LENGTH, remaining_prefill_target=LENGTH-CACHED,
                    source=dict(path=shared.SOURCE_REL, sha256=source_sha, rows=source_rows),
                    seed=SEED, files={name: dict(sha256=digests[name].hexdigest(),
                        bytes=(out/name).stat().st_size, rows=n,
                        tokens_per_row=lengths.get(name)) for name in shared.WORKLOAD_FILES},
                    configs=config_records, effective_budget=8192, max_seqs_per_rank=16,
                    max_context=98304, formal_selection=dict(normal="R1", hot="fixedR2"),
                    checks_required=["Exact frozen input IDs and actual600 delivered IDs",
                        "Native8-rank cache deltas exactly73728 per request on aggregate",
                        "DI per-request cached73728 and actual remaining8192",
                        "Fresh prior completed HTTP prefix population in each R1/R2 namespace",
                        "Zero new DI DG-JIT metadata in fixedHOT-R2; graph recapture visibility recorded",
                        "No other NPU model running; all8 ranks idle before/after"])
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(concurrency=c, total_requests=n, configs=len(specs),
                         status="prepared only", cached_target=CACHED)))


if __name__ == "__main__":
    main()
