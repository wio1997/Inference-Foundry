"""DRAFT_PENDING_PRIMARY_REVIEW: one bounded L17 native C32 functional PD cohort.

Exactly 32 logical / 64 backend requests, two per each of 16 D DPs.  Input is
the unchanged 512-token SHA d6e4cf...; P output is 1 token and D output is
min=max=600 with ignore_eos.  This is a correctness/coverage guard, not a short
benchmark.  The old whole-16-token vLLM reference comparison is retained as a
literal field but is not a gate: a 600-token answer cannot equal that whole
16-token text.  A separately labelled first-16-token prefix annotation is
diagnostic only.
"""
import asyncio
import hashlib
import json
import struct
import time
from pathlib import Path

import aiohttp
from tokenizers import Tokenizer
from glm53_sglang_pd_gateway_v2 import bootstrap_room, route

RUN = Path("/run/glm53")
BASE = "glm53-sg-C32-L17-native-functional-v1-20261019"
TOKENIZER_SHA = "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"
INPUT_SHA = "d6e4cf16d46a7e29461cd4a7e19fa45725cb1b42ac3f76fda226260074db107e"


async def main():
    tokenizer_path = Path("/tokenizer.json")
    assert hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() == TOKENIZER_SHA
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    text = (
        "Inventory verification. The code maple means 17, cedar means 23, and river means 41. "
        "Read the complete record and explain how to check that the numbers add up. "
        "A reliable answer uses each stated number exactly once and shows the result.\n"
    ) * 40
    ids = tokenizer.encode(text, add_special_tokens=False).ids[:512]
    sha = hashlib.sha256(struct.pack("<" + "I" * len(ids), *ids)).hexdigest()
    assert sha == INPUT_SHA
    expected_16 = tokenizer.decode(
        [220, 99619, 11, 323, 14785, 3363, 220, 102340, 13, 4457, 279, 4583, 3255, 323, 10334, 1246],
        skip_special_tokens=False,
    )
    rows = []
    attempts = []
    report = {
        "project_id": "glm53",
        "framework": "sglang",
        "namespace": BASE,
        "functional_only": True,
        "qualification": "C32_FUNCTIONAL_COUNTS_AND_PD_LIFECYCLE_ONLY_NOT_PERFORMANCE",
        "offered_concurrency": 32,
        "D_DP_count": 16,
        "cases_per_D_DP": 2,
        "input_sha256_le_u32": sha,
        "input_tokens": 512,
        "P_output_tokens": 1,
        "D_output_tokens": 600,
        "D_min_tokens": 600,
        "ignore_eos": True,
        "cases": rows,
        "backend_attempts": attempts,
        "proposed_logical_pairs": 32,
        "proposed_backend_calls": 64,
        "proposed_output_tokens_if_all_pass": 19232,
        "baseline_accepted": False,
        "formal_requests": 0,
        "retries": 0,
        "vllm_short_reference_scope": (
            "The old optional whole-text 16-token vLLM reference comparison is retained "
            "literally in every row. A 600-token D answer is not expected to equal the whole "
            "16-token reference; the field is not a pass gate. A new prefix-16 annotation is "
            "diagnostic only and does not replace the whole-text field."
        ),
    }

    async def leg(session, body, url, role, rank):
        row = {
            "rid": body["rid"],
            "role": role,
            "global_dp": rank,
            "dispatch_monotonic_ns": time.monotonic_ns(),
            "retry": False,
        }
        attempts.append(row)
        try:
            async with session.post(url, json=body, headers={"X-Request-Id": body["rid"]}) as response:
                row["status"] = response.status
                raw = await response.text()
                row["raw_response"] = raw
                assert response.status == 200, (response.status, raw[-2000:])
                data = json.loads(raw)
                row["response"] = data
                maximum = 1 if role == "P" else 600
                assert data["usage"]["prompt_tokens"] == 512
                assert data["usage"]["completion_tokens"] == maximum
                choice = data["choices"][0]
                assert choice["finish_reason"] == "length"
                assert choice["text"]
                row["pass"] = True
                row["text_sha256"] = hashlib.sha256(choice["text"].encode()).hexdigest()
                reference = tokenizer.decode([220], skip_special_tokens=False) if role == "P" else expected_16
                row["vllm_short_reference_text_equal"] = choice["text"] == reference
                row["vllm_short_reference_text_scope"] = (
                    "Literal whole-text comparison to the old 16-token vLLM short reference; "
                    "a 600-token D output cannot equal the whole 16-token reference. Retained "
                    "as a raw comparison, not used as a pass gate."
                )
                row["vllm_short_reference_prefix16_text_equal"] = choice["text"].startswith(reference)
                row["vllm_short_reference_prefix16_scope"] = (
                    "Diagnostic first-16-token rendered-text prefix only; this is not "
                    "whole-text equality and is not a correctness gate."
                )
        except Exception as error:
            row["pass"] = False
            row["error"] = repr(error)
        finally:
            row["terminal_monotonic_ns"] = time.monotonic_ns()
            print("GLM53_SG_MATRIX_STEP=" + json.dumps(row), flush=True)
        return row

    try:
        async with aiohttp.ClientSession(
            trust_env=False,
            timeout=aiohttp.ClientTimeout(total=300),
            connector=aiohttp.TCPConnector(limit=0, force_close=True),
        ) as session:
            async def one(prefix):
                selected = route(prefix)
                rid = f"{BASE}-p{prefix:02d}-functional"
                room = bootstrap_room(rid, selected["P"]["global_dp"])
                assert room % 4 == selected["P"]["global_dp"]
                common = {
                    "model": "glm53-w8a8",
                    "prompt": ids,
                    "min_tokens": 1,
                    "ignore_eos": True,
                    "temperature": 0,
                    "seed": 20261017,
                    "cache_salt": f"{BASE}-p{prefix:02d}",
                    "stream": False,
                    "skip_special_tokens": False,
                    "return_token_ids": False,
                    "bootstrap_host": "172.16.12.34",
                    "bootstrap_port": 29898,
                    "bootstrap_room": room,
                    "return_cached_tokens_details": True,
                    "return_spec_tokens_details": True,
                }
                prefill = {
                    **common,
                    "rid": rid + ".P",
                    "routed_dp_rank": selected["P"]["global_dp"],
                    "max_tokens": 1,
                }
                decode = {
                    **common,
                    "rid": rid + ".D",
                    "routed_dp_rank": selected["D"]["global_dp"],
                    "disagg_prefill_dp_rank": selected["P"]["global_dp"],
                    "max_tokens": 600,
                    "min_tokens": 600,
                }
                prefill_row, decode_row = await asyncio.gather(
                    leg(session, prefill, "http://172.16.12.34:30500/v1/completions", "P", selected["P"]["global_dp"]),
                    leg(session, decode, "http://172.16.12.36:30600/v1/completions", "D", selected["D"]["global_dp"]),
                )
                row = {
                    "P_DP": prefix % 4,
                    "D_DP": prefix % 16,
                    "prefill": prefill_row,
                    "decode": decode_row,
                    "pass": prefill_row["pass"] and decode_row["pass"],
                }
                rows.append(row)
                (RUN / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
                assert row["pass"], "Stop after first failed logical pair; retain both native results"
            cohort = await asyncio.gather(*(one(prefix) for prefix in range(32)), return_exceptions=True)
            assert not any(isinstance(item, BaseException) for item in cohort), [
                repr(item) for item in cohort if isinstance(item, BaseException)
            ]
        report["pass"] = len(rows) == 32 and all(row["pass"] for row in rows)
        assert report["pass"] and len(attempts) == 64
        report["successful_output_tokens"] = sum(
            item["response"]["usage"]["completion_tokens"] for item in attempts if item["pass"]
        )
        assert report["successful_output_tokens"] == 19232
        report["D_text_hashes"] = sorted({row["decode"]["text_sha256"] for row in rows})
    finally:
        report["model_requests_attempted"] = len(attempts)
        report["successful_inferences"] = sum(bool(item.get("pass")) for item in attempts)
        report["actual_successful_output_tokens"] = sum(
            item.get("response", {}).get("usage", {}).get("completion_tokens", 0)
            for item in attempts
            if item.get("pass")
        )
        (RUN / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print("GLM53_SG_MATRIX_RESULT=" + json.dumps(report), flush=True)


asyncio.run(main())
