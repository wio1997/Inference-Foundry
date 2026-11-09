"""KL test in pytest form.

Run directly:
    python -m pytest tests/models/test_kl.py -s

Uses freshly generated image+text input to compare step-by-step decode logprobs against a full prefill.
Models / parallel sizes are parameterized by CASES.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import pytest

from examples._config import DEFAULT_MODEL_NAME, create_llm
from tests.models._runner import run_model_test

# ===== Parameter tables =====


@dataclass(frozen=True)
class Case:
    model_name: str
    ep_size: int
    cp_size: int = 1
    use_nextn_decode: int = 0
    sampler_force_topk: int = 0
    max_kl: float = 0.0  # upper bound on the KL between the two logprobs


CASES = [
    Case(DEFAULT_MODEL_NAME, ep_size=8, cp_size=2, use_nextn_decode=5),
]

# VL prompt construction parameters (matching benchmark.py's --prompt_mode image)
VL_NUM_PROMPTS = 3
VL_PROMPT_LENGTH = 16384  # total length after image tokens are expanded
VL_OUTPUT_LEN = 1024
VL_PROMPT_RNG_SEED = 20260723

# ===== Test logic =====


def _filter_logp(logp: float) -> float:
    return -9999.0 if logp == float("-inf") else logp


def _build_vl_prompts(llm: object, dp_rank: int) -> tuple[list[list[int]], list[list[int]], list[dict]]:
    """Random image+text prompt (same as benchmark.py's --prompt_mode image).

    Each image occupies one placeholder in token_ids plus a random best-shape BMP, which the engine expands;
    text and golden output tokens are sampled from the non-special token pool (VLPromptParams.from_llm's
    text_token_pool already excludes the special tokens in added_tokens_decoder, since special tokens
    disturb engine behavior, e.g. eos / image special segments). Each rank uses a different seed to cover more combinations.

    Returns (input_tokens_list, golden_tokens_list, multi_modal_data_list).
    """
    import random

    import torch

    # _prompts imports deepinfer at module level, so it must be loaded after create_llm (which does the device and env setup inside)
    from examples._prompts import VLPromptParams, generate_random_image_prompt

    vl_params = VLPromptParams.from_llm(llm)
    rng = random.Random(VL_PROMPT_RNG_SEED + dp_rank)
    input_tokens_list: list[list[int]] = []
    golden_tokens_list: list[list[int]] = []
    multi_modal_data_list: list[dict] = []
    for i in range(VL_NUM_PROMPTS):
        seed = VL_PROMPT_RNG_SEED + dp_rank * VL_NUM_PROMPTS + i
        prompt_token_ids, mm_data = generate_random_image_prompt(
            vl_params,
            VL_PROMPT_LENGTH,
            image_size=(rng.randint(256, 1024), rng.randint(256, 1024)),
            seed=seed,
        )
        pool = vl_params.text_token_pool
        gen = torch.Generator().manual_seed(seed + 233)  # use different seeds from prompt
        golden_ids = pool[torch.randint(0, pool.shape[0], (VL_OUTPUT_LEN,), generator=gen)].tolist()
        input_tokens_list.append(prompt_token_ids)
        golden_tokens_list.append(golden_ids)
        multi_modal_data_list.append(mm_data)
        print(
            f"vl prompt {i}: {len(prompt_token_ids)} tokens (containing {len(mm_data['image_bytes'])} image placeholders), golden={len(golden_ids)}"
        )
    return input_tokens_list, golden_tokens_list, multi_modal_data_list


# ===== Subprocess entry =====


def _run_kl(case: Case, dp_size: int, dp_rank: int) -> None:
    """Run two logprob passes on one DP rank and compare KL (called inside a subprocess by run_dp)."""
    llm = create_llm(
        case.model_name,
        dp_rank,
        ep_size=case.ep_size,
        cp_size=case.cp_size,
        env={
            "DEEPINFER_LOGGING_INTERVAL_STEP": "1000",
            "DEEPINFER_EXPERT_COUNT_WINDOW_SIZE": "3",
            "DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL": "1000",
            "DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS": "32",
            "DEEPINFER_BATCH_INVARIANT": "1",
            "DEEPINFER_SAMPLER_FORCE_TOPK": str(case.sampler_force_topk),
            "DEEPINFER_USE_NEXTN_DECODE": str(case.use_nextn_decode),
        },
        max_num_batched_tokens=32768,
        max_model_len=32768,
        max_num_seqs=16,
        gpu_memory_utilization=0.8,
        lookahead_tokens=min(1, case.use_nextn_decode),
    )

    import torch

    from deepinfer.sampling_params import SamplingParams
    from deepinfer.testing import cal_diff

    # VL prompt construction depends on the engine's image_token_id / processor parameters, so it must come after create_llm
    input_tokens, output_tokens, multi_modal_data = _build_vl_prompts(llm, dp_rank)
    num_prompts = len(input_tokens)

    prompt_token_ids = []
    sps = []

    for token_ids, golden_ids in zip(input_tokens, output_tokens, strict=True):
        prompt_token_ids.append(token_ids)
        sps.append(
            SamplingParams(
                temperature=0.5,
                top_p=0.95,
                max_tokens=len(golden_ids),
                ignore_eos=True,
                top_logprobs=5,
                prompt_logprobs=True,
                golden_token_ids=golden_ids,
            )
        )

    # the engine consumes (pops) multi_modal_data fields in place; pass a shallow copy so the original survives for the second pass
    outputs = llm.generate(
        prompts=None,
        sampling_params=sps,
        prompt_token_ids=prompt_token_ids,
        multi_modal_data=[dict(mm) for mm in multi_modal_data],
        use_tqdm=False,
    )

    # modify the first pass's prompt/sp in place: fold the first pass's generated tokens into the prompt, generate just 1 more token,
    # and recompute the full-sequence logprob from the second pass's prompt_logprobs
    prompt_token_ids_re = []
    sps_re = []
    for token_ids, sp, output in zip(prompt_token_ids, sps, outputs, strict=True):
        for o in output:
            token_ids.append(o.token_id)
        prompt_token_ids_re.append(token_ids)
        sp.max_tokens = 1
        sp.golden_token_ids = None
        sps_re.append(sp)

    # the second prompt only appends the first pass's generated text tokens at the end, so image placeholder positions are unchanged
    # and the engine expands them identically; reuse the same multi_modal_data
    outputs_re = llm.generate(
        prompts=None,
        sampling_params=sps_re,
        prompt_token_ids=prompt_token_ids_re,
        multi_modal_data=[dict(mm) for mm in multi_modal_data],
        use_tqdm=False,
    )

    kls = []

    for i in range(num_prompts):
        output = outputs[i]
        output_re = outputs_re[i]
        # in VL mode the engine expands image placeholders into image token segments; the actual prompt length
        # comes from the first pass's prompt_logprobs (identical to len(token_ids) in text-only mode)
        prompt_len = len(output[0].prompt_logprobs)

        logprobs = [_filter_logp(lp) for lp in output[0].prompt_logprobs]
        for o in output:
            logprobs.append(_filter_logp(o.token_logprobs))
        prompt_logprobs = logprobs[:prompt_len]
        output_logprobs = logprobs[prompt_len:]

        logprobs_re = [_filter_logp(lp) for lp in output_re[0].prompt_logprobs]
        prompt_logprobs_re = logprobs_re[:prompt_len]
        output_logprobs_re = logprobs_re[prompt_len:]

        assert prompt_logprobs == prompt_logprobs_re, (
            f"{prompt_logprobs[-10:], prompt_logprobs_re[-10:], len(prompt_logprobs), len(prompt_logprobs_re)=}"
        )
        cos_diff, rmse, chebyshev_dis, kl = cal_diff(
            torch.tensor(output_logprobs, dtype=torch.float32),
            torch.tensor(output_logprobs_re, dtype=torch.float32),
        )
        # print(f"DP {dp_rank}, {i}: {cos_diff=}, {rmse=}, {chebyshev_dis=}, {kl=}")
        kls.append(kl)

    llm.close()

    assert max(kls) <= case.max_kl, f"DP {dp_rank}, kls={kls}"


# ===== pytest entry =====


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.model_name)
def test_kl(case: Case) -> None:
    """pytest entry: run the logprob KL comparison across all DP ranks."""
    assert case.ep_size % case.cp_size == 0, f"EP({case.ep_size}) must be divisible by CP({case.cp_size})"
    assert case.max_kl == 0 or case.sampler_force_topk == 0, (
        "when max_kl is non-zero, DEEPINFER_SAMPLER_FORCE_TOPK must be off, otherwise KL can become inf"
    )

    run_model_test(partial(_run_kl, case), case.ep_size // case.cp_size)
