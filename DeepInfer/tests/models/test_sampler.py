"""Verify Sampler correctness.

Method: KS test + randomized PIT.
  1. Run DeepInfer to generate rollout data (generated on the fly)
  2. Extract sum_smaller_probs and token_logprobs for each trajectory
  3. Randomized PIT → KS test → KS statistic quantiles
  4. KS test all quantiles against U(0,1) → p_value

Models / parallel sizes are parameterized by CASES.
"""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass
from functools import partial

import numpy as np
import pytest

from examples._config import DEFAULT_MODEL_NAME, create_llm, setup_common_env
from tests.models._runner import run_model_test

# ===== Parameter tables =====


@dataclass(frozen=True)
class Case:
    model_name: str
    ep_size: int


CASES = [Case(DEFAULT_MODEL_NAME, ep_size=8)]

NUM_TRAJS = 128
MAX_TOKENS_PER_TRAJ = 2048

SAMPLING_CONFIGS: list[tuple[str, float, float]] = [
    # (config_name, top_p, temperature)
    ("greedy", 1.0, 0.0),
    ("t1.0_p0.95", 0.95, 1.0),
    ("t8.0_p0.95", 0.95, 8.0),
    ("t1.0_p1.0", 1.0, 1.0),
]

# ===== KS test =====


def ks_survival(d: float, n: int) -> float:
    """P(D_n > d) via the Kolmogorov asymptotic formula plus Stephens' finite-sample correction."""
    if d <= 0:
        return 1.0
    if d >= 1:
        return 0.0
    lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * d
    s = 0.0
    for k in range(1, 200):
        term = (-1) ** (k - 1) * np.exp(-2 * k * k * lam * lam)
        s += term
        if abs(term) < 1e-30:
            break
    return max(0.0, min(1.0, 2 * s))


def ks_test_uniform(values: list[float]) -> tuple[float | None, float | None]:
    """Run a KS test of one set of values against U(0,1), returning (ks_stat, p_value)."""
    v = np.sort(np.asarray(values, dtype=np.float64))
    n = len(v)
    if n == 0:
        return None, None
    idx = np.arange(1, n + 1)
    ks = max(np.max(idx / n - v), np.max(v - (idx - 1) / n))
    return float(ks), float(ks_survival(ks, n))


def pit_ks(ssp: list[float], log_probs: list[float], seed: int = 42) -> tuple[float, float, int] | None:
    """Run randomized PIT + KS test on one trajectory. Returns (ks_stat, ks_quantile, n_tokens) or None."""
    n = len(ssp)
    if n < 1 or len(log_probs) != n:
        return None
    ssp_arr = np.array(ssp, dtype=np.float64)
    p_arr = np.exp(np.array(log_probs, dtype=np.float64))
    if not (np.all(np.isfinite(ssp_arr)) and np.all(np.isfinite(p_arr)) and np.all(p_arr > 0)):
        return None
    if (ssp_arr - p_arr < -0.01).sum() > 0:
        return None
    rng = np.random.RandomState(seed)
    u = np.clip(ssp_arr - p_arr * rng.uniform(0, 1, size=n), 0, 1)
    ks_stat, ks_quantile = ks_test_uniform(u.tolist())
    if ks_stat is None:
        return None
    return ks_stat, ks_quantile, n


def _fmt(pv: float | None) -> str:
    if pv is None:
        return "None"
    if pv < 0.001:
        return f"{pv:.2e}"
    return f"{pv:.4f}"


def _stable_seed(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little") & 0x7FFFFFFF


# ===== Subprocess entry =====


def _run_sampler_test(case: Case, dp_size: int, dp_rank: int) -> None:
    """Generate rollouts on one DP rank and check sampler correctness (called inside a subprocess by run_dp)."""
    os.environ["DEEPINFER_USE_NEXTN_DECODE"] = "5"
    setup_common_env(
        model_name=case.model_name,
        sampler_force_topk=0,
        set_override_experts=True,
        enable_eplb=False,
    )
    os.environ.setdefault("DEEPINFER_BATCH_INVARIANT", "1")
    use_nextn = os.getenv("DEEPINFER_USE_NEXTN_DECODE", "0")

    llm = create_llm(
        case.model_name,
        dp_rank,
        ep_size=case.ep_size,
        env={
            "DEEPINFER_COMPUTE_SUM_SMALLER_PROBS": "1",
            "DEEPINFER_LOGGING_INTERVAL_STEP": "500",
        },
        max_num_batched_tokens=32768,
        max_model_len=32768,
        max_num_seqs=NUM_TRAJS * len(SAMPLING_CONFIGS),
        gpu_memory_utilization=0.85,
        lookahead_tokens=min(1, int(use_nextn)),
    )

    from deepinfer.sampling_params import SamplingParams

    # Engram input token ids must be within the original→compressed token mapping table.
    vocab_size = llm.llm_engine.model_config.get_vocab_size()
    if llm.llm_engine.model_config.use_compressed_tokenizer:
        from deepinfer.model_executor.models.v4_modules.engram import CompressedTokenizer

        vocab_size = len(CompressedTokenizer.lookup_table)
    # avoid special tokens: the VL engine expands a randomly hit image placeholder into an image, which changes the prompt length
    special_tokens_dict = llm.llm_engine.tokenizer.added_tokens_decoder
    non_special_tokens = [token for token in range(vocab_size) if token not in special_tokens_dict]

    # concatenate the prompts and sampling params of all configs
    all_prompt_token_ids: list[list[int]] = []
    all_sps: list[SamplingParams] = []
    config_idx_of_prompt: list[int] = []

    for ci, (config_name, top_p, temperature) in enumerate(SAMPLING_CONFIGS):
        rng = random.Random(dp_rank * 100 + _stable_seed(config_name) % 1000)
        for _ in range(NUM_TRAJS):
            plen = max(1, int(rng.gauss(2500, 1000)))
            all_prompt_token_ids.append([rng.choice(non_special_tokens) for _ in range(plen)])
            all_sps.append(
                SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=MAX_TOKENS_PER_TRAJ,
                    ignore_eos=True,
                    prompt_logprobs=False,
                )
            )
            config_idx_of_prompt.append(ci)

    # one batched generate
    all_outputs = llm.generate(prompts=None, sampling_params=all_sps, prompt_token_ids=all_prompt_token_ids, use_tqdm=False)
    llm.close()

    # group by config and run the KS test per config
    errors: list[str] = []

    for ci, (config_name, _top_p, _temperature) in enumerate(SAMPLING_CONFIGS):
        all_quantiles: list[float] = []
        next1_quantiles: list[float] = []
        all_tokens = 0
        next1_tokens = 0
        skip_counts = {"no_ssp": 0, "short": 0, "invalid": 0}
        total_tokens = 0
        golden_steps = 0

        for i, output in enumerate(all_outputs):
            if config_idx_of_prompt[i] != ci:
                continue

            if not output:
                skip_counts["short"] += 1
                continue
            if all(o.sum_smaller_probs == 0.0 for o in output):
                skip_counts["no_ssp"] += 1
                continue

            all_ssp: list[float] = []
            all_lp: list[float] = []
            next1_ssp: list[float] = []
            next1_lp: list[float] = []

            for j, o in enumerate(output):
                total_tokens += 1
                all_ssp.append(o.sum_smaller_probs)
                all_lp.append(o.token_logprobs)
                if o.draft_tok >= 1:
                    golden_steps += 1
                    if j + 1 < len(output):
                        next1_ssp.append(output[j + 1].sum_smaller_probs)
                        next1_lp.append(output[j + 1].token_logprobs)

            request_id = output[0].request_id

            if all_ssp:
                result = pit_ks(all_ssp, all_lp, seed=_stable_seed(request_id + "_all"))
                if result is not None:
                    _, ks_quantile, n_tokens = result
                    all_quantiles.append(ks_quantile)
                    all_tokens += n_tokens

            if next1_ssp:
                result = pit_ks(next1_ssp, next1_lp, seed=_stable_seed(request_id + "_n1"))
                if result is not None:
                    _, ks_quantile, n_tokens = result
                    next1_quantiles.append(ks_quantile)
                    next1_tokens += n_tokens

        _, all_pv = ks_test_uniform(all_quantiles) if len(all_quantiles) >= 2 else (None, None)
        _, next1_pv = ks_test_uniform(next1_quantiles) if len(next1_quantiles) >= 2 else (None, None)

        avg_tokens_per_step = total_tokens / golden_steps if golden_steps > 0 else 0.0

        print(
            f"[test_sampler.py] [DP {dp_rank}] {config_name:>12s}: "
            f"all num_tokens={all_tokens:>5d} p={_fmt(all_pv)}  "
            f"next1 num_tokens={next1_tokens:>5d} p={_fmt(next1_pv)}  "
            f"avg_tokens_per_step={avg_tokens_per_step:.2f}"
        )

        if len(all_quantiles) == 0:
            errors.append(f"DP {dp_rank} {config_name}: no valid trajectories for all, skip={skip_counts}")
        elif all_pv is None:
            errors.append(f"DP {dp_rank} {config_name}: computing the all p_value failed")
        elif all_pv <= 1e-5:
            errors.append(f"DP {dp_rank} {config_name}: all p_value={all_pv} is too small; the sampler may have a problem")

    if errors:
        raise AssertionError("\n".join(errors))


# ===== pytest entry =====


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.model_name)
def test_sampler_correctness(case: Case) -> None:
    """pytest entry (nightly): verify sampler correctness across DP ranks."""
    run_model_test(partial(_run_sampler_test, case), case.ep_size, timeout=7200)
