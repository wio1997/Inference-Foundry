"""Batch-invariant test in pytest form.

Run directly:
    python -m pytest tests/models/test_batch_invariant.py -s

Verifies that with DEEPINFER_BATCH_INVARIANT=1, running generate on the same prompt batched and singly
gives identical results. Models / parallel sizes are parameterized by CASES.
"""

from __future__ import annotations

import copy
import random
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
    use_nextn_decode: int = 0  # the backbone only by default; cases that need NextN coverage enable it explicitly


CASES = [
    Case(DEFAULT_MODEL_NAME, ep_size=8, use_nextn_decode=5),
]
NUM_SEQS = 64  # number of sequences generated per case

# ===== Test logic =====


def _get_result(output: list) -> tuple[list, list, list, list]:
    """Extract (token_ids, logprobs, top_token_ids, top_logprobs) from the RequestOutput list."""
    token_ids = [x.token_id for x in output]
    logprobs = [x.token_logprobs for x in output]
    top_token_ids = [x.top_token_ids for x in output]
    top_logprobs = [x.top_logprobs for x in output]
    return token_ids, logprobs, top_token_ids, top_logprobs


# ===== Subprocess entry =====


def _run_batch_invariant(case: Case, dp_size: int, dp_rank: int) -> None:
    """Verify batch invariance on one DP rank (called inside a subprocess by run_dp)."""
    llm = create_llm(
        case.model_name,
        dp_rank,
        ep_size=case.ep_size,
        env={
            "DEEPINFER_LOGGING_INTERVAL_STEP": "1000",
            "DEEPINFER_USE_NEXTN_DECODE": str(case.use_nextn_decode),
            "DEEPINFER_BATCH_INVARIANT": "1",
        },
        max_num_batched_tokens=16384,
        max_model_len=16384,
        max_num_seqs=NUM_SEQS,
        gpu_memory_utilization=0.9,
        lookahead_tokens=min(1, case.use_nextn_decode),
    )

    from deepinfer.sampling_params import SamplingParams

    random.seed(dp_rank)
    prompt_lens = [max(1, int(random.normalvariate(2500, 1000))) for _ in range(NUM_SEQS)]
    print(f"{dp_rank=}, {prompt_lens=}")
    # Engram input token ids must be within the original→compressed token mapping table.
    vocab_size = llm.llm_engine.model_config.get_vocab_size()
    if llm.llm_engine.model_config.use_compressed_tokenizer:
        from deepinfer.model_executor.models.v4_modules.engram import CompressedTokenizer

        vocab_size = len(CompressedTokenizer.lookup_table)

    # avoid special tokens
    special_tokens_dict = llm.llm_engine.tokenizer.added_tokens_decoder
    non_special_tokens = [token for token in range(vocab_size) if token not in special_tokens_dict]
    prompt_token_ids = [[random.choice(non_special_tokens) for _ in range(length)] for length in prompt_lens]

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.7,
        max_tokens=50,
        ignore_eos=True,
        top_logprobs=5,
        prompt_logprobs=False,
    )

    outputs = llm.generate(None, sampling_params, prompt_token_ids, use_tqdm=False)
    llm.llm_engine.log_stats = False

    for token_ids, output in zip(prompt_token_ids, outputs, strict=True):
        sp = copy.copy(sampling_params)
        sp.golden_token_ids = [x.token_id for x in output]
        output_single = llm.generate(None, sp, [token_ids], use_tqdm=False)[0]
        result_batched = _get_result(output)
        result_single = _get_result(output_single)
        assert result_batched == result_single, f"{dp_rank=}, {result_batched=}, {result_single=}"

    llm.close()


# ===== pytest entry =====


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.model_name)
def test_batch_invariant(case: Case) -> None:
    """pytest entry: verify batch invariance across all DP ranks."""
    run_model_test(partial(_run_batch_invariant, case), case.ep_size)
