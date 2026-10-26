"""
Dataset generator for prefix cache performance testing.

Uses tokenizer vocabulary random sampling to generate test data,
eliminating GSM8K dependency and prefix collision issues across rounds.

Core approach (ported from token_counter.py):
  - HuggingFaceTokenizer.get_some_tokens() pattern
  - Local RNG (random.Random(seed)) per generation call
  - Sorted safe token IDs for cross-environment determinism
  - Seed offset per row/phase to avoid inter-round prefix collision
"""

import json
import logging
import os
import random
from typing import List, Optional, Set

from transformers import AutoTokenizer

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenizer wrapper with random token generation
# ---------------------------------------------------------------------------

def exact_token_length(tokenizer, text, target):
    """Calibrate decoded text, then verify actual raw prompt token count."""
    for _ in range(32):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == target:
            return text
        if len(ids) > target:
            text = tokenizer.decode(ids[:target], skip_special_tokens=True)
        else:
            text += " a" * (target - len(ids))
    raise ValueError(f"Cannot calibrate prompt to {target} tokens")

class TokenizerWrapper:
    """Tokenizer wrapper that supports deterministic random token generation.

    Uses a sorted safe-token-ID list and local RNG instances to ensure:
    - Same seed always produces the same text (reproducibility)
    - No global random state pollution
    - Cross-environment determinism (sorted IDs)
    """

    def __init__(self, tokenizer_path: str) -> None:
        try:
            self._tok = AutoTokenizer.from_pretrained(
                tokenizer_path, trust_remote_code=True
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load tokenizer from {tokenizer_path!r}: {exc}"
            ) from exc

        self._safe_token_ids: Optional[List[int]] = None

    def count_tokens(self, text: str, include_special: bool = True) -> int:
        """Count the number of tokens in *text*."""
        if not text:
            return 0
        return len(self._tok.encode(text, add_special_tokens=include_special))

    def get_some_tokens(self, num_tokens: int, seed: Optional[int] = None) -> str:
        """Generate random token text with the specified number of tokens.

        Args:
            num_tokens: Target number of tokens to generate.
            seed: Random seed. If fixed (non-zero), the generated text is reproducible.
                  If 0 or None, generation is truly random (uses system entropy, non-reproducible).

        Returns:
            A string whose token count equals *num_tokens* (or close to it).
        """
        if num_tokens <= 0:
            return ""

        # Local RNG — does NOT affect global random state
        # seed=0 means pure random (non-reproducible, using system entropy)
        if seed is None or seed == 0:
            rng = random.Random()
        else:
            rng = random.Random(seed)

        safe_ids = self._get_or_build_safe_ids()

        # 1. Randomly select token IDs from the safe vocabulary
        selected = rng.choices(safe_ids, k=num_tokens)

        # 2. Decode to text
        text = self._tok.decode(selected, skip_special_tokens=True)

        # 3. Length calibration: some tokenizers merge tokens during decode
        encoded = self._tok.encode(text, add_special_tokens=False)
        if len(encoded) > num_tokens:
            text = self._tok.decode(encoded[:num_tokens], skip_special_tokens=True)

        return exact_token_length(self._tok, text, num_tokens)

    # ---- internal helpers ----

    def _get_or_build_safe_ids(self) -> List[int]:
        """Return sorted safe token IDs (lazy-built, then cached)."""
        if self._safe_token_ids is None:
            ids = self._build_safe_token_ids()
            ids.sort()  # Sort for cross-environment determinism
            self._safe_token_ids = ids
        return self._safe_token_ids

    def _build_safe_token_ids(self) -> List[int]:
        """Filter out special tokens and empty/whitespace-only tokens."""
        vocab = self._tok.get_vocab()
        all_ids: Set[int] = set(vocab.values())
        special_ids: Set[int] = set(self._tok.all_special_ids)

        safe_ids_set = all_ids - special_ids

        safe_ids: List[int] = []
        for tid in safe_ids_set:
            try:
                decoded = self._tok.decode([tid], skip_special_tokens=True).strip()
                if decoded:  # skip empty / control-char tokens
                    safe_ids.append(tid)
            except Exception:
                continue

        # Fallback: if filtering removed everything, use the full safe set
        return safe_ids if safe_ids else list(safe_ids_set)


# ---------------------------------------------------------------------------
# Length distribution helpers
# ---------------------------------------------------------------------------

def sample_target_length(
    rng: random.Random,
    fixed_length: int,
    length_mean: Optional[int] = None,
    length_std: Optional[float] = None,
    length_min: Optional[int] = None,
    length_max: Optional[int] = None,
) -> int:
    """Sample a target prompt length from Gaussian or uniform distribution."""
    fixed_length = max(1, int(fixed_length))

    has_gauss = (length_mean is not None) and (length_std is not None)
    has_range = (length_min is not None) and (length_max is not None)

    lo = 1 if length_min is None else max(1, int(length_min))
    hi = None if length_max is None else max(1, int(length_max))
    if hi is not None and lo > hi:
        lo, hi = hi, lo

    if has_gauss:
        mu = max(1, int(length_mean))
        sigma = max(0.0, float(length_std))
        val = mu if sigma == 0 else int(round(rng.gauss(mu, sigma)))
        if hi is not None:
            val = min(val, hi)
        val = max(lo, val)
        return max(1, val)

    if has_range:
        return rng.randint(lo, hi)

    return fixed_length


def build_length_tag(
    input_len: int,
    length_mean: Optional[int],
    length_std: Optional[float],
    length_min: Optional[int],
    length_max: Optional[int],
) -> str:
    """Build a short tag string describing the length distribution."""
    if (length_mean is not None) and (length_std is not None):
        tag = f"G{int(length_mean)}_{str(length_std).replace('.', 'd')}"
        if (length_min is not None) and (length_max is not None):
            tag += f"_C{int(length_min)}_{int(length_max)}"
        return tag
    if (length_min is not None) and (length_max is not None):
        return f"U{int(length_min)}_{int(length_max)}"
    return f"L{int(input_len)}"


# ---------------------------------------------------------------------------
# Truncate / pad helper
# ---------------------------------------------------------------------------

def truncate_or_pad_text(tokenizer, text: str, target_len: int) -> str:
    """Adjust *text* to have exactly *target_len* tokens (truncate or repeat-pad)."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) >= target_len:
        tokens = tokens[:target_len]
    else:
        repeat_times = (target_len + len(tokens) - 1) // len(tokens)
        tokens = (tokens * repeat_times)[:target_len]
    return tokenizer.decode(tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Parse repeat_rate string
# ---------------------------------------------------------------------------

def parse_prefix_ratio(r: str) -> float:
    """Parse a repeat-rate string: '50%' -> 0.5, '0.5' -> 0.5."""
    r = str(r).strip()
    if r.endswith("%"):
        v = float(r[:-1]) / 100.0
    else:
        v = float(r)
    if not (0.0 <= v <= 1.0):
        raise ValueError("repeat_rate must be in [0,1] or percent [0%,100%]")
    return v


# ---------------------------------------------------------------------------
# Write dataset to JSONL
# ---------------------------------------------------------------------------

def write_jsonl(path: str, dataset: list, num: Optional[int] = None) -> None:
    """Write dataset entries to a JSONL file in GSM8K-compatible format.

    Each line: {"question": <text>, "answer": "none"}
    """
    if num is not None:
        if len(dataset) < num:
            repeats = num // len(dataset)
            remainder = num % len(dataset)
            dataset = dataset * repeats + dataset[:remainder]
        else:
            dataset = dataset[:num]

    with open(path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps({"question": item, "answer": "none"}, ensure_ascii=False))
            f.write("\n")


# ---------------------------------------------------------------------------
# Main dataset creation function
# ---------------------------------------------------------------------------

def create_prefix_dataset(
    tokenizer_path: str,
    input_len: int,
    number: int,
    save_path: str,
    dp: int,
    repeat_rate: float,
    seed: int,
    prefix_num: int,
    length_mean: Optional[int] = None,
    length_std: Optional[float] = None,
    length_min: Optional[int] = None,
    length_max: Optional[int] = None,
) -> tuple:
    """Create a prefix-cache test dataset using random token generation.

    Generates two JSONL files:
      - prefix file: contains each prefix repeated *dp* times (for warmup)
      - full dataset file: prefix + 3 separator tokens + suffix per row

    Args:
        tokenizer_path: HuggingFace tokenizer path or model name.
        input_len: Base input token length.
        number: Number of dataset entries (rows).
        save_path: Directory to save generated JSONL files.
        dp: Data-parallelism degree — each prefix is replicated *dp* times.
        repeat_rate: Fraction of input_len that is the shared prefix (0.0–1.0).
        seed: Random seed controlling all token generation.
        prefix_num: Number of distinct prefixes.
        length_mean / length_std / length_min / length_max:
            Optional parameters for variable-length distribution.

    Returns:
        (prefix_jsonl_path, dataset_jsonl_path) — absolute paths to the two files.
    """
    base_name = os.path.basename(os.path.normpath(tokenizer_path))
    use_variable = (
        (length_mean is not None and length_std is not None)
        or (length_min is not None and length_max is not None)
    )

    tok_wrapper = TokenizerWrapper(tokenizer_path)
    tokenizer = tok_wrapper._tok  # reuse the loaded tokenizer for truncation

    if use_variable:
        return _create_prefix_dataset_variable(
            tok_wrapper, tokenizer, input_len, number, save_path, base_name,
            dp, repeat_rate, seed, prefix_num,
            length_mean, length_std, length_min, length_max,
        )

    # ========== Fixed-length prefix dataset ==========

    prefix_len = int(input_len * repeat_rate)
    separator_len = 3  # unique separator tokens between prefix and suffix
    suffix_len = int(input_len - prefix_len - separator_len)

    # ---- Generate prefix pool ----
    prefix_pool = []
    pbar = tqdm(total=prefix_num, desc="Generating prefixes", unit="row") if tqdm else None
    for i in range(prefix_num):
        prefix_seed = 0 if seed == 0 else seed + i  # seed=0: pure random; else: deterministic offset
        prefix_text = tok_wrapper.get_some_tokens(prefix_len, seed=prefix_seed)
        prefix_pool.append(prefix_text)
        if pbar:
            pbar.update(1)
    if pbar:
        pbar.close()

    # ---- Write prefix warmup file (each prefix repeated dp times) ----
    prefix_dataset = []
    for i in range(prefix_num):
        for _ in range(dp):
            prefix_dataset.append(prefix_pool[i])

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    prefix_path = os.path.join(
        save_path, f"prefix-GSM8K-in{prefix_len}-num{dp * prefix_num}-{base_name}.jsonl"
    )
    write_jsonl(prefix_path, prefix_dataset, dp * prefix_num)
    logger.info(f"Prefix warmup file: {prefix_path} ({dp * prefix_num} rows)")

    if repeat_rate >= 1.0:
        # Entire input is prefix — dataset = prefix file
        dataset_path = os.path.join(
            save_path, f"GSM8K-in{prefix_len}-num{number}-{base_name}-repeatRate{repeat_rate}.jsonl"
        )
        write_jsonl(dataset_path, prefix_dataset, number)
        return prefix_path, dataset_path

    # ---- Generate separator tokens (3 per row, unique) ----
    separator_pool = []
    pbar = tqdm(total=number, desc="Generating separators", unit="row") if tqdm else None
    for i in range(number):
        sep_seed = 0 if seed == 0 else seed + prefix_num + i  # seed=0: pure random
        sep_text = tok_wrapper.get_some_tokens(separator_len, seed=sep_seed)
        separator_pool.append(sep_text)
        if pbar:
            pbar.update(1)
    if pbar:
        pbar.close()

    # ---- Generate suffix pool ----
    suffix_pool = []
    pbar = tqdm(total=number, desc="Generating suffixes", unit="row") if tqdm else None
    for i in range(number):
        suffix_seed = 0 if seed == 0 else seed + prefix_num + number + i  # seed=0: pure random
        suffix_text = tok_wrapper.get_some_tokens(suffix_len, seed=suffix_seed)
        suffix_pool.append(suffix_text)
        if pbar:
            pbar.update(1)
    if pbar:
        pbar.close()

    # ---- Stitch full dataset: prefix + separator + suffix ----
    dataset = []
    pbar = tqdm(total=number, desc="Stitching dataset", unit="row") if tqdm else None
    for i in range(number):
        entry = prefix_pool[i % prefix_num] + separator_pool[i] + suffix_pool[i]
        entry = exact_token_length(tokenizer, entry, input_len)
        dataset.append(entry)
        if pbar:
            pbar.update(1)
    if pbar:
        pbar.close()

    dataset_path = os.path.join(
        save_path,
        f"GSM8K-in{input_len}-num{number}-{base_name}-repeatRate{repeat_rate}.jsonl",
    )
    write_jsonl(dataset_path, dataset, number)
    logger.info(f"Full dataset file: {dataset_path} ({number} rows)")

    return prefix_path, dataset_path


# ---------------------------------------------------------------------------
# Variable-length prefix dataset
# ---------------------------------------------------------------------------

def _create_prefix_dataset_variable(
    tok_wrapper: TokenizerWrapper,
    tokenizer,
    input_len: int,
    number: int,
    save_path: str,
    base_name: str,
    dp: int,
    repeat_rate: float,
    seed: int,
    prefix_num: int,
    length_mean: Optional[int],
    length_std: Optional[float],
    length_min: Optional[int],
    length_max: Optional[int],
) -> tuple:
    """Create a variable-length prefix dataset using random token generation.

    Pre-samples actual lengths and common-prefix lengths per row,
    then generates max-length text pools and truncates per row.
    """
    # seed=0 means pure random (non-reproducible, using system entropy)
    rng = random.Random() if seed == 0 else random.Random(seed)

    # Pre-sample per-row lengths
    real_lens = [
        max(1, int(sample_target_length(rng, input_len, length_mean, length_std, length_min, length_max)))
        for _ in range(number)
    ]
    common_lens = [
        max(0, min(rl, int(round(rl * repeat_rate))))
        for rl in real_lens
    ]
    max_common_len = max(common_lens) if common_lens else 0

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # ---- Generate prefix pool at max common length ----
    prefix_pool = []
    pbar = tqdm(total=prefix_num, desc="Generating prefix pool", unit="row") if tqdm else None
    for i in range(prefix_num):
        prefix_seed = 0 if seed == 0 else seed + i
        if max_common_len > 0:
            prefix_text = tok_wrapper.get_some_tokens(max_common_len, seed=prefix_seed)
        else:
            prefix_text = ""
        prefix_pool.append(prefix_text)
        if pbar:
            pbar.update(1)
    if pbar:
        pbar.close()

    # Write prefix warmup file
    prefix_dataset = []
    for i in range(prefix_num):
        for _ in range(dp):
            prefix_dataset.append(prefix_pool[i])

    prefix_path = os.path.join(
        save_path, f"prefix-GSM8K-in{max_common_len}-num{dp * prefix_num}-{base_name}.jsonl"
    )
    write_jsonl(prefix_path, prefix_dataset, dp * prefix_num)
    logger.info(f"Prefix warmup file: {prefix_path} ({dp * prefix_num} rows)")

    if repeat_rate >= 1.0:
        dataset_path = os.path.join(
            save_path,
            f"GSM8K-in{max_common_len}-num{number}-{base_name}-repeatRate{repeat_rate}.jsonl",
        )
        write_jsonl(dataset_path, prefix_dataset, number)
        return prefix_path, dataset_path

    # ---- Generate separators (3 tokens per row) ----
    separator_pool = []
    pbar = tqdm(total=number, desc="Generating separators", unit="row") if tqdm else None
    for i in range(number):
        sep_seed = 0 if seed == 0 else seed + prefix_num + i
        sep_text = tok_wrapper.get_some_tokens(3, seed=sep_seed)
        separator_pool.append(sep_text)
        if pbar:
            pbar.update(1)
    if pbar:
        pbar.close()

    # ---- Generate suffix pool at max suffix length ----
    max_suffix_len = max(
        (rl - cl - 3 for rl, cl in zip(real_lens, common_lens)),
        default=1,
    )
    max_suffix_len = max(max_suffix_len, 1)

    suffix_pool = []
    pbar = tqdm(total=number, desc="Generating suffix pool", unit="row") if tqdm else None
    for i in range(number):
        suffix_seed = 0 if seed == 0 else seed + prefix_num + number + i
        suffix_text = tok_wrapper.get_some_tokens(max_suffix_len, seed=suffix_seed)
        suffix_pool.append(suffix_text)
        if pbar:
            pbar.update(1)
    if pbar:
        pbar.close()

    # ---- Stitch per-row: truncated prefix + separator + truncated suffix ----
    dataset = []
    pbar = tqdm(total=number, desc="Stitching variable-length dataset", unit="row") if tqdm else None
    for idx in range(number):
        rl = real_lens[idx]
        cl = common_lens[idx]
        suffix_len_needed = max(0, rl - cl - 3)

        # Truncate prefix to actual common length
        if cl > 0 and prefix_pool[idx % prefix_num]:
            prefix_text = truncate_or_pad_text(tokenizer, prefix_pool[idx % prefix_num], cl)
        else:
            prefix_text = ""

        # Truncate suffix to needed length
        if suffix_len_needed > 0 and suffix_pool[idx]:
            suffix_text = truncate_or_pad_text(tokenizer, suffix_pool[idx], suffix_len_needed)
        else:
            suffix_text = ""

        entry = prefix_text + separator_pool[idx] + suffix_text
        entry = exact_token_length(tokenizer, entry, input_len)
        dataset.append(entry)
        if pbar:
            pbar.update(1)
    if pbar:
        pbar.close()

    length_tag = build_length_tag(input_len, length_mean, length_std, length_min, length_max)
    dataset_path = os.path.join(
        save_path,
        f"GSM8K-{length_tag}-num{number}-{base_name}-repeatRate{repeat_rate}.jsonl",
    )
    write_jsonl(dataset_path, dataset, number)

    avg_hit_ratio = sum(c / r for c, r in zip(common_lens, real_lens)) / len(real_lens)
    logger.info(f"  max_common_len={max_common_len}, max_suffix_len={max_suffix_len}")
    logger.info(f"  avg_hit_ratio={avg_hit_ratio:.2%}")

    return prefix_path, dataset_path
