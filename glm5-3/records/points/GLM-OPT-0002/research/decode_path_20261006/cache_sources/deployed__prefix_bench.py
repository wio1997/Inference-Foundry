"""
Prefix Cache Performance Benchmark Tool.

A streamlined tool for testing vLLM prefix cache performance using AISBench.
Only supports the two-phase prefix test workflow:
  Phase 1 (warmup): Send prefix-only requests at dp concurrency to warm KV cache
  Phase 2 (full):   Send full dataset requests at target concurrency

Configuration priority: CLI arguments > config.py > hardcoded defaults
Results are written to CSV and JSONL with per-DP and aggregated hit rates.

Multi-round support: --rounds accepts JSON file path or inline JSON string,
each round overriding test-specific fields while sharing service/global config.
"""

import argparse
import copy
import errno
import json
import logging
import os
import re
import sys

from config import *  # Load default configuration
from dataset_generator import create_prefix_dataset, parse_prefix_ratio
from hit_rate_collector import HitRateCollector
from result_writer import (
    archive_log,
    build_result_row,
    parse_aisbench_log,
    write_csv,
    write_jsonl,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Round-overridable fields — these can differ per round in multi-round mode
# ---------------------------------------------------------------------------
ROUND_OVERRIDABLE_KEYS = [
    "input_len", "output_len", "data_num", "concurrency",
    "request_rate", "prefix_num", "repeat_rate", "dp", "seed",
    "test_name",  # Optional name label for this round (written to result CSV)
]


# ---------------------------------------------------------------------------
# CLI argument parser — defaults from config.py, CLI overrides
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AISBench Prefix Cache Performance Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration priority: CLI arguments > config.py > hardcoded defaults.
Edit config.py for convenience when running tests manually.

Examples:
  # Basic prefix test with dp=2 (using config.py defaults where not specified)
  python prefix_bench.py --model_path /home/weights/model --work_path /home/benchmark \
      --host_ip 10.0.0.1 --host_port 8004 --input_len 8192 --output_len 1 \
      --data_num 32 --concurrency 8 --repeat_rate 0.5 --dp 2 --seed 1

  # Multi-round testing via --rounds (JSON file or inline)
  python prefix_bench.py --model_path /home/weights/model --work_path /home/benchmark \
      --host_ip 10.0.0.1 --host_port 8004 \
      --rounds '[{"input_len":3500,"seed":1},{"input_len":8500,"seed":2}]'

  # Multi-round testing via JSON file
  python prefix_bench.py --model_path /home/weights/model --work_path /home/benchmark \
      --host_ip 10.0.0.1 --host_port 8004 --rounds rounds_config.json

  # Legacy: shell script multi-round (results accumulate in same CSV)
  for cc in 8 16 32; do
      python prefix_bench.py --model_path /home/weights/model --work_path /home/benchmark \
          --host_ip 10.0.0.1 --host_port 8004 --concurrency $cc --data_num $(($cc*4)) \
          --repeat_rate 0.5 --dp 2 --seed $cc
  done
        """,
    )

    # ===== Service configuration =====
    parser.add_argument("--host_ip", type=str, default=HOST_IP,
                        help=f"Target server IP (default from config: {HOST_IP})")
    parser.add_argument("--host_port", type=int, default=HOST_PORT,
                        help=f"Target server port (default from config: {HOST_PORT})")
    parser.add_argument("--url", type=str, default=URL,
                        help=f"Full URL overriding host_ip/host_port (default from config: {URL}). "
                             "Useful for Docker: http://host.docker.internal:8080")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME,
                        help=f"Model name for API (default from config: {MODEL_NAME}, empty=auto-detect)")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH,
                        help=f"Tokenizer/model weights path (default from config: {MODEL_PATH})")
    parser.add_argument("--npu_num", type=int, default=NPU_NUM,
                        help=f"NPU card count (default from config: {NPU_NUM})")

    # ===== Test configuration =====
    parser.add_argument("--test_name", type=str, default=TEST_NAME,
                        help=f"Optional name label for this test (default from config: {TEST_NAME})")
    parser.add_argument("--input_len", type=int, default=INPUT_LEN,
                        help=f"Input token length (default from config: {INPUT_LEN})")
    parser.add_argument("--output_len", type=int, default=OUTPUT_LEN,
                        help=f"Output token length (default from config: {OUTPUT_LEN})")
    parser.add_argument("--data_num", type=int, default=DATA_NUM,
                        help=f"Number of dataset entries (default from config: {DATA_NUM})")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY,
                        help=f"Max concurrency for full test (default from config: {CONCURRENCY})")
    parser.add_argument("--request_rate", type=int, default=REQUEST_RATE,
                        help=f"Request rate (default from config: {REQUEST_RATE}, 0=burst)")
    parser.add_argument("--test_type", type=str, default=TEST_TYPE,
                        choices=["stream", "text"],
                        help=f"API test type (default from config: {TEST_TYPE})")

    # ===== Prefix configuration =====
    parser.add_argument("--prefix_num", type=int, default=PREFIX_NUM,
                        help=f"Number of distinct prefixes (default from config: {PREFIX_NUM})")
    parser.add_argument("--repeat_rate", type=float, default=REPEAT_RATE,
                        help=f"Prefix repeat rate (default from config: {REPEAT_RATE}, "
                             "supports 0.5 or percentage string via --rounds)")
    parser.add_argument("--dp", type=int, default=DP,
                        help=f"Data-parallelism degree (default from config: {DP})")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"Random seed (default from config: {SEED}; 0=pure random)")

    # ===== Variable-length configuration =====
    parser.add_argument("--length_mean", type=int, default=LENGTH_MEAN,
                        help=f"Gaussian mean for variable length (default from config: {LENGTH_MEAN})")
    parser.add_argument("--length_std", type=float, default=LENGTH_STD,
                        help=f"Gaussian std for variable length (default from config: {LENGTH_STD})")
    parser.add_argument("--length_min", type=int, default=LENGTH_MIN,
                        help=f"Min length (default from config: {LENGTH_MIN})")
    parser.add_argument("--length_max", type=int, default=LENGTH_MAX,
                        help=f"Max length (default from config: {LENGTH_MAX})")

    # ===== AISBench configuration =====
    parser.add_argument("--work_path", type=str, default=WORK_PATH,
                        help=f"AISBench installation directory (default from config: {WORK_PATH}, "
                             "AISBench container default: /benchmark)")
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH,
                        help=f"Dataset storage directory (default from config: {DATASET_PATH})")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help=f"AISBench output directory (default from config: {OUTPUT_DIR})")
    parser.add_argument("--summarizer", type=str, default=SUMMARIZER,
                        help=f"AISBench summarizer (default from config: {SUMMARIZER})")
    parser.add_argument("--enable_think", action="store_true", default=ENABLE_THINK,
                        help="Enable thinking mode for DeepSeek V3.1")

    # ===== Results configuration =====
    parser.add_argument("--result_csv", type=str, default=RESULT_CSV,
                        help=f"Result CSV path (default from config: {RESULT_CSV})")
    parser.add_argument("--result_jsonl", type=str, default=RESULT_JSONL,
                        help=f"Result JSONL path (default from config: {RESULT_JSONL})")

    # ===== Hit rate collection =====
    parser.add_argument("--pod_info", type=str, nargs="+", default=None,
                        help="Pod addresses for hit rate (ip:port pairs). "
                             "If not specified, uses POD_INFO from config or host_ip:host_port")

    # ===== Dataset override =====
    parser.add_argument("--dataset", type=str, default=None,
                        help="Use existing dataset JSONL file (skip generation)")

    # ===== Multi-round configuration =====
    parser.add_argument("--rounds", type=str, default=None,
                        help="Multi-round test config: JSON file path or inline JSON string. "
                             "Each round is a dict overriding test-specific fields "
                             "(input_len, output_len, data_num, concurrency, "
                             "request_rate, prefix_num, repeat_rate, dp, seed, test_name). "
                             "Missing fields inherit from CLI/config defaults. "
                             "seed=0 means pure random. "
                             "E.g.: '[{\"input_len\":8500,\"seed\":2}]' or 'rounds_config.json'. "
                             "Also supports AISBENCH_TEST_CASE env var when --rounds not specified.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Load and parse rounds config
# ---------------------------------------------------------------------------

def load_rounds(rounds_arg) -> list:
    """Load multi-round config from inline JSON, JSON file, or environment variable.

    Args:
        rounds_arg: Either an inline JSON string (starting with '[' or '{'),
                    or a path to a JSON file, or None.
                    If None, falls back to AISBENCH_TEST_CASE environment variable.

    Returns:
        List of dicts, each representing a round's overrides.
        A single dict is wrapped into a list of one element.
        None if no config is available (single-round mode).
    """
    # Resolve source: CLI --rounds > env AISBENCH_TEST_CASE
    source = rounds_arg
    if source is None:
        env_val = os.getenv("AISBENCH_TEST_CASE")
        if env_val:
            source = env_val
            logger.info("Using test config from AISBENCH_TEST_CASE env var")

    if source is None:
        return None

    stripped = source.strip()

    if stripped.startswith("[") or stripped.startswith("{"):
        # Inline JSON string
        try:
            rounds = json.loads(stripped)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse inline JSON: {e}")
            sys.exit(1)
    else:
        # JSON file path
        try:
            with open(stripped, "r", encoding="utf-8") as f:
                rounds = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load JSON from file '{stripped}': {e}")
            sys.exit(1)

    # Wrap single dict into list
    if isinstance(rounds, dict):
        rounds = [rounds]

    if not isinstance(rounds, list):
        logger.error(f"--rounds must be a JSON list or dict, got {type(rounds).__name__}")
        sys.exit(1)

    # Validate each round dict
    for i, round_cfg in enumerate(rounds):
        if not isinstance(round_cfg, dict):
            logger.error(f"Round {i+1} must be a dict, got {type(round_cfg).__name__}")
            sys.exit(1)
        for key in round_cfg:
            if key not in ROUND_OVERRIDABLE_KEYS:
                logger.warning(f"Round {i+1}: unknown override key '{key}' "
                               f"(allowed: {ROUND_OVERRIDABLE_KEYS})")

    return rounds


# ---------------------------------------------------------------------------
# Merge round overrides into base args
# ---------------------------------------------------------------------------

def merge_round_args(base_args: argparse.Namespace, overrides: dict) -> argparse.Namespace:
    """Create a new Namespace from base_args, overridden by round-specific values.

    Only ROUND_OVERRIDABLE_KEYS are allowed to be overridden.
    Global config (host, model, AISBench, etc.) is shared across all rounds.
    """
    round_args = copy.deepcopy(base_args)

    for key, value in overrides.items():
        if key in ROUND_OVERRIDABLE_KEYS:
            setattr(round_args, key, value)
        else:
            logger.warning(f"Skipping non-overridable key '{key}' in round config")

    # Handle repeat_rate that may come as percentage string "50%"
    # parse_prefix_ratio handles both "0.5" and "50%"
    # If overrides provided a string, convert it properly
    if isinstance(round_args.repeat_rate, str):
        round_args.repeat_rate = parse_prefix_ratio(round_args.repeat_rate)

    return round_args


# ---------------------------------------------------------------------------
# Resolve pod_info with priority: CLI > config.py > host_ip:host_port
# ---------------------------------------------------------------------------

def resolve_pod_info(args: argparse.Namespace) -> list:
    """Resolve pod_info with priority: CLI > config.py > url > host_ip:host_port.

    When url is provided (e.g. Docker: http://host.docker.internal:8080),
    extract host:port from it for metrics collection, since the vLLM /metrics
    endpoint is typically on the same host as the API endpoint.
    """
    if args.pod_info is not None:
        # CLI explicitly provided
        return args.pod_info

    # Use config.py POD_INFO (imported via `from config import *`)
    if POD_INFO:
        return POD_INFO

    # If url is provided, extract host:port from it for metrics endpoint
    if args.url:
        url_match = re.search(r"https?://([^:/]+):(\d+)", args.url)
        if url_match:
            return [f"{url_match.group(1)}:{url_match.group(2)}"]

    # Fallback to host_ip:host_port
    return [f"{args.host_ip}:{args.host_port}"]


# ---------------------------------------------------------------------------
# Symlink helper
# ---------------------------------------------------------------------------

def symlink_force(target: str, link_name: str) -> None:
    """Create a symlink, replacing existing one if necessary."""
    logger.info(f"Symlink: {link_name} ==> {target}")
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise


# ---------------------------------------------------------------------------
# AISBench command and config generation
# ---------------------------------------------------------------------------

def generate_aisbench_command(summarizer: str, output_dir: str) -> str:
    """Generate the AISBench CLI command string with platform-appropriate log capture."""
    base_cmd = (
        f"ais_bench --models vllm_api_chat_temp "
        f"--datasets gsm8k_gen_0_shot_cot_str_perf "
        f"--mode perf --summarizer {summarizer} "
        f"--work-dir {output_dir} --debug --num-warmups 0"
    )
    if sys.platform == "win32":
        # Windows: tee not available on cmd.exe, redirect to log file
        return f"{base_cmd} > aisbench.log 2>&1"
    else:
        # Linux/macOS: use tee for simultaneous terminal + log output
        return f"{base_cmd} 2>&1 | tee aisbench.log"


def modify_aisbench_api(
    model_path: str,
    model_name: str,
    host_ip: str,
    host_port: int,
    url: str,
    concurrency: int,
    output_len: int,
    request_rate: int,
    test_type: str,
    enable_think: bool,
    work_path: str,
) -> None:
    """Generate the AISBench model config file from template and symlink it.

    Uses re.sub() line-by-line substitution. Numeric parameters are
    converted to strings internally for template substitution (the template
    uses bare numeric placeholders like port_for_replace without quotes).

    The `url` parameter overrides host_ip/host_port for the actual API endpoint.
    When url is non-empty, AISBench uses it directly (bypassing host_ip validation).
    host_ip must still be a valid IPv4/IPv6 address to pass config validation.
    """
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "default_api.py")
    temp_path = os.path.join(os.getcwd(), "temp_api.py")

    # Determine API class and abbreviation
    if test_type == "text":
        api_class = "VLLMCustomAPIChat"
        api_abbr = "vllm-api-general-chat"
    else:
        api_class = "VLLMCustomAPIChatStream"
        api_abbr = "vllm-api-stream-chat"

    # Generation kwargs string (replaces the dict body inside generation_kwargs=dict(...))
    generation_kwargs = "temperature=0,\n\t\t\tignore_eos=True"
    if enable_think:
        generation_kwargs += ",\n\t\t\tchat_template_kwargs={\"enable_thinking\": True}"

    with open(template_path, "r", encoding="utf-8") as f_in:
        lines = f_in.readlines()

    with open(temp_path, "w", encoding="utf-8") as f_out:
        for line in lines:
            t = re.sub("model_path_for_replace", model_path, line)
            t = re.sub("model_name_for_replace", model_name, t)
            t = re.sub("rr_for_replace", str(request_rate), t)
            t = re.sub("test_type_for_replace", api_class, t)
            t = re.sub("test_abbr_for_replace", api_abbr, t)
            t = re.sub("ip_for_replace", host_ip, t)
            t = re.sub("port_for_replace", str(host_port), t)
            t = re.sub("url_for_replace", url, t)
            t = re.sub("outputlen_for_replace", str(output_len), t)
            t = re.sub("concurrency_for_replace", str(concurrency), t)
            t = re.sub("generation_kwargs_for_replace", generation_kwargs.expandtabs(4), t)
            f_out.write(t)

    # Symlink to AISBench config directory
    target_config_dir = os.path.join(
        work_path, "ais_bench/benchmark/configs/models/vllm_api"
    )
    if not os.path.exists(target_config_dir):
        logger.warning(f"AISBench model config dir not found: {target_config_dir}")

    symlink_force(
        temp_path,
        os.path.join(target_config_dir, "vllm_api_chat_temp.py"),
    )


# ---------------------------------------------------------------------------
# AISBench workspace preparation
# ---------------------------------------------------------------------------

def prepare_aisbench_dataset_dir(work_path: str) -> str:
    """Ensure the AISBench GSM8K dataset directory exists with train.jsonl."""
    dst_dir = os.path.join(work_path, "ais_bench/datasets/gsm8k")

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)
        logger.info(f"Created dataset dir: {dst_dir}")

    train_file = os.path.join(dst_dir, "train.jsonl")
    if not os.path.exists(train_file):
        with open(train_file, "w", encoding="utf-8") as f:
            f.write("")
        logger.info(f"Created empty train.jsonl: {train_file}")

    return dst_dir


def link_dataset_to_aisbench(src_file: str, dst_dir: str) -> None:
    """Symlink the source dataset file as test.jsonl in the AISBench dataset dir."""
    dst_file = os.path.join(dst_dir, "test.jsonl")
    logger.info(f"Linking dataset: {src_file} → {dst_file}")
    symlink_force(src_file, dst_file)


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

def validate_args(args: argparse.Namespace) -> None:
    """Validate variable-length parameter constraints."""
    if (args.length_mean is None) ^ (args.length_std is None):
        raise ValueError("length_mean and length_std must both be provided or both omitted")
    if (args.length_min is None) ^ (args.length_max is None):
        raise ValueError("length_min and length_max must both be provided or both omitted")
    if args.length_mean is not None and args.length_mean < 1:
        raise ValueError("length_mean must be >= 1")
    if args.length_std is not None and args.length_std < 0:
        raise ValueError("length_std must be >= 0")
    if args.length_min is not None and args.length_min < 1:
        raise ValueError("length_min must be >= 1")
    if args.length_max is not None and args.length_max < 1:
        raise ValueError("length_max must be >= 1")


# ---------------------------------------------------------------------------
# Single round execution (warmup → full → write results)
# ---------------------------------------------------------------------------

def run_single_round(args: argparse.Namespace, round_index: int) -> None:
    """Execute a complete two-phase prefix test for one round configuration.

    Args:
        args: Merged argparse Namespace for this round.
        round_index: Round number (1-based), written into result rows.
    """
    # Parse repeat_rate (handles both float and percentage string)
    repeat_rate = parse_prefix_ratio(args.repeat_rate)

    # Resolve pod_info
    pod_info = resolve_pod_info(args)
    collector = HitRateCollector(pod_info)

    logger.info("-" * 60)
    logger.info(f"[Round {round_index}] Configuration:")
    for key in ROUND_OVERRIDABLE_KEYS:
        logger.info(f"  {key}: {getattr(args, key)}")
    logger.info(f"  repeat_rate (parsed): {repeat_rate}")
    logger.info(f"  pod_info (resolved): {pod_info}")
    logger.info("-" * 60)

    # ---- Step 1: Generate or load dataset ----
    if args.dataset:
        if not os.path.exists(args.dataset):
            logger.error(f"Dataset file not found: {args.dataset}")
            return
        src_file_data = args.dataset
        src_file_prefix = ""
        logger.info(f"Using provided dataset: {args.dataset}")
    else:
        if not os.path.exists(args.dataset_path):
            os.makedirs(args.dataset_path, exist_ok=True)
            logger.info(f"Created dataset directory: {args.dataset_path}")

        src_file_prefix, src_file_data = create_prefix_dataset(
            tokenizer_path=args.model_path,
            input_len=args.input_len,
            number=args.data_num,
            save_path=args.dataset_path,
            dp=args.dp,
            repeat_rate=repeat_rate,
            seed=args.seed,
            prefix_num=args.prefix_num,
            length_mean=args.length_mean,
            length_std=args.length_std,
            length_min=args.length_min,
            length_max=args.length_max,
        )
        logger.info(f"Prefix file: {src_file_prefix}")
        logger.info(f"Full dataset: {src_file_data}")

    # Persist actual tokenizer-based dataset evidence before any request.
    if not args.dataset:
        import json as _json
        from transformers import AutoTokenizer as _AutoTokenizer
        _tok = _AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        _full = [_json.loads(l)["question"] for l in open(src_file_data) if l.strip()]
        _prefix = [_json.loads(l)["question"] for l in open(src_file_prefix) if l.strip()]
        _lens = [len(_tok.encode(x, add_special_tokens=False)) for x in _full]
        _plens = [len(_tok.encode(x, add_special_tokens=False)) for x in _prefix]
        _valid = len(_full) == args.data_num and all(n == args.input_len for n in _lens)
        _valid = _valid and all(n == int(args.input_len * repeat_rate) for n in _plens)
        _valid = _valid and all(x.startswith(_prefix[i % args.prefix_num]) for i,x in enumerate(_full))
        _evidence = dict(raw_prompt_token_lengths=_lens, warmup_prefix_token_lengths=_plens,
                         prefix_text_matches=True if _valid else False, valid=_valid,
                         input_semantics="raw prompt; chat template tokens are additional")
        with open("dataset_validation.json", "w") as _f: _json.dump(_evidence, _f, indent=2)
        if not _valid: raise ValueError("Dataset token length/prefix validation failed")

    # ---- Step 2: Prepare AISBench workspace ----
    dst_dir = prepare_aisbench_dataset_dir(args.work_path)

    # ---- Step 3: Generate AISBench command ----
    aisbench_cmd = generate_aisbench_command(args.summarizer, args.output_dir)
    logger.info(f"AISBench command: {aisbench_cmd}")

    # ======================================================================
    # Phase 1: Warmup prefix (concurrency=dp, output_len=1)
    # ======================================================================
    logger.info(f"[Round {round_index} Phase 1] Warmup prefix — "
                f"concurrency={args.dp}, output_len=1")

    modify_aisbench_api(
        model_path=args.model_path,
        model_name=args.model_name,
        host_ip=args.host_ip,
        host_port=args.host_port,
        url=args.url,
        concurrency=args.dp,
        output_len=1,
        request_rate=args.request_rate,
        test_type=args.test_type,
        enable_think=args.enable_think,
        work_path=args.work_path,
    )

    # Link prefix dataset
    if src_file_prefix:
        link_dataset_to_aisbench(src_file_prefix, dst_dir)

    # Snapshot before warmup
    warmup_before = collector.snapshot()

    # Execute warmup
    ret = os.system(aisbench_cmd)
    if ret != 0:
        logger.warning(f"AISBench warmup phase returned exit code: {ret}")

    # Snapshot after warmup
    warmup_after = collector.snapshot()

    # Compute warmup hit rate
    warmup_hit_rate = collector.compute_hit_rate(warmup_before, warmup_after)
    collector.print_hit_rate_table(warmup_hit_rate)

    # Parse warmup metrics and write to results
    warmup_perf, warmup_log_dir = parse_aisbench_log(
        "aisbench.log", str(args.request_rate), args.npu_num
    )
    archive_log("aisbench.log", warmup_log_dir)

    warmup_row = build_result_row(
        warmup_perf, warmup_hit_rate, args, phase="warmup", round_index=round_index
    )
    write_csv(warmup_row, args.result_csv)
    write_jsonl(warmup_row, args.result_jsonl)

    logger.info(f"[Round {round_index} Phase 1] Warmup complete")

    # ======================================================================
    # Phase 2: Full dataset test
    # ======================================================================
    logger.info(f"[Round {round_index} Phase 2] Full dataset test — "
                f"concurrency={args.concurrency}, output_len={args.output_len}")

    modify_aisbench_api(
        model_path=args.model_path,
        model_name=args.model_name,
        host_ip=args.host_ip,
        host_port=args.host_port,
        url=args.url,
        concurrency=args.concurrency,
        output_len=args.output_len,
        request_rate=args.request_rate,
        test_type=args.test_type,
        enable_think=args.enable_think,
        work_path=args.work_path,
    )

    # Link full dataset
    link_dataset_to_aisbench(src_file_data, dst_dir)

    # Snapshot before full test
    full_before = collector.snapshot()

    # Execute full test
    ret = os.system(aisbench_cmd)
    if ret != 0:
        logger.warning(f"AISBench full test phase returned exit code: {ret}")

    # Snapshot after full test
    full_after = collector.snapshot()

    # Compute full test hit rate
    full_hit_rate = collector.compute_hit_rate(full_before, full_after)
    collector.print_hit_rate_table(full_hit_rate)

    # Parse full test metrics
    full_perf, full_log_dir = parse_aisbench_log(
        "aisbench.log", str(args.request_rate), args.npu_num
    )
    archive_log("aisbench.log", full_log_dir)

    # Build and write final result row
    full_row = build_result_row(
        full_perf, full_hit_rate, args, phase="full", round_index=round_index
    )
    write_csv(full_row, args.result_csv)
    write_jsonl(full_row, args.result_jsonl)

    logger.info(f"[Round {round_index} Phase 2] Full test complete")


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_arguments()
    validate_args(args)

    # Load rounds config (None = single-round mode)
    rounds = load_rounds(args.rounds)

    # Build round configs list
    if rounds is None:
        # Single-round mode: use base args as-is
        rounds_config = [{}]
    else:
        rounds_config = rounds

    # Log global configuration (shared across all rounds)
    logger.info("=" * 60)
    logger.info("Prefix Cache Performance Test")
    logger.info("=" * 60)
    logger.info("  Priority: CLI arguments > config.py > hardcoded defaults")
    logger.info(f"  Rounds: {len(rounds_config)}")
    logger.info("")
    logger.info("  Global config (shared across rounds):")
    for key, value in sorted(vars(args).items()):
        if key in ROUND_OVERRIDABLE_KEYS or key == "rounds" or key == "pod_info":
            continue  # shown per-round or separately
        logger.info(f"    {key}: {value}")
    logger.info(f"  pod_info (resolved): {resolve_pod_info(args)}")
    logger.info("")

    if rounds is not None:
        logger.info("  Per-round overrides:")
        for i, rc in enumerate(rounds_config):
            logger.info(f"    Round {i+1}: {rc}")
    else:
        logger.info("  Single-round mode (no --rounds specified)")
    logger.info("=" * 60)

    # Execute each round
    for i, overrides in enumerate(rounds_config):
        round_args = merge_round_args(args, overrides)
        run_single_round(round_args, round_index=i + 1)

    logger.info("=" * 60)
    logger.info(f"All {len(rounds_config)} rounds complete")
    logger.info(f"Results saved to: {args.result_csv} and {args.result_jsonl}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
