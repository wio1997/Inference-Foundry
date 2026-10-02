# Default configuration for prefix cache performance testing.
# CLI arguments override these values with higher priority.
# Modify this file for convenience when running tests manually.
#
# Note: All numeric values are stored as proper int/float types.
# They are converted to strings internally when substituting into
# the AISBench model config template (modify_aisbench_api).

# ===== Service configuration =====
HOST_IP = "localhost"
HOST_PORT = 8000               # int (converted to str for AISBench template)
URL = ""                       # Full URL overrides host_ip/host_port (e.g. "http://host.docker.internal:8080")
MODEL_NAME = ""                # Empty = auto-detect by AISBench
MODEL_PATH = "/home/weights/model_weights"  # Tokenizer/model weights path
NPU_NUM = 1                    # NPU card count (for single-card throughput)

# ===== AISBench configuration =====
WORK_PATH = "/benchmark"       # AISBench installation directory (container default: /benchmark)
DATASET_PATH = "/home/dataset" # Directory to store generated datasets
OUTPUT_DIR = "./outputs/default"
SUMMARIZER = "default_perf"    # "default_perf" or "stable_stage"

# ===== Test configuration =====
TEST_NAME = ""             # Optional name label (written to result CSV, overridden by --rounds)
INPUT_LEN = 3500               # int
OUTPUT_LEN = 1500              # int (converted to str for AISBench template)
DATA_NUM = 8192                # int
CONCURRENCY = 2048             # int (converted to str for AISBench template)
REQUEST_RATE = 0               # int (converted to str for AISBench template; 0 = burst)
TEST_TYPE = "stream"           # "stream" or "text"
ENABLE_THINK = False           # DeepSeek V3.1 thinking mode

# ===== Prefix configuration =====
PREFIX_NUM = 1                 # int
REPEAT_RATE = 0.5              # float or percentage string (converted by parse_prefix_ratio)
DP = 1                         # Data-parallelism degree (int)
SEED = 1                       # Random seed for dataset generation; 0 = pure random (non-reproducible)

# ===== Variable-length configuration =====
# Uncomment and set values if using variable-length datasets
# LENGTH_MEAN = 32768
# LENGTH_STD = 49152.0
# LENGTH_MIN = 8192
# LENGTH_MAX = 131072
LENGTH_MEAN = None
LENGTH_STD = None
LENGTH_MIN = None
LENGTH_MAX = None

# ===== Results configuration =====
RESULT_CSV = "prefix_bench_result.csv"
RESULT_JSONL = "prefix_bench_result.jsonl"

# ===== Hit rate collection =====
# Pod addresses for prefix cache hit rate metrics.
# Format: ["ip:port"] for each DP domain.
# If empty, defaults to HOST_IP:HOST_PORT.
# PD-disjoint: fill P-node IP + each DP-domain port
# PD-separated with pooling: fill each node IP + each DP-domain port
# POD_INFO = ["141.xx.xx.11:8000", "141.xx.xx.12:8000"]
POD_INFO = []
