"""DRAFT_PENDING_PRIMARY_REVIEW: L17 remaining-pipeline constants and local guards.

This module performs no SSH, HTTP, Docker, NPU, SDK, model or Git action.  It
only reads frozen local JSON/Python files, verifies their SHA-256 values and
combines the immutable P and D-only capacity records in memory.  The raw
capacity files are never rewritten.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = "DRAFT_PENDING_PRIMARY_REVIEW"
PROJECT = "glm53"
FRAMEWORK = "sglang"
CONCURRENCY = 32
IMAGE = "sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263"
META_CID = "0dd00f5c35ee8c60cbd697a38fc73a8c37ae498106f9ebfc2933139e953b165c"
P_NODES = (34, 35)
D_NODES = (36, 41)
ALL_NODES = P_NODES + D_NODES

P_CAPACITY = "results/sglang-C32-L12C-selected-P-pool-capacity-v1.json"
P_CAPACITY_SHA = "19355aab78afeb145fd9f27a91954dabe14696f96d09be90da10a36ff66f7d50"
D_CAPACITY = "results/sglang-C32-L17-selected-D-pool-capacity-v2.json"
D_CAPACITY_SHA = "dbf1e19ea5a21e557f03556f5fd950414c4657b7716ad8667ee03849a56ca085"
P34_CID = "a6639da9774e3b62098b21d6e7af803299e11ea3b4f3332705eedb6312902179"
P35_CID = "94dd0021d5343d9c9e9cc5e4c07877e9306bdf93d8bf6382d388c7121e7c57f1"
D36_CID = "fe95e2fca4f0e2832685f688d0616c060fa5774ad5baf0cad375f9bd4c89bb37"
D41_CID = "c5e28348c3ede16066218a3729602f7e409d84c55088d719386e8e90dc8c82b9"
D36_SID = 117
D41_SID = 116
D_REMOTE_ROOT = "/data/wio/inference-foundry/glm53/runs/20261019T-L17-D-native-eager-v2"
P_REMOTE_ROOT = "/data/wio/inference-foundry/glm53/runs/20261019T-L12C-candidate-v1"

MIXED_IDENTITY = "scripts/glm53_C32_L17_mixed_model_identity_v2.py"
MIXED_IDENTITY_SHA = "52277af9ea7bbf6893da0276a93365884d93c05cf3f8619b1a73fec7218eb011"
OLD_CAPTURE_FACTORY = "scripts/glm53_C32_L12C_native_capture_v1.py"
OLD_CAPTURE_FACTORY_SHA = "e08773c054c1e6a636e9ff6de3e5a0f22257c14ddd6b16e53b0bf36a31d86610"
OLD_CAPTURE_BASE = "scripts/glm53_sglang_C32_target_graph_native_capture_v3.py"
OLD_CAPTURE_BASE_SHA = "8cad4c626dd24b028d35bdea17863f047d150ac9e15e7e1486669fe5ad668e03"
INVENTORY = "scripts/inventory_readonly.py"
INVENTORY_SHA = "e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777"
CODEC = "scripts/glm53_native_output_transport_v1.py"
CODEC_SHA = "ed265b69437cc93a9fe8afdc1ac87f151e8168e9c82ff28a5d374f14d0c622d5"
STREAM = "scripts/glm53_stream_C32_packed_native_logs_v1.py"
STREAM_SHA = "e03a299a23f50ee7d7fc7233c0f1cbb221043d8fde62c0cd3e5b04ea55cc5abc"
EAGER_AUDIT = "scripts/glm53_C32_L17_eager_input_audit_v2.py"
EAGER_AUDIT_SHA = "0df4276b5e5f8d10aec1faf25cd24e847dc3ef9a57570f70f257c0ebf539d56f"
EAGER_SOURCE_SHA = "a04d7344319eccbeacefad3f60e559c716f412e680ed0f712645ef149238a6b5"
ENVIRON_SOURCE_SHA = "0b0bb923ab563986fe8b1e068330959d249c859f6b24ae3f884df4812d457d1c"
D_STAGE = "config/sglang-C32-L17-D-native-eager-stage-v2.json"
D_STAGE_SHA = "a8d6759af5f29a3da15e60b1312bc81eae046e93c99ca3eb366a78bdf055da0c"
P_STAGE = "config/sglang-C32-native-HiCache-L12C-candidate-stage-v1.json"
P_STAGE_SHA = "29194e69b322471e91049179731aa0c84a25c0eb7e7f1481e69cfad437a58d7e"
CACHE_ANNOTATION_HELPER = "scripts/glm53_sglang_cache_annotation_review_local_v1.py"
CACHE_ANNOTATION_HELPER_SHA = "9f408fdfd1df85d1d6ea9e3431d0f3e4dfd171cd50cc67ca3b11a0e5e0fbdb76"
STORAGE_PROOF_HELPER = "scripts/glm53_storage_proof_v1.py"
STORAGE_PROOF_SHA = "eaef90335bc82d3e258c1c25808a3418f58526b5c8ad52d12fd3930f7b7e92c5"
REQUEST_TERMINAL = "scripts/glm53_request_terminal_v1.py"
REQUEST_TERMINAL_SHA = "34e9f176acea0bde847b349bf85442a12104461260145bead3529aef2ca82ba4"
AIS_TOKEN_IDS = "scripts/aisbench_glm53_token_ids.py"
AIS_TOKEN_IDS_SHA = "d7157131f0b7a536731e2f9921db4740c2fb62668731b504c2c687342d09e13c"
PREFIX_KEYS = "scripts/glm53_sglang_prefix_keys_v1.py"
PREFIX_KEYS_SHA = "a2188e0205d85615a35ca3dd80a5339a7e3a28707c778ecc882aeab6ac3a15e7"
GATEWAY_V2 = "scripts/glm53_sglang_pd_gateway_v2.py"
GATEWAY_V2_SHA = "80922bbe499ddf061b34297bd728e71693241724bc982b527b341eab6faf264f"
GATEWAY_V3 = "scripts/glm53_sglang_pd_gateway_v3.py"
GATEWAY_V3_SHA = "cb2d3581755ec6e55d9d5449a3f9b69c233d611edc42d7a1a6dbf6d891d5d431"
OLD_SHORT_PROBE = "scripts/probe_C32_L12C_native_pd_payload_v1.py"
OLD_SHORT_PROBE_SHA = "d227c13cf48d79a3c68c1e5551def2191942deafea841eb9900de7b2fb114446"
OLD_LONG_CLIENT = "scripts/glm53_C32_L12C_native_long_client_v1.py"
OLD_LONG_CLIENT_SHA = "675825ba0cebf2f3a4541c86f05280e95e10e1852a164c041bd5949f212f2538"
OLD_SEMANTIC_PROBE = "scripts/glm53_C32_L12C_native_semantic_probe_v1.py"
OLD_SEMANTIC_PROBE_SHA = "311638d715844400891cbd1445ac87d8a70c3a08c2f34df92c0d0eaf7dea6fa2"
OLD_LONG_RUNNER = "scripts/run_C32_L12C_native_long_v1.py"
OLD_LONG_RUNNER_SHA = "cbf90e8738c84215bbff06c8932e50c718a2f0c75de220d328c51ea6bff5aff2"
OLD_FORMAL_RUNNER = "scripts/run_C32_L12D_native_formal_v1.py"
OLD_FORMAL_RUNNER_SHA = "fee8feef3991fb2c07d9e4be02868566e24ef30c71c3eb8294da5696173469c5"
OLD_FORMAL_CONTROLLER = "scripts/glm53_C32_L12C_formal_resource_controller_v1.py"
OLD_FORMAL_CONTROLLER_SHA = "1118a029150b92d32f336117b904c3987e59204111df1220bbf46b41f17ce5a3"
OLD_FORMAL_CAPTURE = "scripts/glm53_C32_L12D_formal_native_capture_v1.py"
OLD_FORMAL_CAPTURE_SHA = "548833e16e2cf31dd985f54b13821b33eb54f73ef65b5a662fd9e9af2528562a"
OLD_FORMAL_CLIENT_AUDIT = "scripts/audit_C32_L12D_native_formal_client_local_v1.py"
OLD_FORMAL_CLIENT_AUDIT_SHA = "65ab678c6a57d0b082037022d31d8f5a2bc351e2cdeb78fd41087d32d23c5075"
OLD_FORMAL_NATIVE_AUDIT = "scripts/audit_C32_L12D_native_formal_native_local_v2.py"
OLD_FORMAL_NATIVE_AUDIT_SHA = "5bc9bd543d421e0469d85f5ab62db2f440593e40654e5203d30a3137fef33356"
OLD_READINESS_AUDIT = "scripts/audit_C32_L12C_native_readiness_local_v1.py"
OLD_READINESS_AUDIT_SHA = "e7b5947be61faa80237c440fb02ac1d0c04bc32f97e63c24ee49f9b704e6c54e"
FORMAL_GATE = "scripts/glm53_formal_gate_remote_v1.py"
FORMAL_GATE_SHA = "4d2460e632d37898a63fb9a5d8d8d6cbec480017f6e7e5dc2ed1809412c8e042"
PREFIX_TRANSFER = "scripts/glm53_C32_L12C_prefix_transfer_v1.py"
PREFIX_TRANSFER_SHA = "9d08ccf0b8fe174d3a423bf35c904b07d59154d88e7c6dea12f5fdf74245b980"
FORMAL_RPC = "scripts/glm53_formal_rpc_v2.py"
D_TRANSITION_LIBRARY = "scripts/glm53_C32_L17_D_transition_library_v2.py"
D_TRANSITION_LIBRARY_SHA = "da4bbcab4ddf1dd4670c45db78a20aace39f12e1429e73aa5da9d27d36b7b8ce"
OLD_SHORT_RUNNER = "scripts/run_C32_L12C_native_P_functional_v1.py"
OLD_SHORT_RUNNER_SHA = "da3edf91492d5e9e42b9b94f0e6d810ae9f7a2c173cf87438d548403f3432908"
OLD_STARTUP_AUDIT = "scripts/audit_C32_L12C_native_startup_local_v1.py"
OLD_STARTUP_AUDIT_SHA = "26ca583e149e5935078870f7848e97c59d0f08a940ed8e42e855e615baccf430"
OLD_FUNCTIONAL_AUDIT = "scripts/audit_C32_L12C_native_functional_local_v1.py"
OLD_FUNCTIONAL_AUDIT_SHA = "25f3f97e57fbb8c5cc65eb4b546bb0f8224598f2de6a030bcdff9e6de849c529"
FORMAL_AIS_CLIENT = "scripts/glm53_formal_ais_client_v2.py"
FORMAL_AIS_CLIENT_SHA = "935a4a1c3188b3c7d2041f91e946423838dece7bcc5669daee6ae81c6e6c56a3"
FORMAL_HOST_CONTROLLER = "scripts/glm53_formal_host_controller_v2.py"
FORMAL_HOST_CONTROLLER_SHA = "4cc8d26decf358f34fdc98a1d37d52bb460e72562bb3450728d582d9c9c3b11c"
FORMAL_CACHE_GATE = "scripts/glm53_formal_cache_gate_v1.py"
FORMAL_CACHE_GATE_SHA = "0f04bdeb09a1a8cf4cb6d27d81b4db3858eea081d782aa1a1159cae95096ed23"
FORMAL_RPC_SHA = "da3b46240572916e9b6433da4937ce33433a452912d2279e937df77fd57d8163"

SHORT_RUN = "20261019T-L17-sglang-native-C32-functional-v1"
SHORT_REMOTE = "/data/wio/inference-foundry/glm53/runs/" + SHORT_RUN
SHORT_NAMESPACE = "glm53-sg-C32-L17-native-functional-v1-20261019"
SHORT_RESULT = "results/sglang-C32-L17-native-functional-v1.json"
SHORT_REVIEW = "results/C32-L17-native-short-actual-review-v1.json"
CLIENT_RESULT = "results/sglang-C32-L17-native-short-client-v1.json"
CLIENT_LABEL = "CLIENT_COUNTS_AND_RESPONSE_ONLY"
SHORT_CLIENT_AUDIT = "scripts/audit_C32_L17_native_short_client_local_v2.py"
LONG_RUN = "20261019T-L17-sglang-native-long-v1"
LONG_REMOTE = "/data/wio/inference-foundry/glm53/runs/" + LONG_RUN
LONG_NAMESPACE = "glm53-sem-sglang-L17-native-v1-20261019"
LONG_RESULT = "results/sglang-C32-L17-native-long-v1.json"
LONG_REVIEW = "results/C32-L17-native-long-actual-review-v1.json"
STARTUP_CAPTURE = "20261019T-L17-sglang-native-startup-capture-v3"
STARTUP_CAPTURE_REVIEW = "results/C32-L17-native-startup-capture-review-v3.json"
FUNCTIONAL_CAPTURE = "20261019T-L17-sglang-native-functional-capture-v1"
FUNCTIONAL_CAPTURE_REVIEW = "results/C32-L17-native-post-functional-capture-review-v1.json"
FORMAL_CAPTURE = "20261019T-L17-sglang-native-formal-capture-v1"
FORMAL_CAPTURE_REVIEW = "results/C32-L17-native-formal-capture-review-v1.json"
STARTUP_RESULT = "results/sglang-C32-L17-native-startup-v3.json"
READINESS_RESULT = "results/sglang-C32-L17-native-readiness-v1.json"
FORMAL_RUN = "20261019T-L17-sglang-formal-C32-native-v1"
FORMAL_NAMESPACE = "glm53-formal-" + FORMAL_RUN + "-C32"
FORMAL_SETTINGS = "config/C32-L17-native-runtime-v1/formal-C32-native-v1.json"
FORMAL_REVIEW = "results/C32-L17-native-formal-actual-review-v1.json"
FORMAL_CLIENT_AUDIT = "results/sglang-C32-L17-native-formal-client-audit-v1.json"
FORMAL_NATIVE_AUDIT = "results/sglang-C32-L17-native-formal-native-v1.json"
FORMAL_MEASUREMENT = "results/sglang-C32-L17-native-formal-measurement-v1.json"
FORMAL_CACHE_REVIEW = "results/sglang-C32-L17-native-formal-cache-annotation-review-v1.json"
INDEPENDENT_METRICS_RUN = "20261019T-L17-independent-formal-metrics-archive-v1"
INDEPENDENT_METRICS_NODE = "runs/" + INDEPENDENT_METRICS_RUN + "/node-43.json"
INDEPENDENT_METRICS_REVIEW = "results/C32-L17-independent-formal-metrics-actual-review-v1.json"

INPUT512_SHA = "d6e4cf16d46a7e29461cd4a7e19fa45725cb1b42ac3f76fda226260074db107e"
DATASET_MANIFEST = "runs/20261017T0920-semantic-inputs-v1/manifest.json"
DATASET_MANIFEST_SHA = "ac8fbf5e4425c349440c85789d9a4be7137579a0b2a4072f8a2adf00c9bf6ece"
FORMAL_MANIFEST_SHA = "9bdba915c372eb2c23da018a18c2c9653ded9c5012d875db8b496effb5983b67"
FORMAL_MANIFEST_FILE = "runs/20261017T0615-formal-inputs-v3/C32-manifest.json"
FORMAL_MANIFEST_FILE_SHA = "2f4ea7bf15c92a4e802251a4cc4b77deb6bf246bbb364f0ed5451b095ad0bc52"
AIS_IDENTITY = "runs/20261017T0610-aisbench-assets-v1/identity-summary.json"
AIS_IDENTITY_SHA = "7cf76483c63972186dc4c589fe04e4d8c49d8fa8eb020aab49fc4a10b955bb8f"
NATIVE_CONFIG = "config/sglang-correctness-v11/local.conf"
NATIVE_CONFIG_SHA = "405679fd396268145d85b1bdeee79ae93015237881fc1cf85b76ae2cbe54a8a6"
EXPECTED_SDK_ROOT = "/usr/local/python3.12.13/lib/python3.12/site-packages/memcache_hybrid/"
META_SERVICE_URL = "tcp://172.16.12.34:29780"
CONFIG_STORE_URL = "tcp://172.16.12.34:29781"
MAX_DRAM_SIZE = "160GB"

P_KV_FLOOR = 254848
D_KV_FLOOR = 161280
OLD_D_KV = 181632
SHORT_LOGICAL = 32
SHORT_BACKEND = 64
SHORT_OUTPUT_IF_ALL = 32 * (600 + 1)
LONG_LOGICAL = 6
LONG_BACKEND = 12
LONG_OUTPUT_IF_ALL = 2408
FUNCTIONAL_BACKEND = SHORT_BACKEND + LONG_BACKEND
FUNCTIONAL_OUTPUT_IF_ALL = SHORT_OUTPUT_IF_ALL + LONG_OUTPUT_IF_ALL
FORMAL_WARM_BACKEND = 64
FORMAL_BACKEND = 256
FORMAL_OFFERED_BACKEND = FORMAL_WARM_BACKEND + FORMAL_BACKEND
FORMAL_OUTPUT_IF_ALL = 64 * 1 + 128 * 601
TOTAL_BACKEND_IF_ALL = FUNCTIONAL_BACKEND + FORMAL_OFFERED_BACKEND
TOTAL_OUTPUT_IF_ALL = FUNCTIONAL_OUTPUT_IF_ALL + FORMAL_OUTPUT_IF_ALL

D_EAGER_ENV_KEYS = (
    "SGLANG_EAGER_INPUT_NO_COPY",
    "GLM53_EAGER_INPUT_AUDIT",
    "GLM53_EAGER_INPUT_AUDIT_ROOT",
    "GLM53_NODE",
    "TASK_QUEUE_ENABLE",
    "HCCL_BUFFSIZE",
)
EAGER_ROOT_IN_CONTAINER = "/run/glm53/eager-input-audit"
AUDIT_ALLOWLIST_FIELDS = (
    "input_ids",
    "positions",
    "seq_lens",
    "out_cache_loc",
    "req_pool_indices",
    "seq_lens_cpu",
    "spec_info",
)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_json(relative_or_path):
    path = Path(relative_or_path)
    if not path.is_absolute():
        path = ROOT / path
    return json.loads(path.read_text(encoding="utf-8"))


def assert_source(relative, expected):
    path = ROOT / relative
    actual = sha256_file(path)
    assert actual == expected, "%s sha256 %s != %s" % (relative, actual, expected)
    return path


def load_module(name, relative, expected):
    path = assert_source(relative, expected)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_capacity_pair():
    """Load and combine the immutable P-only and D-only capacity records."""
    p = read_json(P_CAPACITY)
    d = read_json(D_CAPACITY)
    assert sha256_file(ROOT / P_CAPACITY) == P_CAPACITY_SHA
    assert sha256_file(ROOT / D_CAPACITY) == D_CAPACITY_SHA
    assert p["project_id"] == PROJECT and p["pass"] is True
    assert p["selected_variant"] == "candidate"
    assert p["meta_container_id"] == META_CID
    assert d["project_id"] == PROJECT and d["pass"] is True
    assert d["selected_P_variant"] == "candidate"
    assert d["selected_D_variant"] == "candidate"
    assert d["selected_variant"] == "candidate"
    assert d["selected_variant_scope"] == "D_ONLY_NOT_P"
    assert d["model_started"] is False
    assert d["meta_container_id"] == META_CID
    assert d["capacity_floor_tokens"] == D_KV_FLOOR
    assert d["old_D_reported_KV_capacity"] == OLD_D_KV
    assert p["providers"]["34"]["container_id"] == P34_CID
    assert p["providers"]["35"]["container_id"] == P35_CID
    assert p["providers"]["34"]["local_service_id"] == 82
    assert p["providers"]["35"]["local_service_id"] == 83
    assert p["providers"]["34"]["dram_GiB"] == 136
    assert p["providers"]["35"]["dram_GiB"] == 112
    assert d["providers"]["36"]["container_id"] == D36_CID
    assert d["providers"]["41"]["container_id"] == D41_CID
    assert d["providers"]["36"]["local_service_id"] == D36_SID
    assert d["providers"]["41"]["local_service_id"] == D41_SID
    assert d["providers"]["36"]["dram_GiB"] == d["providers"]["41"]["dram_GiB"] == 160
    assert d["providers"]["36"]["remote_root"] == d["providers"]["41"]["remote_root"] == D_REMOTE_ROOT
    old_d_cids = {
        "25ecbd3a55e199caf4c6012a61b702d1b6035c955406d75bb17107991a932dd1",
        "c39fa7c594a8776e70c365baec435d4288abb1db5cc492405183047d57b805c3",
    }
    cids = [p["providers"][str(n)]["container_id"] for n in P_NODES]
    cids += [d["providers"][str(n)]["container_id"] for n in D_NODES]
    assert len(set(cids)) == 4 and not (set(cids) & old_d_cids) and META_CID not in cids
    return p, d


def combined_providers():
    """Return all four role rows in memory; no combined file is written."""
    p, d = check_capacity_pair()
    out = {}
    for node in P_NODES:
        row = dict(p["providers"][str(node)])
        row.update(role="P", native_p=True, selected_D_variant=None)
        out[str(node)] = row
    for node in D_NODES:
        row = dict(d["providers"][str(node)])
        row.update(role="D", native_p=False, selected_D_variant="candidate")
        out[str(node)] = row
    assert set(out) == {str(n) for n in ALL_NODES}
    return out


def capture_review_check(path, host_hashes):
    review_path = ROOT / path
    assert review_path.is_file() and not review_path.is_symlink()
    review = read_json(review_path)
    assert review.get("project_id") == PROJECT
    assert str(review.get("decision", "")).startswith("APPROVED"), "actual review is not approved"
    assert review.get("host_codes_sha256") == host_hashes, "review host hashes differ"
    assert review.get("model_requests") in (0, None)
    return review


def execution_review_check(path, host_hashes):
    review = capture_review_check(path, host_hashes)
    assert review.get("zero_retries") is True, "review must pin zero retries"
    assert review.get("runtime_authorized") is True, "review does not authorize runtime"
    return review


def inference_review_check(path, host_hashes, *, launcher_sha256, files_sha256,
                           logical_client_cases, backend_model_calls,
                           successful_output_tokens_if_all_pass, provenance):
    """Strict once-only model-inference review gate.

    This deliberately does not reuse ``capture_review_check``'s zero-model
    budget: a short review must pin 32 logical / 64 backend / 19232 output and a
    long review must pin 6 logical / 12 backend / 2408 output.  Every source,
    capacity, startup/client-result and host-code hash is asserted explicitly.
    """
    review = read_json(path)
    assert review.get("project_id") == PROJECT
    assert review.get("decision") == "APPROVED", "inference review is not APPROVED"
    assert review.get("runtime_authorized") is True, "review does not authorize runtime"
    assert review.get("zero_retries") is True, "review must pin zero retries"
    assert type(review.get("retries")) is int and review["retries"] == 0
    assert type(review.get("new_profiles")) is int and review["new_profiles"] == 0
    assert type(review.get("formal_requests")) is int and review["formal_requests"] == 0
    assert type(review.get("logical_client_cases")) is int, "logical budget missing"
    assert review["logical_client_cases"] == logical_client_cases, "logical budget differs"
    assert type(review.get("backend_model_calls")) is int, "backend budget missing"
    assert review["backend_model_calls"] == backend_model_calls, "backend budget differs"
    assert type(review.get("successful_output_tokens_if_all_pass")) is int, "output budget missing"
    assert review["successful_output_tokens_if_all_pass"] == successful_output_tokens_if_all_pass, "output budget differs"
    assert review.get("launcher_sha256") == launcher_sha256, "review launcher hash differs"
    assert review.get("files_sha256") == files_sha256, "review file hashes differ"
    assert review.get("host_codes_sha256") == host_hashes, "review host hashes differ"
    assert review.get("p_capacity_sha256") == P_CAPACITY_SHA, "review P capacity differs"
    assert review.get("d_capacity_sha256") == D_CAPACITY_SHA, "review D capacity differs"
    for key, digest in provenance.items():
        assert review.get(key) == digest, "review provenance differs: " + key
    return review


def load_native_capture(relative):
    """Load one saved capture wire and verify the raw native frame before decode."""
    path = ROOT / relative
    wire = read_json(path)
    assert wire["rc"] == 0
    frame = wire.get("native_output_transport")
    raw = wire["stdout"].encode()
    assert frame is not None
    assert frame["version"] == 1
    assert len(raw) == frame["raw_bytes"]
    assert sha256_bytes(raw) == frame["raw_sha256"]
    doc = json.loads(raw)
    assert doc["project_id"] == PROJECT
    return path, frame, doc


def persistent_events(doc):
    rows = []
    for item in doc["persistent_audit"].values():
        normalized = "".join(json.dumps(x, sort_keys=True) + "\n" for x in item["rows"]).encode()
        assert len(normalized) == item["bytes"] and sha256_bytes(normalized) == item["sha256"]
        rows.extend(item["rows"])
    return rows


def alignment_events(doc):
    rows = []
    for item in doc["alignment_audit"].values():
        normalized = "".join(json.dumps(x, separators=(",", ":")) + "\n" for x in item["rows"]).encode()
        assert len(normalized) == item["bytes"] and sha256_bytes(normalized) == item["sha256"]
        rows.extend(item["rows"])
    return rows


def derived_require(relative, extra=None):
    """Read a JSON record that must already exist; never synthesize a pass."""
    path = ROOT / relative
    assert path.is_file(), "actual evidence missing: " + relative
    doc = read_json(path)
    if extra is not None:
        assert extra(doc), "actual evidence gate failed: " + relative
    return doc


def old_source_provenance():
    return {
        "mixed_identity": {"path": MIXED_IDENTITY, "sha256": MIXED_IDENTITY_SHA},
        "old_capture_factory": {"path": OLD_CAPTURE_FACTORY, "sha256": OLD_CAPTURE_FACTORY_SHA},
        "old_capture_base": {"path": OLD_CAPTURE_BASE, "sha256": OLD_CAPTURE_BASE_SHA},
        "inventory": {"path": INVENTORY, "sha256": INVENTORY_SHA},
        "codec": {"path": CODEC, "sha256": CODEC_SHA},
        "stream": {"path": STREAM, "sha256": STREAM_SHA},
        "eager_audit": {"path": EAGER_AUDIT, "sha256": EAGER_AUDIT_SHA},
        "d_stage": {"path": D_STAGE, "sha256": D_STAGE_SHA},
        "p_stage": {"path": P_STAGE, "sha256": P_STAGE_SHA},
        "cache_annotation_helper": {"path": CACHE_ANNOTATION_HELPER, "sha256": CACHE_ANNOTATION_HELPER_SHA},
        "storage_proof": {"path": STORAGE_PROOF_HELPER, "sha256": STORAGE_PROOF_SHA},
        "request_terminal": {"path": REQUEST_TERMINAL, "sha256": REQUEST_TERMINAL_SHA},
        "ais_token_ids": {"path": AIS_TOKEN_IDS, "sha256": AIS_TOKEN_IDS_SHA},
        "prefix_keys": {"path": PREFIX_KEYS, "sha256": PREFIX_KEYS_SHA},
        "gateway_v2": {"path": GATEWAY_V2, "sha256": GATEWAY_V2_SHA},
        "gateway_v3": {"path": GATEWAY_V3, "sha256": GATEWAY_V3_SHA},
        "old_short_probe": {"path": OLD_SHORT_PROBE, "sha256": OLD_SHORT_PROBE_SHA},
        "old_long_client": {"path": OLD_LONG_CLIENT, "sha256": OLD_LONG_CLIENT_SHA},
        "old_semantic_probe": {"path": OLD_SEMANTIC_PROBE, "sha256": OLD_SEMANTIC_PROBE_SHA},
        "old_long_runner": {"path": OLD_LONG_RUNNER, "sha256": OLD_LONG_RUNNER_SHA},
        "old_formal_runner": {"path": OLD_FORMAL_RUNNER, "sha256": OLD_FORMAL_RUNNER_SHA},
        "old_formal_controller": {"path": OLD_FORMAL_CONTROLLER, "sha256": OLD_FORMAL_CONTROLLER_SHA},
        "old_formal_capture": {"path": OLD_FORMAL_CAPTURE, "sha256": OLD_FORMAL_CAPTURE_SHA},
        "old_formal_client_audit": {"path": OLD_FORMAL_CLIENT_AUDIT, "sha256": OLD_FORMAL_CLIENT_AUDIT_SHA},
        "old_formal_native_audit": {"path": OLD_FORMAL_NATIVE_AUDIT, "sha256": OLD_FORMAL_NATIVE_AUDIT_SHA},
        "old_readiness_audit": {"path": OLD_READINESS_AUDIT, "sha256": OLD_READINESS_AUDIT_SHA},
        "d_transition_library": {"path": D_TRANSITION_LIBRARY, "sha256": D_TRANSITION_LIBRARY_SHA},
        "old_short_runner": {"path": OLD_SHORT_RUNNER, "sha256": OLD_SHORT_RUNNER_SHA},
        "old_startup_audit": {"path": OLD_STARTUP_AUDIT, "sha256": OLD_STARTUP_AUDIT_SHA},
        "old_functional_audit": {"path": OLD_FUNCTIONAL_AUDIT, "sha256": OLD_FUNCTIONAL_AUDIT_SHA},
        "formal_ais_client": {"path": FORMAL_AIS_CLIENT, "sha256": FORMAL_AIS_CLIENT_SHA},
        "formal_host_controller": {"path": FORMAL_HOST_CONTROLLER, "sha256": FORMAL_HOST_CONTROLLER_SHA},
        "formal_cache_gate": {"path": FORMAL_CACHE_GATE, "sha256": FORMAL_CACHE_GATE_SHA},
        "formal_gate": {"path": FORMAL_GATE, "sha256": FORMAL_GATE_SHA},
        "prefix_transfer": {"path": PREFIX_TRANSFER, "sha256": PREFIX_TRANSFER_SHA},
        "formal_rpc": {"path": FORMAL_RPC, "sha256": FORMAL_RPC_SHA},
    }


def verify_old_sources():
    for item in old_source_provenance().values():
        assert_source(item["path"], item["sha256"])
    return old_source_provenance()
