"""New L22 identity/settings for the unchanged bounded read-only collector."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'scripts/collect_C32_L20_paired_native_evidence_v1.py'
assert hashlib.sha256(BASE.read_bytes()).hexdigest() == '7f92cc151635d9d395f3083cc34ccec257040555aa0f7efbbd0a5f6a25b4ba0a'
spec = importlib.util.spec_from_file_location('l21_evidence_base', BASE)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
core.PREPARE_RID = '20261019T-L22-native-evidence-prepare-v3'
core.REVIEW_PATH = 'results/C32-L22-native-evidence-collect-review-v3.json'
core.COLLECT_RUN = {
    fw: f'20261019T-L22-native-evidence-collect-{fw}-v3'
    for fw in ('vllm', 'sglang')
}
core.SETTINGS = {
    'vllm': {
        'path': 'config/C32-L22-paired-diagnostic-v1/vllm-cohort-settings-v1.json',
        'sha256': 'd1aa403505d6d2856981a90896a44fba4760d3d6eaf284e41997896ee3bc2bb9',
    },
    'sglang': {
        'path': 'config/C32-L22-paired-diagnostic-v1/sglang-cohort-settings-v1.json',
        'sha256': '494e746744a085896824c611410828ec625774d7e86caab19b5286e1f24c200a',
    },
}
core.SGLANG_HOST_ROOT = (
    '/data/wio/inference-foundry/glm53/runs/20261019T-L17-D-native-eager-v2/profile/'
    '20261019T-L22-sglang-C32-paired-diagnostic-v1'
)


# Representatives correspond to the empirically qualified nativeDP4/DP10,
# attentionTP2 lanes. Keep raw rank and explicit folder evidence separate.
core.PREFERRED_RANK = {'vllm': {43: 8, 45: 20}, 'sglang': {36: 8, 41: 20}}
old_selection = """    if len(preferred) == 1:
        representative = preferred[0]
        selection_reason = 'unique_preferred_native_rank_id'
    elif workers:"""
new_selection = """    explicit_anchor = [worker for worker in workers
                       if worker['derived_dp'] == PREFERRED_RANK // 2
                       and worker['derived_tp'] == 0]
    if len(preferred) == 1:
        representative = preferred[0]
        selection_reason = 'unique_preferred_native_rank_id'
    elif len(explicit_anchor) == 1:
        representative = explicit_anchor[0]
        selection_reason = 'explicit_folder_DP_anchor_TP0'
    elif workers:"""
assert core.CONTAINER_TEMPLATE.count(old_selection) == 1
core.CONTAINER_TEMPLATE = core.CONTAINER_TEMPLATE.replace(old_selection, new_selection, 1)


# Read native gauges once within the same bounded node program. This proves
# post-cohort idle before admitting the other framework; no model endpoint POST.
core.HOST_TEMPLATE = core.HOST_TEMPLATE.replace(
    'import ast, hashlib, json, pathlib, socket, subprocess',
    'import ast, hashlib, json, pathlib, socket, subprocess, urllib.request, re', 1)
idle_code = r"""    metrics_port = 30400 if FRAMEWORK == 'vllm' else 30600
    metrics_url = 'http://172.16.12.%d:%d/metrics' % (NODE, metrics_port)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(metrics_url, timeout=8) as response:
        metrics_raw = response.read(8*1024**2 + 1)
        assert response.status == 200 and len(metrics_raw) <= 8*1024**2
    names = ('vllm:num_requests_running{', 'vllm:num_requests_waiting{') if FRAMEWORK == 'vllm' else ('sglang:num_running_reqs{', 'sglang:num_queue_reqs{')
    label = 'engine' if FRAMEWORK == 'vllm' else 'dp_rank'
    gauges = {}
    for line in metrics_raw.decode().splitlines():
        if not line.startswith(names):
            continue
        if FRAMEWORK == 'sglang' and 'engine_type="decode"' not in line:
            continue
        rank = int(re.search(label + r'="(\d+)"', line).group(1))
        kind = 'running' if line.startswith(names[0]) else 'waiting'
        assert (rank, kind) not in gauges
        gauges[(rank, kind)] = float(line.rsplit(' ', 1)[1])
    expected = set(range(0, 8) if NODE in (36, 43) else range(8, 16))
    assert set(gauges) == {(rank, kind) for rank in expected for kind in ('running', 'waiting')}
    assert all(value == 0 for value in gauges.values())
    report['native_idle'] = {'url': metrics_url, 'raw_sha256': hashlib.sha256(metrics_raw).hexdigest(), 'raw_bytes': len(metrics_raw), 'native_DP': sorted(expected), 'gauge_count': len(gauges), 'all_running_waiting_zero': True}
"""
needle = '    try:\n        inner = subprocess.run'
assert core.HOST_TEMPLATE.count(needle) == 1
core.HOST_TEMPLATE = core.HOST_TEMPLATE.replace(needle, idle_code + needle, 1)

if __name__ == '__main__':
    if '--collect' in sys.argv:
        review = json.loads((ROOT / core.REVIEW_PATH).read_bytes())
        assert review['collector_entry_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    core.main()
