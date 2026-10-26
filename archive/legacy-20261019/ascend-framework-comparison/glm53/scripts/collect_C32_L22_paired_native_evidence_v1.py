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
core.PREPARE_RID = '20261019T-L22-native-evidence-prepare-v1'
core.REVIEW_PATH = 'results/C32-L22-native-evidence-collect-review-v1.json'
core.COLLECT_RUN = {
    fw: f'20261019T-L22-native-evidence-collect-{fw}-v1'
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

if __name__ == '__main__':
    if '--collect' in sys.argv:
        review = json.loads((ROOT / core.REVIEW_PATH).read_bytes())
        assert review['collector_entry_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    core.main()
