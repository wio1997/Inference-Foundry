"""New L21 identity/settings for the unchanged bounded read-only collector."""
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
core.PREPARE_RID = '20261019T-L21-native-evidence-prepare-v1'
core.REVIEW_PATH = 'results/C32-L21-native-evidence-collect-review-v1.json'
core.COLLECT_RUN = {
    fw: f'20261019T-L21-native-evidence-collect-{fw}-v1'
    for fw in ('vllm', 'sglang')
}
core.SETTINGS = {
    'vllm': {
        'path': 'config/C32-L21-paired-diagnostic-v1/vllm-cohort-settings-v1.json',
        'sha256': 'c4fa7ebd23d7925716b2b700b8373ee20c0897fa0f5c3c107c36c996fc6d75c7',
    },
    'sglang': {
        'path': 'config/C32-L21-paired-diagnostic-v1/sglang-cohort-settings-v1.json',
        'sha256': '85c87b338dfe5089a7dee81994241117e71f1cc3d3747cde435cf473ccb36a6b',
    },
}
core.SGLANG_HOST_ROOT = (
    '/data/wio/inference-foundry/glm53/runs/20261019T-L17-D-native-eager-v2/profile/'
    '20261019T-L21-sglang-C32-paired-diagnostic-v1'
)

if __name__ == '__main__':
    if '--collect' in sys.argv:
        review = json.loads((ROOT / core.REVIEW_PATH).read_bytes())
        assert review['collector_entry_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    core.main()
