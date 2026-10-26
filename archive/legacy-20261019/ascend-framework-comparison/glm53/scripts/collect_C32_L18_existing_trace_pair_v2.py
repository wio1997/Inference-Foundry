"""New read-only run after fresh authorized SSH authentication; preserve v1 failure."""
import hashlib
import json
from pathlib import Path

import collect_C32_L18_existing_trace_pair_v1 as core

core.RID = '20261019T-L18-existing-SG-native-trace-pair-v2'
core.REVIEW = 'results/C32-L18-existing-trace-transfer-review-v2.json'
core.SOCKET = '/Users/wio/work/sglang-glm53-pd/ssh-control-l18-20261019'

if __name__ == '__main__':
    review = json.loads((core.ROOT / core.REVIEW).read_bytes())
    assert review['launcher_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    core.main()
