"""One-shot read-only L15 D existing-timeline/source inspector; fixed reviewed entry only."""
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = '20261019T-L15-sglang-existing-D-timeline-source-v1'
REVIEW = 'results/C32-L15-D-timeline-source-actual-review-v1.json'
COLLECTOR_SHA = 'bdc451ea0927c63cd47d76bd7ec9ab4a46270221db952d48a18e6509a236e3a7'


def main():
    path = ROOT / 'scripts/glm53_C32_L15_D_timeline_source_v1.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == COLLECTOR_SHA
    spec = importlib.util.spec_from_file_location('l15_timeline_source', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.RUN_ID == RUN_ID and module.REVIEW == REVIEW
    module.execute(RUN_ID, REVIEW)


if __name__ == '__main__':
    main()
