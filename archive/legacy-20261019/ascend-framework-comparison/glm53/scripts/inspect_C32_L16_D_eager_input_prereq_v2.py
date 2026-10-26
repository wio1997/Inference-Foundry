"""One-shot read-only L16 D eager-input/source prerequisite; fixed reviewed entry only."""
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = '20261019T-L16-sglang-D-eager-input-source-prereq-v1'
REVIEW = 'results/C32-L16-D-eager-input-prereq-actual-review-v1.json'
COLLECTOR_SHA = '8f28796af2b956a70775b1492a41c0acede820604795061bd8063afd5759c333'


def main():
    path = ROOT / 'scripts/glm53_C32_L16_D_eager_input_prereq_v2.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == COLLECTOR_SHA
    spec = importlib.util.spec_from_file_location('l16_eager_input_prereq', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.RUN_ID == RUN_ID and module.REVIEW == REVIEW
    module.execute(RUN_ID, REVIEW)


if __name__ == '__main__':
    main()
