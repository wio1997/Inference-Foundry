"""One fresh C32 diagnostic per framework after accepted current prefix metadata.

Reuse the reviewed L20 controller; fresh seventh suffix and new request/run IDs.
No service, model, profiler scope, or sampling budget change.
"""
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'scripts/glm53_C32_L20_paired_diagnostic_controller_v1.py'
assert hashlib.sha256(BASE.read_bytes()).hexdigest() == 'e073caa002e588ad422676d8d560aa1154464cdaafa0177d9d4f98ff6957defd'
spec = importlib.util.spec_from_file_location('l21_reviewed_controller', BASE)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
core.__file__ = str(Path(__file__).resolve())
core.APPROVAL = 'results/C32-L22-paired-diagnostic-actual-review-v1.json'

if __name__ == '__main__':
    core.main()
