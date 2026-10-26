"""Fresh once-only L20 entry after v1 stopped locally before any model dispatch.

The reviewed controller implementation is unchanged. This entry separately pins
that implementation and becomes the durable child entry/self-hash identity.
"""
import hashlib
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'scripts/glm53_C32_L20_paired_diagnostic_controller_v1.py'
assert hashlib.sha256(BASE.read_bytes()).hexdigest()=='e073caa002e588ad422676d8d560aa1154464cdaafa0177d9d4f98ff6957defd'
spec=importlib.util.spec_from_file_location('l20_reviewed_controller_v2',BASE)
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
core.__file__=str(Path(__file__).resolve())
core.APPROVAL='results/C32-L20-paired-diagnostic-actual-review-v2.json'

if __name__=='__main__':
    core.main()
