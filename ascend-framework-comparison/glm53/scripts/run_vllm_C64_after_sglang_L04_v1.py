"""Resource exclusion before the frozen fresh-HTTP vLLM C64 baseline."""
import hashlib,importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(name,relative,expected):
 p=ROOT/relative;assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main():
 guard=load('resource_exclusion','scripts/glm53_sglang_paired_launch_v1.py','b227e29c24f0f85e4462457db3623a421c35cd1d3a165392e91f96bfd91f077f')
 guard.validate_no_overlap(ROOT,{'complete_before_start':[{'framework':'sglang','run_id':'20261017T1990-sglang-formal-C32-L04-v2','native_capture_run':'20261017T1992-sglang-L04-C32-native-v2','nodes':[34,35,36,41]}]})
 # Performance acceptance is separate: exclusion requires the prior clients and
 # native capture closed plus their captured idle/native health observations.
 baseline=load('frozen_vllm_C64','scripts/run_vllm_formal_C64_v2.py','f3538c8eccb97144764472ef2027870cc6469cc4deefe2a0d3b3585c0d0c1701')
 baseline.main()
if __name__=='__main__':main()
