import os,runpy,sys
from pathlib import Path
os.environ['HCCL_BUFFSIZE']='1280'
os.environ['TASK_QUEUE_ENABLE']='1'
os.environ['HCCL_OP_EXPANSION_MODE']='AIV'
os.environ['VLLM_USE_V2_MODEL_RUNNER']='0'
from deepinfer.model_executor.models.glm5_next.full_model.shared_kernel_preparation import install_shared_kernel_preparation
out=Path(sys.argv[sys.argv.index('--output-dir')+1])
install_shared_kernel_preparation(out/'kernel-preparation')
from deepinfer.model_executor.models.glm5_next.full_model.native_prefill_state import NativePrefillState
import json

def state_off_forward(self, *args, **kwargs):
    self.counts['target_forward_calls'] += 1
    self.counts['control_passthrough_calls'] += 1
    result = self.original_forward(*args, **kwargs)
    if self.counts['control_passthrough_calls'] == 1 and self.stats_path is not None:
        # Exactly one engineering preflight receipt, never per-token logging.
        temp = self.stats_path.with_suffix('.tmp')
        temp.write_text(json.dumps(self.snapshot()) + '\n')
        temp.replace(self.stats_path)
    return result

NativePrefillState.forward = state_off_forward

from execution_route_observer import install_runtime_hooks
install_runtime_hooks('--native-admission-cohort' in sys.argv)

entry=Path(__file__).resolve().parents[1]/'DeepInfer/examples/glm5_full_model_offline.py'
sys.argv[0]=str(entry)
runpy.run_path(str(entry),run_name='__main__')
