import os,runpy,sys
from pathlib import Path
os.environ['HCCL_BUFFSIZE']='1280'
os.environ['TASK_QUEUE_ENABLE']='1'
os.environ['HCCL_OP_EXPANSION_MODE']='AIV'
os.environ['VLLM_USE_V2_MODEL_RUNNER']='0'
from deepinfer.model_executor.models.glm5_next.full_model.shared_kernel_preparation import install_shared_kernel_preparation
out=Path(sys.argv[sys.argv.index('--output-dir')+1])
install_shared_kernel_preparation(out/'kernel-preparation')
entry=Path(__file__).resolve().parents[1]/'DeepInfer/examples/glm5_full_model_offline.py'
sys.argv[0]=str(entry)
runpy.run_path(str(entry),run_name='__main__')
