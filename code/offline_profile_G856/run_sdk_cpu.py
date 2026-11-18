"""Full SDK analysis in a device-free container, using job-scoped CLI adaptation."""
import glob,json,os,sys
from pathlib import Path
assert os.environ['DS41_CPU_CANN_ADAPTER']=='g853'
import subprocess
assert subprocess.Popen.__name__=='CANNBackendPopen'
import torch_npu
from torch_npu.profiler.profiler import analyse
from torch_npu.profiler.analysis.prof_common_func._constant import Constant

if __name__=='__main__':
    assert not glob.glob('/dev/davinci*') and not Path('/dev/devmm_svm').exists()
    assert not torch_npu.npu.is_initialized()
    print(json.dumps({'stage':'CPU_only_analysis_start','NPU_initialized':False,'NPU_devices':[],'source':sys.argv[1],'CLI_adapter':'g853'}),flush=True)
    analyse(profiler_path=sys.argv[1],max_process_number=1,export_type=[Constant.Db])
    assert not torch_npu.npu.is_initialized()
    print(json.dumps({'stage':'CPU_only_analysis_returned','NPU_initialized':False,'NPU_devices':[]}),flush=True)
