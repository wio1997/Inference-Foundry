from pathlib import Path
from torch_npu.profiler.profiler import analyse
root=Path('/data/wio/Inference_Foundry/evidence/20260926_loop062_nongmm/run260/profile')
paths=sorted(root.glob('rank*_ascend_pt'))
print('directories',len(paths),flush=True)
for p in paths:
 print('ANALYZE',p.name,flush=True)
 analyse(str(p))
