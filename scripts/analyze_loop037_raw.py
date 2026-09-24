from pathlib import Path
from torch_npu.profiler.profiler import analyse
root = Path("/data/wio/Inference_Foundry/evidence/20260924_loop037_target/run100/profile")
for path in sorted(root.glob("*rank*")):
    print("ANALYZE", path.name, flush=True)
    try:
        analyse(str(path))
    except Exception as exc:
        print("FAILED", path.name, type(exc).__name__, str(exc), flush=True)
