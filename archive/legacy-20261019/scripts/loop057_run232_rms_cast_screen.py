#!/usr/bin/env python3
"""One-card operator screen only; no serving or product verdict."""
import json, math, time
from pathlib import Path
import torch
import torch_npu

out = Path("evidence/20260925_loop057_target_bound/run232")
out.mkdir(parents=True, exist_ok=True)
result = {"run": 232, "kind": "one_card_operator_screen", "service_started": False,
          "shape": [96, 4096], "dtype": "bfloat16", "threshold_us_per_call": 1000/43}
try:
    torch.npu.set_device(0)
    torch.manual_seed(232)
    x = torch.randn((96, 4096), device="npu:0", dtype=torch.bfloat16)
    z = torch.zeros_like(x)
    w = torch.randn((4096,), device="npu:0", dtype=torch.bfloat16)
    eps = 1e-6
    def eager():
        y = torch_npu.npu_rms_norm(x, w, eps)[0]
        return y, y.float()
    def fused():
        return torch_npu.npu_add_rms_norm_cast(x, z, w, eps)
    for _ in range(20):
        eager(); fused()
    torch.npu.synchronize()
    a, af = eager()
    outputs = fused()
    result["fused_outputs"] = [{"shape": list(t.shape), "dtype": str(t.dtype)} for t in outputs]
    result["matching"] = []
    for i,t in enumerate(outputs):
        if t.shape == x.shape:
            for label,ref in [("bf16_norm",a),("fp32_norm",af)]:
                diff = (t.float()-ref.float()).abs()
                result["matching"].append({"output":i,"reference":label,"max_abs":float(diff.max().cpu()),"mean_abs":float(diff.mean().cpu())})
    bf = min((r for r in result["matching"] if r["reference"]=="bf16_norm" and result["fused_outputs"][r["output"]]["dtype"]=="torch.bfloat16"), key=lambda r:r["max_abs"])
    fp = min((r for r in result["matching"] if r["reference"]=="fp32_norm" and result["fused_outputs"][r["output"]]["dtype"]=="torch.float32"), key=lambda r:r["max_abs"])
    result["selected_outputs"] = {"bf16":bf,"fp32":fp}
    assert bf["max_abs"] <= 0.015625 and fp["max_abs"] <= 0.015625
    def timed(fn):
        s=torch.npu.Event(enable_timing=True); e=torch.npu.Event(enable_timing=True)
        s.record(); fn(); e.record(); return s,e
    pairs=[]
    for i in range(400):
        if i%2==0:
            ea=timed(eager); fb=timed(fused)
        else:
            fb=timed(fused); ea=timed(eager)
        pairs.append((ea,fb))
    torch.npu.synchronize()
    d=[((a[0].elapsed_time(a[1])*1000),(b[0].elapsed_time(b[1])*1000)) for a,b in pairs]
    import statistics
    eager_us=[x[0] for x in d]; fused_us=[x[1] for x in d]
    savings=[x[0]-x[1] for x in d]
    result["timing_us"]={"eager_median":statistics.median(eager_us),"fused_median":statistics.median(fused_us),"paired_saving_median":statistics.median(savings),"saving_p10":sorted(savings)[39],"saving_p90":sorted(savings)[359],"pairs":len(d)}
    result["gate"]="pass_one_card" if result["timing_us"]["paired_saving_median"] > result["threshold_us_per_call"] else "reject_below_1ms_43layer_screen"
except Exception as exc:
    result["gate"]="error"
    result["error"]=type(exc).__name__+": "+str(exc)[:1000]
(out/"analysis.json").write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result, indent=2))
if result["gate"]=="error":
    raise SystemExit(1)
