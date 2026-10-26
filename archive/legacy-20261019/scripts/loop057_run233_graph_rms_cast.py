#!/usr/bin/env python3
"""One-card captured-operator replay screen, not product E2E."""
import json, statistics
from pathlib import Path
import torch
import torch_npu

out=Path("evidence/20260925_loop057_target_bound/run233")
out.mkdir(parents=True,exist_ok=True)
r={"run":233,"service_started":False,"shape":[96,4096],"dtype":"bfloat16","threshold_us_per_call":1000/43}
try:
 torch.npu.set_device(0)
 torch.manual_seed(233)
 x=torch.randn((96,4096),device="npu:0",dtype=torch.bfloat16)
 z=torch.zeros_like(x)
 w=torch.randn((4096,),device="npu:0",dtype=torch.bfloat16)
 def separate():
  y=torch_npu.npu_rms_norm(x,w,1e-6)[0]
  return y,y.float()
 def combined():
  return torch_npu.npu_add_rms_norm_cast(x,z,w,1e-6)
 for _ in range(20):separate();combined()
 torch.npu.synchronize()
 ga=torch.npu.NPUGraph();gb=torch.npu.NPUGraph()
 with torch.npu.graph(ga): ya,yaf=separate()
 with torch.npu.graph(gb): zfp,zbf,_,_=combined()
 ga.replay();gb.replay();torch.npu.synchronize()
 r["parity"]={"bf16_max_abs":float((ya.float()-zbf.float()).abs().max().cpu()),"fp32_max_abs":float((yaf-zfp).abs().max().cpu())}
 assert r["parity"]["bf16_max_abs"]<=0.015625 and r["parity"]["fp32_max_abs"]<=0.015625
 def timed(g):
  a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True)
  a.record();g.replay();b.record();return a,b
 pairs=[]
 for i in range(600):
  if i%2==0: aa=timed(ga);bb=timed(gb)
  else: bb=timed(gb);aa=timed(ga)
  pairs.append((aa,bb))
 torch.npu.synchronize()
 times=[(a[0].elapsed_time(a[1])*1000,b[0].elapsed_time(b[1])*1000) for a,b in pairs]
 ea=[p[0] for p in times];fu=[p[1] for p in times];s=[a-b for a,b in times]
 r["timing_us"]={"separate_median":statistics.median(ea),"combined_median":statistics.median(fu),"paired_saving_median":statistics.median(s),"paired_saving_p10":sorted(s)[59],"paired_saving_p90":sorted(s)[539],"pairs":len(s)}
 r["gate"]="pass_one_card_graph" if r["timing_us"]["paired_saving_median"]>r["threshold_us_per_call"] else "reject_below_1ms_43layer_screen"
except Exception as e:
 r["gate"]="error";r["error"]=type(e).__name__+": "+str(e)[:1000]
(out/"analysis.json").write_text(json.dumps(r,indent=2)+"\n")
print(json.dumps(r,indent=2))
if r["gate"]=="error":raise SystemExit(1)
