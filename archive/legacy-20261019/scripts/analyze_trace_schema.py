#!/usr/bin/env python3
"""Inspect bounded event schema samples in saved TP0 Ascend trace."""
import glob,json
from collections import Counter
from pathlib import Path
base=Path("/data/wio/Inference_Foundry/evidence/20260920_decode_c12_profile")
path=glob.glob(str(base/"torch_raw/*rank0*/ASCEND_PROFILER_OUTPUT/trace_view.json"))[0]
events=json.loads(Path(path).read_text())
ph=Counter();cat=Counter();samples={}
for e in events:
    p=e.get("ph");c=e.get("cat")
    ph[p]+=1;cat[c]+=1
    key=(p,c)
    if len(samples.get(key,[]))<2:
        samples.setdefault(key,[]).append({k:v for k,v in e.items() if k!="args"})
out={"source":path,"events":len(events),"ph":ph,"cat":cat,
     "samples":{str(k):v for k,v in samples.items()}}
p=Path("/data/wio/Inference_Foundry/evidence/20260921_dspark_audit/trace_schema.json")
p.write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps({"events":out["events"],"ph":ph,"cat":cat,"sample_flow":{k:v for k,v in out["samples"].items() if "flow" in k or "'s'," in k or "'f'," in k}},indent=2)[:4500])
