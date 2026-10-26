#!/usr/bin/env python3
"""Check stream-local ordering around cold ratio4 Compressor tasks in frozen msprof CSV."""
import csv
import glob
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

root=Path("/data/wio/Inference_Foundry")
source=Path(glob.glob(str(root/"evidence/20260920_diagnostic/app_profile_cold2/raw/*/mindstudio_profiler_output/op_summary_*.csv"))[0])
streams=defaultdict(list)
with source.open() as f:
    for row in csv.DictReader(f):
        streams[(row["Device_id"],row["Stream ID"])].append((
            float(row["Task Start Time(us)"].strip()),
            float(row["Task Duration(us)"]),
            row["OP Type"],
            row["Input Shapes"]))
next_ops=Counter()
two_after=Counter()
gaps=[]
attn_deltas=[]
targets=0
for stream, rows in streams.items():
    rows.sort()
    for i, row in enumerate(rows):
        if row[2]!="Compressor" or "8096,4096;1024,4096" not in row[3]:
            continue
        targets+=1
        if i+1<len(rows):
            next_ops[rows[i+1][2]]+=1
            gaps.append((rows[i+1][0]-row[0]-row[1])/1000)
        if i+2<len(rows):
            two_after[rows[i+2][2]]+=1
        for later in rows[i+1:i+25]:
            if later[2]=="SparseAttnSharedkv":
                attn_deltas.append((later[0]-row[0])/1000)
                break
result={
 "source":str(source),"target_shape":"8096,4096;1024,4096",
 "targets":targets,"next_operator_counts":dict(next_ops),
 "second_operator_counts":dict(two_after),
 "next_operator_gap_ms_median":statistics.median(gaps),
 "downstream_sparse_attention_within_24_ops":len(attn_deltas),
 "compressor_to_sparse_attention_start_ms_median":statistics.median(attn_deltas),
 "scope":"Same-device, same-stream order only; does not prove global critical-path savings."
}
out=root/"evidence/20260920_loop017_compressor/stream_order.json"
out.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
