#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
p=Path(__file__).resolve().parents[2]/"captured-layer-engine-proxies/summary.json"
b=p.read_bytes(); expected="010499ffc68cd21e042407cf075b339736c5f2d547dd538547b9eef261a5ed9f"
assert hashlib.sha256(b).hexdigest()==expected
d=json.loads(b);layers={x["layer"]:x for x in d["layers"]};z=layers[0]
assert d["rank_count"]==16 and z["calls"]==48 and z["rows"]==1824
m=z["metrics"];a=m["aic_mac_ratio"];v=m["aiv_vec_ratio"]
assert a["no_positive_cycles"]==1248 and v["no_positive_cycles"]==432
assert a["missing_with_positive_cycles"]==v["missing_with_positive_cycles"]==0
lo=max(0,a["no_positive_cycles"]-v["no_positive_cycles"])
out={"summary_sha256":expected,"total_rows":z["rows"],"aic_excluded_aiv_positive_at_least":lo,"percent_of_rows_lower_bound":100*lo/z["rows"],"layer_proxies":{i:{k:layers[i]["metrics"][k]["cycle_weighted_ratio"] for k in ("aic_mac_ratio","aiv_vec_ratio")} for i in (16,27)}}
assert out["layer_proxies"][16]["aic_mac_ratio"]<out["layer_proxies"][27]["aic_mac_ratio"]
assert out["layer_proxies"][16]["aiv_vec_ratio"]>out["layer_proxies"][27]["aiv_vec_ratio"]
print(json.dumps(out,ensure_ascii=False,indent=2))
