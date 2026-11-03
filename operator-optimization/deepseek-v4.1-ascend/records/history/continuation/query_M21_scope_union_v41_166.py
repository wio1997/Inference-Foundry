from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,socket
assert socket.gethostname()=="bm-jn-zs-zone1-910C-64G-10-105"
A=Path("/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18")
OUT=A/"M21-exact-originalscope-selector-union-v41.json";assert not OUT.exists()
def pin(p,sha):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha,p;return json.loads(b)
old=pin(A/"M21-allrank-input-and-attention-metadata-native-scope-v37.json","8d8115d611461fdc669c627b2bd062bb00ed9410197e57815d12d12974dfe5cd")
new=pin(Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18/M21-original-entrypoint-and-selector-gap-native-census-v41.json'),'8fa7edfbfad9b66487ff9f1e41a8258e2681a6e6f1ab57c142cd819834262203')
def taskkey(x):return(x["database_index"],x["TASK"]["task_record_rowid"])
oldkeys={taskkey(x)for x in old["native_TASKs"]};outside={taskkey(x)for x in new["TASK_records"]if x["has_outside_chain"]}
oldapi={(x["database_index"],x["API"]["api_rowid"])for x in old["native_APIs"]};newapi={(x["database_index"],x["API"]["api_rowid"])for x in new["API_records"]if not x["in_v37_selector"]}
result=dict(created_utc=datetime.now(timezone.utc).isoformat(),old_TASKs=len(oldkeys),outside_chain_TASKs=len(outside),overlap_TASKs=len(oldkeys&outside),additional_distinct_TASKs=len(outside-oldkeys),union_TASKs=len(oldkeys|outside),old_APIs=len(oldapi),outside_APIs=len(newapi),overlap_APIs=len(oldapi&newapi),union_APIs=len(oldapi|newapi),old_chains=len(old["native_canonical_fullchains"]),additional_chains=sum(not h["in_v37_selector"]for h in new["chain_headers"]),additional_TASK_keys=[list(k)for k in sorted(outside-oldkeys)],no_model_import_device_service_work=True,overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(",",":"))+"\n").encode();OUT.write_bytes(b)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
