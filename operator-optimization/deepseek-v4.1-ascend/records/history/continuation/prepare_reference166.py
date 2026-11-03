import pathlib,json,hashlib,socket,subprocess,time
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=pathlib.Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030');R=P/'analysis/whole-scope-continuation-20261030';old=pathlib.Path('/data/tiankuan/wio/migration-166-20261030/source169-project/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/official_profile_client.py')
assert old.is_file();root=P/'runtime/benchmark-20261028/workload/frozen/measure1';assert (root/'warmup.json').is_file()
source=old.read_text();assert "Path('/bench-inputs/workload/frozen')" in source;code=source.replace("Path('/bench-inputs/workload/frozen')","Path('/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash/benchmark-20261028/workload/frozen')")
script=P/'runtime/tools/whole_scope166_reference_client.py';assert not script.exists();script.write_text(code)
files=[]
for p in sorted(root.rglob('*')):
 if p.is_file():files.append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
record={'status':'REFERENCE_CLIENT_PREPARED_FROM_SERVER_SOURCE','script':str(script),'original_script':str(old),'original_sha256':hashlib.sha256(source.encode()).hexdigest(),'current_sha256':hashlib.sha256(code.encode()).hexdigest(),'only_client_input_path_adjusted':True,'model_or_serve_script_changed':False,'workload_files':files,'case':{'concurrency':8,'input_tokens':32768,'output_tokens':1024,'expected_prefix_queries':262144,'expected_prefix_hits':235904},'created_unix':time.time()};(R/'reference166-plan.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='workload_files'}))
