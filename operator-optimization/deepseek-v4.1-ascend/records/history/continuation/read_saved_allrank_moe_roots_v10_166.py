from pathlib import Path
import socket,json,hashlib
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
p=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/profile-readonly-20261031/allrank-moe-regimes-and-ep3-graph-roots-v10.json')
a=p.stat();assert 0<a.st_size<=50000000;b=p.read_bytes();z=p.stat();assert (a.st_size,a.st_mtime_ns)==(z.st_size,z.st_mtime_ns)
d=json.loads(b);assert d['summary']['rank_count']==16 and len(d['source_manifest'])==16
public=dict(summary=d['summary'],full_result=str(p),result_sha256=hashlib.sha256(b).hexdigest(),source_stats_unchanged=True,
 original_job_status='FULL_DATA_WRITTEN; STDOUT_DISPLAY_GUARD_ONLY_FAILED_AFTER_SAVE; NO_CSV_REQUERY',
 rank_regimes=[dict(ep=r['ep'],MC2_rows=r['all_MC2_rows'],graph_MC2_rows=r['retained_graph_MC2_rows'],regimes=[{k:x[k] for k in ['model_id','physical_M','local_experts','K','core','block_num','mix_block_num','records']} for x in r['regimes']]) for r in d['rank_rows']],
 first_outside_groups=d['EP3_outside_marker_rows'][0]['signature_groups'],first_outside_groups_display_truncated=False)
while len(json.dumps(public,ensure_ascii=False).encode())>30000 and public['first_outside_groups']:
 public['first_outside_groups']=public['first_outside_groups'][:-1];public['first_outside_groups_display_truncated']=True
assert len(json.dumps(public,ensure_ascii=False).encode())<=32000
print(json.dumps(public,ensure_ascii=False))
