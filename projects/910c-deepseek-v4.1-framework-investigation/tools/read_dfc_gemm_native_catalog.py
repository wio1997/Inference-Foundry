#!/usr/bin/env python3
"""Read frozen DFC profiler schema/launch fields and selected include locations."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess

DB='/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/joint168/runtime/dp0_pp0_tp0_dcp0_ep0_rank0_728_20261102190110875_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_0.db'
REMOTE=r'''
import json,subprocess,sys,sqlite3
from pathlib import Path
name='wio-vllm-v41-repro171-20261102'
c,=json.loads(subprocess.check_output(['docker','inspect',name]));assert c['State']['Running'] and c['Image']=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
inner=r"""
from pathlib import Path
import json,sqlite3,sys
p=Path(sys.argv[1]);out={'database':str(p),'exists':p.is_file()}
if p.is_file():
 con=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True)
 tables=[r[0] for r in con.execute('select name from sqlite_master where type="table"')]
 out['schemas']={t:[list(r) for r in con.execute('pragma table_info("'+t.replace('"','""')+'")')] for t in tables if any(s in t.upper() for s in ['TASK','COMPUTE','STRING'])}
 out['selected_task_tables']={}
 for t,cols in out['schemas'].items():
  if any('block' in col[1].lower() for col in cols):
   names=[c[1] for c in cols];cur=con.execute('select * from "'+t.replace('"','""')+'" limit 1');out['selected_task_tables'][t]={'columns':names,'first_row':list(cur.fetchone() or [])}
 con.close()
roots=[Path('/vllm-workspace/vllm-ascend/csrc'),Path('/vllm-workspace/vllm-ascend/third_party')]
files=[]
for root in roots:
 if not root.exists():continue
 for n in ['copy_l0c_to_gm.hpp','tile_copy.hpp','row_major.hpp','block_swizzle.hpp','arch.hpp','dispatch_policy.hpp']:
  for q in root.rglob(n):
   if q.is_file() and '/catlass/' in str(q):files.append({'path':str(q),'bytes':q.stat().st_size})
opp=Path('/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe')
for folder in ['kernel/ascend910_93','config/ascend910_93']:
 for q in (opp/folder).glob('*dispatch_ffn_combine*'):files.append({'path':str(q),'kind':'directory' if q.is_dir() else 'file'})
headers=[]
for r in [Path('/usr/local/Ascend/cann-9.2.0/aarch64-linux/asc'),Path('/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/asc')]:
 if r.exists():
  for n in ['*fixpipe*.h','*data_copy*.h']:
   for q in r.rglob(n):
    if any(a in str(q) for a in ['/dav_c220/','/dav_c2201/','/interface/']):headers.append({'path':str(q),'bytes':q.stat().st_size})
out['catlass_and_package_inventory']=files;out['selected_arch_header_inventory']=headers
print(json.dumps(out))
"""
d=json.loads(subprocess.check_output(['docker','exec',name,'python3','-B','-I','-S','-c',inner,sys.argv[1]]))
p=Path(sys.argv[1]);host={'path':str(p),'exists':p.is_file()}
if p.is_file():
 con=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True)
 tables=[r[0] for r in con.execute('select name from sqlite_master where type="table"')]
 schemas={t:[list(r) for r in con.execute('pragma table_info("'+t.replace('"','""')+'")')] for t in tables if any(v in t.upper() for v in ['TASK','COMPUTE','STRING'])}
 host.update(size_bytes=p.stat().st_size,schemas=schemas,selected_task_tables={})
 for t,cols in schemas.items():
  if any('block' in col[1].lower() for col in cols):host['selected_task_tables'][t]={'columns':[v[1] for v in cols],'first_row':list(con.execute('select * from "'+t.replace('"','""')+'" limit 1').fetchone() or [])}
 con.close()
d['host_database_inventory']=host
d.update(container_id=c['Id'],image=c['Image'],new_NPU_work=0,model_loads=0,service_requests=0,new_profile_collection=0,remote_writes=[]);print(json.dumps(d))
'''


def main():
 p=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,DB],capture_output=True,text=True,timeout=100)
 if p.returncode:raise RuntimeError(p.stderr)
 d=json.loads(p.stdout);d['reader_sha256']=sha256(Path(__file__).read_bytes()).hexdigest();Path('audit/execution-search/dfc-gemm-native-catalog.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(database_exists=d['exists'],schemas=list(d.get('schemas',{})),tables_with_blockdim=list(d.get('selected_task_tables',{})),includes=len(d['catlass_and_package_inventory']),arch_headers=len(d['selected_arch_header_inventory']))))

if __name__=='__main__':main()
