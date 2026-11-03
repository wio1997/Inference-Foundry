import json,hashlib,sqlite3
from pathlib import Path
import torch
from torch_npu.profiler.profiler import analyse
torch.set_num_threads(1)
root=Path('/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash/analysis/history167-creation-cpu')
raw=root/'raw'
files=[p for p in raw.rglob('*') if p.is_file()]
before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
analyse(str(raw),max_process_number=1,export_type='db')
assert all(p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==before[str(p)] for p in files)
dbs=list(raw.rglob('ascend_pytorch_profiler.db'));assert len(dbs)==1
p=dbs[0];c=sqlite3.connect(p);c.row_factory=sqlite3.Row
names=[r[0] for r in c.execute("select name from sqlite_master where type='table'")]
out={'database':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'raw_inputs_unchanged':True,'raw_input_files':len(files),'tables':{},'all_table_names':names}
for name in ['STRING_IDS','MSTX_EVENTS','CANN_API','TASK','COMPUTE_TASK_INFO','ENUM_API_TYPE','ENUM_TASK_TYPE','PYTORCH_API','PYTORCH_CALLCHAINS']:
 if name in names:out['tables'][name]={'schema':[r[1] for r in c.execute('pragma table_info('+name+')')],'rows':[dict(r) for r in c.execute('select * from '+name)]}
(root/'OFFLINE-REVIEW.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'raw_unchanged':True,'rows':{n:len(v['rows']) for n,v in out['tables'].items()}}))
