from pathlib import Path
import socket,json

assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
archive=Path('/data/tiankuan/wio/migration-166-20261030/source169-project')
relative_roots=['runtime/diagnostic-traces','diagnostic-traces',
 'analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces',
 'analysis/operator-ceiling-v2-20261028/diagnostic-20261029/runtime/diagnostic-traces',
 'analysis/operator-ceiling-v2-20261028/diagnostic-20261029/diagnostic-traces']
found=[];roots=[]
for rel in relative_roots:
    p=archive/rel;roots.append({'path':str(p),'exists':p.is_dir()})
    if not p.is_dir():continue
    for rank in p.glob('dp0_pp0_tp3_dcp0_ep3_rank3_2128_*'):
        db=rank/'ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_3.db'
        found.append({'rank_dir':str(rank),'database':str(db),'exists':db.is_file(),'bytes':db.stat().st_size if db.is_file() else None})
print(json.dumps({'hostname':socket.gethostname(),'scope':'bounded archived profile path lookup only','roots':roots,'found':found,'no_device_work':True},ensure_ascii=False))
