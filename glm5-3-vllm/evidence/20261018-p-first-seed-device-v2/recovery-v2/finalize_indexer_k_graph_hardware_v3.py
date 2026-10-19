"""Actual all16 transport proof after finite quality review, before formal timing."""
from pathlib import Path
import datetime
import hashlib
import importlib.util
import json
import subprocess
import sys

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

r=Path(__file__).resolve().parent
repo=r.parents[3]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


m=module('transport',r/'run_prefix_count_service_v1.py')
g=module('guard',repo/'glm5-3-vllm/runtime-patches/state-tail-probe-v1/prepare_launch.py')
ssh=['ssh','-S',m.control,'-oBatchMode=yes','-p','60011','root@61.241.77.34']
review=json.loads((r/'indexer-k-graph-quality-reviewed-v2.json').read_text())
assert review['proceed_finite_hardware_and_formal_review']
export=json.loads(next((r/'indexer-k-graph-copy-export-recovery-v5').glob('*/metrics.json')).read_text())
assert export['completed'] and export['CPU_only_export'] and len(export['rows'])==32
run=next((r/'indexer-k-graph-copy-export-recovery-v5').glob('*/result.json')).parent
# Build a finite fast gzip archive once; generic backup then downloads exact SHA chunks.
remote=json.loads((run/'result.json').read_text())['remote_root']
code="""from pathlib import Path
import subprocess
r=Path(ROOT);p=r/'artifacts.tar.gz';tmp=r/'artifacts.tar.partial'
assert not p.exists() and not tmp.exists()
subprocess.run(['tar','-I','gzip -1','-cf',str(tmp),'-C',str(r),'artifacts'],check=True,timeout=600)
tmp.rename(p)
print(p.stat().st_size)
""".replace('ROOT',repr(remote))
print(json.dumps(dict(phase='CPU_archive_started')),flush=True)
subprocess.run(ssh+['kubectl exec -i -n default sglang-tuning-access-33-20261013 -- nsenter -t1 -m -- python3 -'],input=code.encode(),check=True,timeout=620)
subprocess.run([sys.executable,str(repo/'glm5-3-vllm/tools/backup_artifacts.py'),'--run',str(run),'--control',m.control,
                '--backup-root','/Users/wio/work/glm53-execution-evidence-backup/20261019-indexer-K-graph-all16-copy','--chunk-size-mib','8'],check=True)
receipt=json.loads((run/'artifacts-backup.json').read_text())
assert file_sha(receipt['local_archive'])==receipt['sha256']
report=dict(reviewed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed=True,
            all16_C1_C4_new_copies_SDMA=export['all16_C1_C4_new_copies_SDMA'],total_copies=sum(z['added_copies'] for z in export['rows']),rows=export['rows'],
            archive=receipt,P_CID=review['P_CID'],D_CID=review['D_CID'],
            CPU_export_recovery=dict(first_timeout_rc=124,reused_exported=export['reused_exported'],missing=export['CPU_missing_reexport'],additional_device_requests=0),
            scope='Stopped actual C1/C4 diagnostic profiles and32 exported traces, raw/full trace archives SHA-verified. Strict proof remains incomplete if any tagged copy lacks an exact physical task; never infer RDMA/TCP fallback or complete SDMA from missing links. Not formal performance.')
report['linked_SDMA_copies']=sum(z['physical_types'].get('SDMA_SQE',0) for z in export['rows'])
report['unlinked_copies']=report['total_copies']-report['linked_SDMA_copies']
(r/'indexer-k-graph-hardware-reviewed-v2.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['rows','archive']}),flush=True)

if not report['all16_C1_C4_new_copies_SDMA']:
    closed=dict(closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),decision='INCONCLUSIVE_SDMA_DEFAULT_OFF_INDEXER_K_GRAPH_OVERLAP',formal_executed=False,service_gain_claimed=False,quality_gate_passed=True,total_copies=report['total_copies'],linked_SDMA_copies=report['linked_SDMA_copies'],unlinked_copies=report['unlinked_copies'],unresolved_rows=[dict(condition=z['condition'],rank=z['rank'],added_copies=z['added_copies'],linked_SDMA=z['physical_types'].get('SDMA_SQE',0),trace_sha256=z['trace_sha256']) for z in export['rows'] if not z['all_added_copy_SDMA']],P_CID=report['P_CID'],D_CID=report['D_CID'],hardware_review='indexer-k-graph-hardware-reviewed-v2.json',archive=receipt,stop_rule='User requested finish this experiment, summarize, GitHub, then pause. No additional NPU profile, formal run or new mechanism.',scope='7155 exact physical SDMA links and45 unlinked copies do not establish all-SDMA. No observed alternative physical type is inferred for missing links. Finite functional/quality gate passed, but service acceptance is blocked by the frozen transport precondition. Default helper env remains off.')
    (r/'indexer-k-graph-closed-without-formal-v1.json').write_text(json.dumps(closed,indent=2)+'\n')
    print(json.dumps({k:v for k,v in closed.items() if k not in ['archive','unresolved_rows']}),flush=True)
