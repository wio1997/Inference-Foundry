"""Compare actual same-instance Prefill compiler snapshots at two idle boundaries."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

def read(path):
    raw=path.read_bytes()
    value=json.loads(raw)
    return value,hashlib.sha256(raw).hexdigest()

def compare(before_path, after_path):
    before,bsha=read(before_path)
    after,asha=read(after_path)
    a,b=before['summary'],after['summary']
    for key in ('P_controller_PID','P_container_ID','image_ID','source_function_SHA256','selected_env'):
        assert a[key]==b[key],key
    assert datetime.datetime.fromisoformat(a['at'])<datetime.datetime.fromisoformat(b['at'])
    assert a['ready']==b['ready']==200 and a['guard_FD_verified'] and b['guard_FD_verified']
    assert all(v and sum(v)==0 for snap in (a,b) for v in snap['idle'].values())
    def changed(x,y):
        return {name:{'before':x.get(name),'after':y.get(name)}
                for name in sorted(set(x)|set(y)) if x.get(name)!=y.get(name)}
    x,y=before['compiler_files'],after['compiler_files']
    assert len(x)==a['compiler_file_count'] and len(y)==b['compiler_file_count']
    active={k:v for k,v in changed(x,y).items() if Path(k).name!='lock' and Path(k).suffix!='.lock'}
    raw_work=lambda snap:{k:v for group in snap['separate_Ascend_work_files'].values() for k,v in group.items()}
    auxiliary=changed(raw_work(before),raw_work(after))
    def diagnostic(name):
        path=Path(name)
        return '.lock' in path.parts or path.suffix=='.log' or (path.parent.name=='kernel_meta' and path.name=='buildPidInfo.json')
    output_changes={k:v for k,v in auxiliary.items() if not diagnostic(k)}
    return {'same_actual_P_instance':True,'P_container_ID':a['P_container_ID'],
            'before_snapshot_SHA256':bsha,'after_snapshot_SHA256':asha,
            'before_at':a['at'],'after_at':b['at'],'before_label':a.get('boundary_label'),'after_label':b.get('boundary_label'),
            'primary_cache_changes':active,'ascend_work_output_changes':output_changes,
            'ascend_work_diagnostic_changes':{k:v for k,v in auxiliary.items() if diagnostic(k)},
            'persistent_compiler_artifacts_unchanged':not active and not output_changes,
            'scope':'Actual immutable identity and file metadata comparison. The caller must bind the snapshots to the correct client phase; log changes are retained separately.'}

if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('before',type=Path)
    cli.add_argument('after',type=Path)
    args=cli.parse_args()
    print(json.dumps(compare(args.before,args.after),indent=2))
