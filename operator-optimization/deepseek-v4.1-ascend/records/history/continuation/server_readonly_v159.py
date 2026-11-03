"""Exact verified original asset, no account/asset fallback or device operation."""
from pathlib import Path
import hashlib, shlex
from finalize_origin_coverage_v9 import N, read
from inspect_saved_raw_profile_v155 import BASE


def ssh_args(code):
    identity=read(N/'original166-xinyufeng-recheck-v159.json');assert identity['application_status']=='REMOTE_ROOT_IDENTITY_RETURNED';remote=identity['remote_result'];assert remote['hostname']=='bm-jn-zs-zone1-910C-64G-10-105'and remote['euid']==0
    for row in remote['saved_profile_paths']:
        path=N/'original-profile-cache-v53'/Path(row['path']).relative_to(BASE);b=path.read_bytes();assert len(b)==row['bytes']and hashlib.sha256(b).hexdigest()==row['sha256']
    control=N/'ssh166-v159-xinyufeng.sock';assert control.exists(),'No verified master;do not fallback.'
    return ['/usr/bin/ssh','-F','/dev/null','-S',str(control),'-o','ControlMaster=no','-o','ProxyCommand=false','-o','BatchMode=yes','-o','LogLevel=ERROR','-T','-p','2224','xinyufeng@root@10.55.0.166@bastion.aiops.baai.ac.cn','python3 -I -S -c '+shlex.quote(code)]
