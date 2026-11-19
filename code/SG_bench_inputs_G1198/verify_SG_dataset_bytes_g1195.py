"""Bind the current read-only inputs to the original 448-row tokenizer audit."""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import time

ROOT = Path('/PRIVATE_TASK_ROOT')
RUN = ROOT/'dataset_bytes_g1195'
CLIENT = ROOT/'client_ds41_g1122'
PRIOR = ROOT/'client_ds41_g503'
HOST = '<DECODE_HOSTNAME>'
CID = 'd8593a995ef7192cfd5fd9e696d652e24b3d02d77253d2b96a3415a4097f9090'
SOURCES = {
    'original_G83_dataset_manifest.json':'03d895b25c3832d95c2d3091a48cebe7c73969ac7882e5226da3df833495ec7a',
    'dataset_manifest.json':'2491233b84c38ed476d885d8379d464af728b95ea810a4428e2ad65ca076513a',
    'cold_prefix_audit.json':'39c76847a19c277257fb16afb712afa473ae03fd54e094f0745dc1d0d3a3f4ab',
    'asset_revalidation.json':'e063bba111fcab00bcb3795523644a2ef89e533ea9389057db6e2dd47a3ff001',
    'assets_state.json':'553cc4b1299028d11ab280cc44a43047285b9fb40a00821cb0616957ee3f3dd5',
    'create_receipt.json':'b2a3d8b0ba53478a86f6d1cc97b8501577a8d369c1080dbd120414888e497e86',
}
LEASE_CPUS = set(range(160, 184))
VERIFY_CPU = 183

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value, replace=False):
    temp = path.with_suffix('.tmp')
    if not replace:
        assert not path.exists(), path
    with temp.open('x') as f:
        json.dump(value,f,indent=2)
        f.write('\n')
    temp.replace(path)

def inspect(container):
    return json.loads(subprocess.check_output(['docker','inspect',container],timeout=8))[0]

def cpu_set(spec):
    result=set()
    for chunk in spec.split(','):
        if chunk:
            ends=chunk.split('-')
            result.update(range(int(ends[0]),int(ends[-1])+1))
    return result

def verify_datasets():
    previous = {}
    for name, sha in SOURCES.items():
        path=PRIOR/name
        assert file_hash(path)==sha,name
        previous[name]=json.loads(path.read_text())
    original=previous['original_G83_dataset_manifest.json']
    manifest=previous['dataset_manifest.json']
    prefix=previous['cold_prefix_audit.json']
    proof=previous['asset_revalidation.json']
    state=previous['assets_state.json']
    assert state['phase']=='complete' and state['profiles']==24 and state['prefix_count']==448
    assert proof['passed'] and proof['prefixes']==448
    assert manifest['source_original_manifest_sha256']==proof['source_original_manifest_sha256']==SOURCES['original_G83_dataset_manifest.json']
    assert prefix['verified'] and prefix['prefix_tokens']==128 and prefix['prefix_count']==448
    assert all(prefix[k] for k in ('formal_first_prefixes_unique','warmup_first_prefixes_unique','no_formal_warmup_prefix_collision'))
    assert original['tokenizer_sha256']==manifest['tokenizer_sha256']==prefix['tokenizer_sha256']
    current=inspect(CID)
    cpid=2760614
    assert str(CLIENT/'cpu_client_controller.py').encode() in Path('/proc',str(cpid),'cmdline').read_bytes().split(b'\0')
    assert current['Config']['Labels']['codex.controller_pid']==str(cpid)
    assert current['State']['Running'] and current['Image']==previous['create_receipt.json']['image_id']
    spec=current['HostConfig']
    assert cpu_set(spec['CpusetCpus'])==set(range(184,188)) and not spec['Devices'] and not spec['Privileged']
    server=inspect('9f49ac17d640303bc153b130f1334b8531f724a3480fafde003817a8dfd50650')
    assert server['State']['Running'] and cpu_set(server['HostConfig']['CpusetCpus'])==set(range(160))
    assert VERIFY_CPU not in cpu_set(spec['CpusetCpus'])|cpu_set(server['HostConfig']['CpusetCpus'])
    current_mounts={m['Destination']:m for m in current['Mounts']}
    previous_mounts={m['Destination']:m for m in previous['create_receipt.json']['mounts']}
    base='/task/'+CLIENT.name
    assert current_mounts[base]['RW'] and current_mounts[base]['Source']==str(CLIENT)
    assert not current_mounts[base+'/datasets']['RW'] and not current_mounts['/models/ds41']['RW']
    for dest,prior_dest in [(base+'/datasets','/task/client_ds41_g503/datasets'),('/models/ds41','/models/ds41')]:
        assert current_mounts[dest]['Source']==previous_mounts[prior_dest]['Source'] and not previous_mounts[prior_dest]['RW']
    fs=Path('/proc',str(current['State']['Pid']),'root')
    assert file_hash(fs/'models/ds41/tokenizer.json')==original['tokenizer_sha256']

    def indexed(collection, length_key):
        result={(row['phase'],row[length_key]):row for row in collection}
        assert len(result)==len(collection)==24
        return result

    phases=[f'warmup{i}' for i in range(1,6)]+['formal']
    lengths=[32768,65536,131072,262144]
    expected={(phase,length) for phase in phases for length in lengths}
    orig=indexed(original['profiles'],'input_tokens')
    tokens=indexed(proof['profiles'],'length')
    unique=indexed(prefix['datasets'],'input_tokens')
    existing=indexed(manifest['profiles'],'input_tokens')
    assert set(orig)==set(tokens)==set(unique)==set(existing)==expected
    assert manifest['output_tokens']==1024 and manifest['client_concurrency']==16

    all_rows=[]
    for phase,length in sorted(expected):
        item=orig[phase,length]
        key=(phase,length)
        requests=32 if phase=='formal' else 16
        filename=f'{phase}_{length}.jsonl'
        path=fs/base.lstrip('/')/'datasets'/filename
        assert Path(item['path']).name==filename and item['requests']==existing[key]['requests']==tokens[key]['rows']==requests
        assert item['min_actual']==item['max_actual']==existing[key]['min_actual']==existing[key]['max_actual']==length
        sha=hashlib.sha256()
        size=rows=0
        tail=b''
        with path.open('rb') as inp:
            before=os.fstat(inp.fileno())
            while block:=inp.read(1024*1024):
                sha.update(block)
                size+=len(block)
                rows+=block.count(b'\n')
                tail=block[-1:]
            assert (before.st_size,before.st_mtime_ns)==(os.fstat(inp.fileno()).st_size,os.fstat(inp.fileno()).st_mtime_ns)
        rows+=int(bool(tail and tail!=b'\n'))
        assert size==before.st_size and rows==requests
        assert sha.hexdigest()==item['sha256']==tokens[key]['sha256']==unique[key]['sha256']==existing[key]['sha256'],filename
        all_rows.append({'phase':phase,'length':length,'rows':rows,'bytes':size,'sha256':sha.hexdigest()})
    assert sum(v['rows'] for v in all_rows)==448
    reason='Current data and tokenizer bytes exactly match G503. Token lengths and 128-token prefix uniqueness reuse its actual tokenizer audit.'
    checked=now()
    manifest=dict(manifest)
    manifest['profiles']=[dict(e,path=base+'/datasets/'+Path(e['path']).name) for e in manifest['profiles']]
    manifest.update(revalidated_at=checked,verification_method=reason,server_usage_validated=False)
    prefix=dict(prefix)
    prefix.update(verification_method=reason,current_bytes_verified_at=checked)
    result={'at':checked,'profiles':all_rows,'passed':True,'prefixes':448,'CPU_container_ID':CID,
            'source_SHA256':SOURCES,'verification_method':reason,'server_usage_validated':False,
            'model_requests_submitted':False,'host_CPU_affinity':sorted(os.sched_getaffinity(0))}
    for filename in ('dataset_manifest.json','cold_prefix_audit.json','asset_revalidation.json','aisbench_assets_state_g1183.json'):
        assert not (CLIENT/filename).exists(),filename
    write(CLIENT/'dataset_manifest.json',manifest)
    write(CLIENT/'cold_prefix_audit.json',prefix)
    write(CLIENT/'asset_revalidation.json',result)
    write(CLIENT/'aisbench_assets_state_g1183.json',{'phase':'complete','pid':os.getpid(),'at':checked,
        'profiles':24,'prefix_count':448,'server_usage_validated':False,'verification_method':reason})
    result['artifacts']={name:file_hash(CLIENT/name) for name in ('dataset_manifest.json','cold_prefix_audit.json','asset_revalidation.json','aisbench_assets_state_g1183.json')}
    return result

def main():
    assert socket.gethostname()==HOST and os.geteuid()==0
    assert Path(__file__).resolve().parent==RUN
    assert VERIFY_CPU in os.sched_getaffinity(0)
    os.sched_setaffinity(0,{VERIFY_CPU})
    coord=Path('/PRIVATE_COORD_ROOT')
    lock_path=coord/f'{HOST}-cpu160-183.lock'
    assert lock_path.is_file()
    owner=lock_path.with_suffix('.owner.json')
    def interrupted(*_):
        raise TimeoutError('Bounded low-priority input proof interrupted')
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):
        signal.signal(sig,interrupted)
    signal.alarm(60)
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=json.loads(owner.read_text())
        assert previous['state']=='released',previous
        for path in coord.glob(f'{HOST}-cpu*.owner.json'):
            info=json.loads(path.read_text())
            if info['state']=='released':continue
            match=re.search(r'-cpu(\d+)(?:-(\d+))?\.owner\.json$',path.name)
            assert match and not LEASE_CPUS & set(range(int(match[1]),int(match[2] or match[1])+1))
        gen=previous['generation']+1
        claim={'state':'claimed','pid':os.getpid(),'generation':gen,'task_id':ROOT.name,'run_directory':str(RUN),'at':now()}
        write(owner,claim,replace=True)
        start=time.monotonic()
        try:
            result=verify_datasets()
        finally:
            assert json.loads(owner.read_text())==claim
            write(owner,dict(claim,state='released',released_at=now()),replace=True)
    result.update(leased_group='cpu160-183',lease_generation=gen,lease_released=True,
                  elapsed_s=time.monotonic()-start,source_script_sha256=file_hash(Path(__file__)))
    write(RUN/'actual_receipt.json',result)
    print('G1195_CURRENT_DATASET_BYTES_VERIFIED_JSON '+json.dumps(result))

if __name__=='__main__':
    main()
