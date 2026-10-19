"""Offline CPU parsing of two copied raw profiles; original 32 captures retained."""
import ast
import base64
import gzip
import hashlib
import importlib.util
import io
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RID = '20261019T-L22-vllm-representative-offline-export-v1'
REVIEW = 'results/C32-L22-vllm-offline-export-review-v1.json'
CIDS = {43: '0b3266fcc1f72005a763fdbd5191cecb564d67b448508c2e1d42e965a2973779',
        45: 'c1f460d681f216677d48e3ca881c72bd3a325f10b19d2d83cba299d47dae8a99'}
FOLDERS = {43: 'dp4_pp0_tp0_dcp0_ep8_rank0_311_20261019080556352_ascend_pt',
           45: 'dp10_pp0_tp0_dcp0_ep20_rank0_305_20261019080534597_ascend_pt'}
RAW_CAP = 192 * 1024**2
GZIP_CAP = 32 * 1024**2
INNER = r'''import base64,gzip,hashlib,io,json,multiprocessing,os,pathlib,shutil,time
source=pathlib.Path('/run/glm53/profile')/FOLDER
assert source.is_dir() and not source.is_symlink()
assert all(not p.is_symlink() for p in source.rglob('*'))
infos=sorted(source.glob('profiler_info*.json'));assert len(infos)==1
assert hashlib.sha256(infos[0].read_bytes()).hexdigest()==INFO_SHA
assert json.loads(infos[0].read_bytes())['rank_id']==EP
api=pathlib.Path('/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler/profiler.py')
assert hashlib.sha256(api.read_bytes()).hexdigest()==API_SHA
root=pathlib.Path('/run/glm53/profile-parsed-L22-representatives-v1');assert not root.exists()
assert shutil.disk_usage(source).free>=4*1024**3
root.mkdir();parsed=root/FOLDER;shutil.copytree(source,parsed)
assert hashlib.sha256((parsed/infos[0].name).read_bytes()).hexdigest()==INFO_SHA
assert not multiprocessing.current_process().daemon
from torch_npu.profiler.profiler import analyse
started=time.monotonic_ns()
analyse(str(parsed),max_process_number=1,export_type='text')
ended=time.monotonic_ns()
out=parsed/'ASCEND_PROFILER_OUTPUT';trace=out/'trace_view.json'
result={'project_id':'glm53','node':NODE,'native_EP':EP,'native_DP':EP//2,'native_TP':0,'original_raw_folder':str(source),'parsed_copy':str(parsed),'original_info_sha256':INFO_SHA,'analyse_monotonic_ns':[started,ended],'analysis_process_daemon':False,'max_process_number':1,'model_requests':0,'new_profiles':0,'MemCache_init_calls':0,'NPU_tensor_ops_in_program':0,'original_raw_untouched':hashlib.sha256(infos[0].read_bytes()).hexdigest()==INFO_SHA,'files':{},'trace':None}
for name in ['api_statistic.csv','kernel_details.csv','op_statistic.csv','operator_details.csv','step_trace_time.csv','task_time.csv']:
 p=out/name
 if not p.is_file():result['files'][name]={'missing':True};continue
 size=p.stat().st_size
 if size>128*1024**2:result['files'][name]={'bytes':size,'hash_skipped':'128MiBcap'};continue
 digest=hashlib.sha256()
 with p.open('rb') as stream:
  header=stream.readline(65536).decode(errors='replace');digest.update(header.encode())
  for chunk in iter(lambda:stream.read(1024**2),b''):digest.update(chunk)
 result['files'][name]={'bytes':size,'sha256':digest.hexdigest(),'header':header.rstrip()}
if trace.is_file():
 stat=trace.stat();item={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'path':str(trace)}
 if stat.st_size<=192*1024**2:
  raw=trace.read_bytes();assert trace.stat().st_mtime_ns==stat.st_mtime_ns and len(raw)==stat.st_size
  item['sha256']=hashlib.sha256(raw).hexdigest();packed=gzip.compress(raw,compresslevel=1,mtime=0)
  item['gzip_bytes']=len(packed);item['gzip_sha256']=hashlib.sha256(packed).hexdigest()
  if len(packed)<=32*1024**2:item['gzip_base64']=base64.b64encode(packed).decode()
  else:item['transfer_error']='32MiBgzipcap'
 else:item['transfer_error']='192MiBrawcap'
 result['trace']=item
result['pass']=bool(result['trace'] and result['trace'].get('gzip_base64') and not result['files']['kernel_details.csv'].get('missing'))
print('GLM53_L22_OFFLINE_EXPORT='+json.dumps(result),flush=True)
'''

def load_inventory():
    p = ROOT / 'scripts/inventory_readonly.py'
    assert hashlib.sha256(p.read_bytes()).hexdigest() == 'e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
    spec = importlib.util.spec_from_file_location('l22_offline_inventory', p)
    inv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inv)
    inv.SSH = list(inv.SSH)
    inv.SSH[2] = '/Users/wio/work/sglang-glm53-pd/ssh-control-l20-foreground-20261019'
    return inv

def plans():
    inv = load_inventory()
    api = ROOT / 'runs/20261019T-L22-native-export-source-v1/source/profiler__profiler.py'
    api_sha = hashlib.sha256(api.read_bytes()).hexdigest()
    codes = {}
    for node, cid in CIDS.items():
        data = json.loads((ROOT / f'runs/20261019T-L22-native-export-source-v1/node-{node}-metadata-source.json').read_bytes())
        worker = next(x for x in data['workers'] if x['folder'] == FOLDERS[node])
        ep = 8 if node == 43 else 20
        assert worker['info']['rank_id'] == ep and worker['raw_done_files'] > 0
        inner = f'NODE={node!r}\nFOLDER={FOLDERS[node]!r}\nEP={ep!r}\nINFO_SHA={worker["info_sha256"]!r}\nAPI_SHA={api_sha!r}\n' + INNER
        ast.parse(inner)
        boot = 'exec timeout --signal=TERM --kill-after=5s 150s python3 -c "$1"'
        args = ['docker', 'exec', '-i', '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'OMP_NUM_THREADS=1', '-e', 'MKL_NUM_THREADS=1', '-e', 'OPENBLAS_NUM_THREADS=1', cid, 'bash', '-c', boot, 'glm53-offline-analysis', inner]
        code = f'''import socket,subprocess,json,urllib.request,hashlib
assert socket.gethostname()=={inv.NODES[node][1]!r}
cid={cid!r}
p=subprocess.run(['docker','inspect',cid,'--format','{{{{.Id}}}} {{{{.State.Running}}}} {{{{.Image}}}}'],text=True,capture_output=True,timeout=10)
assert p.returncode==0 and p.stdout.split()==[cid,'true','sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83']
opener=urllib.request.build_opener(urllib.request.ProxyHandler({{}}))
with opener.open('http://172.16.12.{node}:30400/metrics',timeout=8) as response:raw=response.read(8*1024**2+1);assert response.status==200 and len(raw)<=8*1024**2
lines=[x for x in raw.decode().splitlines() if x.startswith(('vllm:num_requests_running{{','vllm:num_requests_waiting{{'))]
assert len(lines)==16 and all(float(x.rsplit(' ',1)[1])==0 for x in lines)
p=subprocess.run({args!r},text=True,capture_output=True,timeout=160)
print(json.dumps({{'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'native_idle':{{'raw_sha256':hashlib.sha256(raw).hexdigest(),'gauge_count':16,'all_zero':True}}}}))
'''
        ast.parse(code)
        codes[node] = code
    return inv, codes

def main():
    inv, codes = plans()
    review_raw = (ROOT / REVIEW).read_bytes()
    assert len(sys.argv) == 2 and hashlib.sha256(review_raw).hexdigest() == sys.argv[1]
    review = json.loads(review_raw)
    assert review['script_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert review['host_codes_sha256'] == {str(n): hashlib.sha256(c.encode()).hexdigest() for n,c in codes.items()}
    run = ROOT / 'runs' / RID
    run.mkdir(exist_ok=False)
    receipt = {'project_id': 'glm53', 'run_id': RID, 'max_concurrent_SSH': 1, 'model_requests': 0, 'new_profiles': 0, 'nodes': {}}
    for node, code in codes.items():
        pod, _ = inv.NODES[node]
        mapping = inv.invoke(['kubectl get pod -n default ' + shlex.quote(pod) + " -o jsonpath='{.status.hostIP}'"], timeout=15)
        (run / f'mapping-{node}.json').write_text(json.dumps(mapping)+'\n')
        assert mapping['rc'] == 0 and mapping['stdout'] == f'172.16.12.{node}'
        command = shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
        wire = inv.invoke([command], code, timeout=190)
        (run / f'node-{node}-wire.json').write_text(json.dumps(wire)+'\n')
        row = {'wire_rc':wire['rc'],'pass':False}
        if wire['rc'] == 0:
            host = json.loads(wire['stdout'])
            (run / f'node-{node}-analysis.stdout.txt').write_text(host['stdout'])
            (run / f'node-{node}-analysis.stderr.txt').write_text(host['stderr'])
            row.update(analysis_rc=host['rc'], native_idle=host['native_idle'])
            markers = [x for x in host['stdout'].splitlines() if x.startswith('GLM53_L22_OFFLINE_EXPORT=')]
            if len(markers)==1:
                data = json.loads(markers[0].split('=',1)[1])
                trace = data.get('trace') or {}
                encoded = trace.pop('gzip_base64',None)
                if encoded:
                    packed = base64.b64decode(encoded,validate=True)
                    assert len(packed)<=GZIP_CAP and hashlib.sha256(packed).hexdigest()==trace['gzip_sha256']
                    with gzip.GzipFile(fileobj=io.BytesIO(packed)) as stream:raw=stream.read(RAW_CAP+1)
                    assert len(raw)==trace['bytes']<=RAW_CAP and hashlib.sha256(raw).hexdigest()==trace['sha256']
                    path=run/f'node-{node}-trace.json';path.write_bytes(raw)
                    row['trace']={'path':str(path.relative_to(ROOT)),'bytes':len(raw),'sha256':trace['sha256']}
                    row['pass']=host['rc']==0 and data['pass']
                (run/f'node-{node}-export.json').write_text(json.dumps(data,indent=2)+'\n')
        receipt['nodes'][str(node)]=row
        (run/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps({'node':node,'pass':row['pass'],'trace':row.get('trace')}),flush=True)
    receipt['pass']=len(receipt['nodes'])==2 and all(x['pass'] for x in receipt['nodes'].values())
    (run/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':
    main()
