import base64,json,sys,zlib
from pathlib import Path
H=Path(__file__).resolve().parent;s=(H/'collect_operator_guard_census.py').read_text();compile(s,'guard-census.py','exec')
sys.path.insert(0,'/Users/wio/work/171-vllm-migration-20261102');from control import invoke
code='PAYLOAD='+repr(base64.b64encode(zlib.compress(s.encode())).decode())+'\n'+r'''from pathlib import Path
import base64,json,subprocess,zlib
p=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/census177');f=p/'guard-census.py';f.write_bytes(zlib.decompress(base64.b64decode(PAYLOAD)));dest=p/'profile-selected-operator-guard-census.json'
root='/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash/analysis';r=subprocess.run(['docker','exec','-e','TORCH_DEVICE_BACKEND_AUTOLOAD=0','-e','WIO_CENSUS_ANALYSIS_ROOT='+root,'wio-vllm-v41-repro171-20261102','timeout','120','python3','-I','-S',root+'/census177/guard-census.py'],capture_output=True,text=True,timeout=130)
print(json.dumps({'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'data_z':base64.b64encode(zlib.compress(dest.read_bytes())).decode() if dest.exists() else None}))'''
r=invoke('target',code,timeout=150);(H/'guard-census-execution.json').write_text(json.dumps({k:v for k,v in r.items() if k!='data_z'},ensure_ascii=False,indent=2)+'\n');print({k:v for k,v in r.items() if k!='data_z'})
if r['data_z']:
 b=zlib.decompress(base64.b64decode(r['data_z']));(H/'profile-selected-operator-guard-census.json').write_bytes(b);x=json.loads(b);print('census_bytes',len(b));print('missing',x['missing_or_mismatched']);print('function_count',len(x['functions']),'guards',len(x['guards']))
