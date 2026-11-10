from pathlib import Path
import subprocess,json
HERE=Path(__file__).resolve().parent
p=json.loads((HERE/'launch.json').read_text())['remote']
source='''from pathlib import Path
import sys,json
p=Path(sys.argv[1]);r=p/'receipt.json'
d=json.loads(r.read_text()) if r.exists() else {};print(json.dumps({'status':d.get('status'),'phases':[{'phase':x['phase'],'seconds':x['batch_seconds'],'tokens_s':x['output_tokens_per_second'],'balance':all(x['balance_checks'].values())} for x in d.get('phases',[])],'error':d.get('error')},ensure_ascii=False));
print('CONTROLLER', (p/'controller.log').read_text()[-1600:])
s=p/'service.log'
if s.exists():print('SERVICE',s.read_text()[-600:])
'''
r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-c',source,p],capture_output=True,text=True,timeout=90)
print(r.stdout);print(r.stderr if r.returncode else '')
