from pathlib import Path
import base64,hashlib,json,subprocess,sys
HERE=Path(__file__).resolve().parent
root=json.loads((HERE/'launch.json').read_text())['remote']
names=sys.argv[1:] or ['receipt.json','export-state.json','full-request-functions.json','balanced-reference-0.json','balanced-reference-1.json','balanced-steady.json','scope-analysis.json','function-analysis.json','detailed-performance.json','export-attempt1-state.json']
source='''from pathlib import Path
import base64,json,sys,hashlib
p=Path(sys.argv[1]);out=[]
for name in json.loads(sys.argv[2]):
 assert '/' not in name
 f=p/name
 if not f.exists():out.append({'name':name,'missing':True});continue
 b=f.read_bytes();out.append({'name':name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'data':base64.b64encode(b).decode()})
print(json.dumps(out))
'''
r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',source,root,json.dumps(names)],capture_output=True,timeout=180,check=True)
rows=json.loads(r.stdout)
for row in rows:
 if 'data' in row:
  b=base64.b64decode(row.pop('data'));assert hashlib.sha256(b).hexdigest()==row['sha256'];(HERE/row['name']).write_bytes(b)
print(json.dumps(rows,indent=2))
(HERE/'fetch-receipt.json').write_text(json.dumps(rows,indent=2))
