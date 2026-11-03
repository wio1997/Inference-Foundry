from pathlib import Path
import json,subprocess,sys
p=Path(__file__).with_name('f09_single_coordinate_witness.py')
out=p.with_name('f09-single-coordinate-witness.json')
assert not out.exists()
r=subprocess.run([sys.executable,str(p)],capture_output=True,text=True,check=True)
x=json.loads(r.stdout)
with out.open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False);f.write('\n')
print(json.dumps({'status':'COMPLETED','coordinates':x['distinguishing_coordinates'],'output':str(out)}))
