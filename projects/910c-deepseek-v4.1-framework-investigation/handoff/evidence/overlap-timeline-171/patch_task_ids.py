from pathlib import Path
import json,subprocess
P=Path(__file__).resolve().parent;x=json.load(open(P/'measured-timeline.json'));selected=[r for r in x['ranks'] if r['tasks']]
args=[{'ep':r['ep'],'path':r['source'],'lo':r['origin_us'],'a':r['cycles'][3]['start_ms'],'b':r['cycles'][5]['end_ms']} for r in selected]
code='''from pathlib import Path
import csv,json,sys
out={}
for item in json.loads(sys.argv[1]):
 lo=item['lo'];a=lo+item['a']*1000;b=lo+item['b']*1000;xs=[]
 for row in csv.DictReader(open(Path(item['path']).with_name('kernel_details.csv'))):
  try:s=float(row['Start Time(us)']);e=s+float(row['Duration(us)'])
  except(ValueError,KeyError):continue
  if e<a or s>b:continue
  xs.append([row['Stream ID'],row['Task ID']])
 out[str(item['ep'])]=xs
print(json.dumps(out))
'''
r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',code,json.dumps(args)],capture_output=True,timeout=90,check=True);ids=json.loads(r.stdout)
for rank in selected:
 rows=ids[str(rank['ep'])];assert len(rows)==len(rank['tasks'])
 for t,(stream,task) in zip(rank['tasks'],rows):t['stream']=stream;t['task']=task
(P/'measured-timeline.json').write_text(json.dumps(x,ensure_ascii=False,indent=2));print('Restored Stream ID / Task ID from exact original CSV rows; no trace recapture')
