from pathlib import Path
import csv,json,gzip,hashlib
j=Path('/data/tiankuan/wio/Inference-Foundry/glm5-3/records/points/GLM-OPT-0002/jobs/GLM53-CURRENT-FULL-PROFILE-OFFLINE-20261007')
d=j/'rope_geometry';assert not d.exists();d.mkdir()
out=[]
for rank in (0,13,15):
 r=json.loads((j/'frontier'/('rank%d.json'%rank)).read_text());csvpath=Path(r['path']).parent/'kernel_details.csv';raw=csvpath.read_bytes();rows=list(csv.DictReader(raw.decode().splitlines())); selected=[]
 for ordinal in (2,3):
  ev=json.loads(gzip.decompress((j/'frontier'/('rank%d_period%d_events.json.gz'%(rank,ordinal))).read_bytes()))
  for x in ev:
   z=x['event']
   if z['name']!='aclnnIndex_SliceAiCore_Slice':continue
   a=z['args']; match=[v for v in rows if v['Task ID']==str(a['Task Id']) and v['Stream ID']==str(a['Physic Stream Id']) and v['Model ID']==str(a['Model Id']) and abs(float(v['Start Time(us)'].strip())-float(z['ts']))<1]
   assert len(match)==1
   selected.append(dict(period=ordinal,original_event_index=x['index'],raw_event=z,kernel_row=match[0]))
 assert len(selected)==8
 result=dict(rank=rank,trace_sha256=r['sha256'],kernel_csv=str(csvpath),kernel_csv_sha256=hashlib.sha256(raw).hexdigest(),selected=selected,inference_requests=0,new_profile=0)
 (d/('rank%d.json'%rank)).write_text(json.dumps(result,indent=2)+'\n');out.append(dict(rank=rank,selected=len(selected),kernel_csv_sha256=result['kernel_csv_sha256']))
(d/'index.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
