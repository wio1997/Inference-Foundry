from pathlib import Path
import csv,json,gzip,hashlib,collections
j=Path('/data/tiankuan/wio/Inference-Foundry/glm5-3/records/points/GLM-OPT-0002/jobs/GLM53-CURRENT-FULL-PROFILE-OFFLINE-20261007')
d=j/'mc2_pad_geometry';assert not d.exists();d.mkdir()
summary=[]
for rank in (0,13,15):
 identity=json.loads((j/'frontier'/('rank%d.json'%rank)).read_text())
 csvpath=Path(identity['path']).parent/'kernel_details.csv';raw=csvpath.read_bytes()
 expected=json.loads((j/'rope_geometry'/('rank%d.json'%rank)).read_text())['kernel_csv_sha256']
 assert hashlib.sha256(raw).hexdigest()==expected
 rows=list(csv.DictReader(raw.decode().splitlines()));index=collections.defaultdict(list)
 for z in rows:index[(z['Task ID'],z['Stream ID'],z['Model ID'])].append(z)
 event_path=j/'frontier'/('rank%d_period3_events.json.gz'%rank)
 events=json.loads(gzip.decompress(event_path.read_bytes()));selected=[]
 for x in events:
  z=x['event'];name=z['name']
  if name not in ('aclnnConstantPadNd_PadV3AiCore_PadV3','aclnnConstantPadNd_PadV3AiCore_MemSet'):continue
  a=z['args'];matches=[v for v in index[(str(a['Task Id']),str(a['Physic Stream Id']),str(a['Model Id']))] if abs(float(v['Start Time(us)'].strip())-float(z['ts']))<1]
  assert len(matches)==1,(rank,x['index'],len(matches))
  selected.append(dict(original_event_index=x['index'],raw_event=z,kernel_row=matches[0]))
 assert len(selected)==304
 groups=collections.defaultdict(lambda:dict(count=0,duration_us=0.0))
 for x in selected:
  z=x['raw_event'];k=x['kernel_row'];a=z['args'];key=(z['name'],a['Model Id'],a['Physic Stream Id'],k['Input Shapes'],k['Output Shapes'],k['Input Data Types'],k['Output Data Types'])
  groups[key]['count']+=1;groups[key]['duration_us']+=float(z['dur'])
 grouped=[dict(name=k[0],model_id=k[1],stream_id=k[2],input_shapes=k[3],output_shapes=k[4],input_dtypes=k[5],output_dtypes=k[6],**v) for k,v in groups.items()]
 result=dict(rank=rank,ordinal=3,trace_sha256=identity['sha256'],event_export_sha256=hashlib.sha256(event_path.read_bytes()).hexdigest(),kernel_csv=str(csvpath),kernel_csv_sha256=expected,selected=selected,groups=grouped,NPU_requests=0,new_profile=0,parser_rerun=0,limits='Observed task-inclusive cost/geometry; capture-time Python stack absent; not savings or additive critical-path budget')
 (d/('rank%d.json'%rank)).write_text(json.dumps(result,indent=2)+'\n');summary.append({k:v for k,v in result.items() if k!='selected'})
(d/'index.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
