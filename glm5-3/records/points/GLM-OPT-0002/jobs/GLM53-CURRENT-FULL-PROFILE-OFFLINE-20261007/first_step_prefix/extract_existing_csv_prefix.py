from pathlib import Path
import csv,json,hashlib,collections,io,base64,gzip
j=Path('/data/tiankuan/wio/Inference-Foundry/glm5-3/records/points/GLM-OPT-0002/jobs/GLM53-CURRENT-FULL-PROFILE-OFFLINE-20261007')
results=[]
for rank in (0,13,15):
 identity=json.loads((j/'frontier'/('rank%d.json'%rank)).read_text())
 csvpath=Path(identity['path']).parent/'kernel_details.csv';raw=csvpath.read_bytes()
 expected=json.loads((j/'rope_geometry'/('rank%d.json'%rank)).read_text())['kernel_csv_sha256'];assert hashlib.sha256(raw).hexdigest()==expected
 first=float(identity['graph_execute'][0]['event']['ts']) if 'graph_execute' in identity else None
 if first is None:
  keys=[k for k in identity if isinstance(identity[k],list) and identity[k] and isinstance(identity[k][0],dict) and identity[k][0].get('event',{}).get('name')=='AscendCL@aclmdlRIExecuteAsync'];assert len(keys)==1,keys
  first=float(identity[keys[0]][0]['event']['ts'])
 rows=[];groups=collections.defaultdict(lambda:dict(count=0,duration_us=0.0,first_ts=None,last_ts=None));all_count=0
 cols=['Name','Model ID','Task ID','Stream ID','Start Time(us)','Duration(us)','Input Shapes','Output Shapes','Input Data Types','Output Data Types','Input Formats','Output Formats','Accelerator Core']
 for i,z in enumerate(csv.DictReader(io.StringIO(raw.decode())),2):
  all_count+=1;t=float(z['Start Time(us)'].strip())
  if not first-1000000 <= t < first:continue
  x={k:z[k] for k in cols if k in z};x['csv_line']=i;rows.append(x)
  key=(z['Name'],z['Model ID'],z['Stream ID'],z['Input Shapes'],z['Output Shapes'])
  g=groups[key];g['count']+=1;g['duration_us']+=float(z['Duration(us)']);g['first_ts']=min(g['first_ts'] or t,t);g['last_ts']=max(g['last_ts'] or t,t)
 grouped=[dict(name=k[0],model_id=k[1],stream_id=k[2],input_shapes=k[3],output_shapes=k[4],**v) for k,v in groups.items()]
 result=dict(rank=rank,trace_sha256=identity['sha256'],kernel_csv=str(csvpath),kernel_csv_sha256=expected,first_graph_host_ts=first,window_lower=first-1000000,all_csv_count=all_count,selected_count=len(rows),selected_rows=rows,groups=grouped,NPU_requests=0,new_profile=0,parser_rerun=0,limits='Existing processed device CSV prefix only; host/request/KV/output boundaries not supplied; one-second bound and all ranks not all32; durations are inclusive not savings')
 results.append(result)
print(base64.b64encode(gzip.compress(json.dumps(results).encode())).decode())
