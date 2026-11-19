"""Cross-prompt same-BOS layer0 comparison; reuses actual G1104 tensors on server CPU."""
from pathlib import Path
import datetime,hashlib,json,time
import torch
r=Path('/task');out=r/'client_ds41_g503/CPU_attention_reference_g1100/actual_G1104_BOS_alllayers_g1119.json';assert not out.exists();torch.set_num_threads(4);start=time.monotonic();folder=r/'sg_pd_g1098/forward_probe/G1104'
groups={}
for p in folder.glob('*.pt'):
 d=torch.load(p,map_location='cpu',weights_only=True)
 if d['positions'][0].item()==0 and max(d['positions'].tolist())>0:groups.setdefault((d['pid'],d['ratio'],d['layer_id']),[]).append((p,d))
rows=[]
for pid,values in sorted(groups.items()):
 assert len(values)==2;values.sort(key=lambda x:x[1]['call_in_ratio']);(pa,a),(pb,b)=values
 assert a['input_ids'][0].item()==b['input_ids'][0].item()==0 and a['positions'][0].item()==b['positions'][0].item()==0
 def k0(x):
  slots=x['original_slots'].tolist();loc=x['original_indices'][0];v=loc[loc>=0];assert len(v)==1;return x['original_values'][slots.index(v.item())]
 metrics={}
 for name,x,y in [('query',a['q'][0],b['q'][0]),('original_BOS_KV',k0(a),k0(b)),('attention_output',a['result'][0],b['result'][0])]:
  x=x.float();y=y.float();d=(x-y).abs();metrics[name]={'elements':x.numel(),'exact_numeric_elements':int((x==y).sum()),'max_abs':d.max().item(),'RMSE':d.square().mean().sqrt().item(),'relative_L2':((x-y).norm()/torch.maximum(x.norm(),torch.tensor(1e-30))).item(),'a_L2':x.norm().item(),'b_L2':y.norm().item()}
 rows.append({'pid':a['pid'],'ratio':a['ratio'],'layer_id':a['layer_id'],'a_file':pa.name,'b_file':pb.name,'a_shape':a['full_q_shape'],'b_shape':b['full_q_shape'],'metrics':metrics})
assert len(rows)==24
v={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-start,'rows':rows,'scope':'SameactualBOS0/position0/selectedlayers0,2,20, differentwholepromptandpadding32vs40; numericalcomparisononly. Batchprecision/embedding/mHC/projection can differ; not solekernelrootcause orqualityclaim.','source_SHA256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};out.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({'ranks':len(rows),'max_KV_diff':max(x['metrics']['original_BOS_KV']['max_abs'] for x in rows),'max_query_diff':max(x['metrics']['query']['max_abs'] for x in rows)}),flush=True)
