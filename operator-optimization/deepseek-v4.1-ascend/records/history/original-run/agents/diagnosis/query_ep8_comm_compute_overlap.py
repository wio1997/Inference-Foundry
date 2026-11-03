import csv,json,hashlib
from decimal import Decimal
p='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces/dp1_pp0_tp0_dcp0_ep8_rank0_727_20261028020238580_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv'
lo,hi=1793152960223455898,1793152960261768012
sha=hashlib.sha256();
with open(p,'rb') as f:
 for b in iter(lambda:f.read(1048576),b''):sha.update(b)
assert sha.hexdigest()=='488d3e3736677d7d041edbe8307d3c06b79553362e54aef07aba37fd74229cc7'
rows=[]
with open(p,newline='') as f:
 for z in csv.DictReader(f):
  try:s=int(Decimal(z['Start Time(us)'].strip())*1000);e=s+int(Decimal(z['Duration(us)'].strip())*1000)
  except:continue
  if e<=lo or s>=hi or e<=s:continue
  if z['Stream ID'].strip()=='N/A' or z['Task ID'].strip()=='N/A':continue
  rows.append({'type':z['Type'],'device':z['Device_id'],'model':z['Model ID'].strip(),'stream':z['Stream ID'].strip(),'task':z['Task ID'].strip(),'core':z['Accelerator Core'],'s':s,'e':e})
comm=[z for z in rows if z['type'].startswith('hcom_')]
calc=[z for z in rows if z['core'] in ('AI_CORE','AI_VECTOR_CORE','MIX_AIC','MIX_AIV','AIC','AIV') and not z['type'].startswith('hcom_')]
pairs=[]
for c in comm:
 for k in calc:
  if c['device']!=k['device']:continue
  overlap=max(0,min(c['e'],k['e'],hi)-max(c['s'],k['s'],lo))
  if overlap>0:pairs.append((overlap,c,k))
pairs.sort(key=lambda x:-x[0])
def clean(z):return {**{k:z[k] for k in ('type','device','model','stream','task','core')},'s_rel_ns':z['s']-lo,'e_rel_ns':z['e']-lo}

def union(intervals):
 out=[]
 for a,b in sorted((max(lo,a),min(hi,b)) for a,b in intervals if max(lo,a)<min(hi,b)):
  if out and a<=out[-1][1]:out[-1]=(out[-1][0],max(out[-1][1],b))
  else:out.append((a,b))
 return sum(b-a for a,b in out)
assert all(z['device']=='8' for z in rows)
comm_u=union([(z['s'],z['e']) for z in comm]); calc_u=union([(z['s'],z['e']) for z in calc]); both_u=union([(z['s'],z['e']) for z in comm+calc]); assert comm_u+calc_u-both_u==0
example=comm[0]; preceding=[z for z in calc if z['stream']==example['stream'] and z['e']<=example['s']]; following=[z for z in calc if z['stream']==example['stream'] and z['s']>=example['e']]; before=max(preceding,key=lambda z:z['e']) if preceding else None;after=min(following,key=lambda z:z['s']) if following else None
print(json.dumps({'comm_streams':sorted({z['stream'] for z in comm}),'calc_streams':sorted({z['stream'] for z in calc}),'shared_streams':sorted({z['stream'] for z in comm}&{z['stream'] for z in calc}),'clipped_comm':sum(z['s']<lo or z['e']>hi for z in comm),'clipped_calc':sum(z['s']<lo or z['e']>hi for z in calc),'comm_union_ns':comm_u,'compute_union_ns':calc_u,'combined_union_ns':both_u,'comm_first':clean(example),'same_stream_prev':clean(before) if before else None,'same_stream_next':clean(after) if after else None,'sha256':sha.hexdigest(),'window_ns':hi-lo,'rows':len(rows),'comm_count':len(comm),'compute_count':len(calc),'core_counts':{v:sum(z['core']==v for z in rows) for v in sorted(set(z['core'] for z in rows))},'pairs_count':len(pairs),'top_pairs':[{'overlap_ns':o,'comm':clean(c),'compute':clean(k)} for o,c,k in pairs[:12]],'comm_sample':[clean(c) for c in comm[:5]]},indent=2))
