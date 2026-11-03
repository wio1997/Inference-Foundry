"""Whole-profile operator/callsite catalog from exact CPU/native relations."""
import collections,json,time
from pathlib import Path
O=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/joint169');idx=json.loads((O/'PROFILE-SOURCE-SYMBOL-INDEX.json').read_text())['source_frame_index'];catalog={};pads=[];f21_counts=collections.Counter();f21_cpu_names=collections.Counter()
for p in sorted((O/'startup-creators').glob('*_ascend_pt.json')):
 d=json.loads(p.read_text());cpu=d['CPU_rows'];chains=d['callchains'];groups=collections.defaultdict(list)
 for x in d['executions']:
  assert len(x['CPU_rows'])==1;groups[x['CPU_rows'][0]].append(x)
 for rowid,ops in groups.items():
  c=cpu[str(rowid)];chain=chains.get(str(c['callchain_id']),[]);linked=[idx[a['text']] for a in chain if a['text'] in idx];models=[a for a in linked if any(k in a['path'] for k in ['/models/','/attention/','/spec_decode/'])];anchor=models[0] if models else (linked[0] if linked else None)
  names=tuple(sorted(x['native_name'] for x in ops));key=(c['name'],names,json.dumps(anchor,sort_keys=True));r=catalog.setdefault(key,{'CPU_operator':c['name'],'native_kernel_multiset':list(names),'source_anchor':anchor,'calls':0,'native_executions':0,'native_work_ns':0,'native_shape_pairs':set()});r['calls']+=1;r['native_executions']+=len(ops);r['native_work_ns']+=sum(x['duration_ns'] for x in ops);r['native_shape_pairs'].update((x['native_name'],x['input_shapes'],x['output_shapes']) for x in ops)
  nf=sum(x['native_name']=='DispatchFFNCombine' for x in ops)
  if nf:f21_counts[nf]+=1;f21_cpu_names[c['name']]+=1
  if 'ConstantPadNd' in c['name']:
   samples=[x for x in ops if x['input_shapes'].strip('"')=='20' and x['output_shapes'].strip('"')=='24']
   if samples:pads.append({'profile':p.stem,'CPU_ROW':rowid,'CPU_operator':c['name'],'native_kernel_multiset':list(names),'source_anchor':anchor,'native_shapes':[{'name':x['native_name'],'input':x['input_shapes'],'output':x['output_shapes']} for x in ops],'native_work_ns':sum(x['duration_ns'] for x in ops)})
rows=sorted(catalog.values(),key=lambda r:r['native_work_ns'],reverse=True)
for r in rows:r['native_shape_pairs']=[list(x) for x in sorted(r['native_shape_pairs'],key=str)]
result={'status':'EXACT_STARTUP_OPERATOR_CALLSITE_CATALOG','generated_unix':time.time(),'operator_callsite_groups':len(rows),'native_compute_executions':sum(r['native_executions'] for r in rows),'CPU_operator_calls':sum(r['calls'] for r in rows),'F21_native_kernels_per_exact_CPU_call':dict(f21_counts),'F21_CPU_operator_names':dict(f21_cpu_names),'BOOL20_to24_ConstantPadNd_groups':pads,'catalog':rows,'scope':'Startup warmup/creation source domain. A group is exactCPUoperator+nativekernelmultiset+profile-driven modelcallsite;not a Cartesian operator pair or formalC8 attribution. Worksum is not criticalpathlatency.','whole_modules_closed':0,'overall_complete':False};(O/'OPERATOR-CALLSITE-CATALOG.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['catalog','BOOL20_to24_ConstantPadNd_groups']}));print(json.dumps({'pad_samples':pads[:2]}))
