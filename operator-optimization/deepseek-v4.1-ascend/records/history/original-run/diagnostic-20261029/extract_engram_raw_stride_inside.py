"""Stream exact EP8 raw TLV op records. Only anonymous metadata is emitted.

Implementation follows installed TLVDecoder (<HI>, 58-byte TorchOp header),
TorchOpBean timestamp conversion and _event_tree_parser's TensorEnum.  The
launcher injects TARGETS from already-saved exact database audit results.
"""
import ast, configparser, hashlib, json, pathlib, re, struct
from collections import Counter
r=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binary-binding-draft/second-binding-run/offline-ep8-20261029/dp1_pp0_tp0_dcp0_ep8_rank0_746_20261029050352206_ascend_pt')
p=r/'FRAMEWORK/torch.op_range'; info=r/'profiler_info_8.json'
prof=list(r.glob('PROF_*/host'))
if len(prof)!=1: raise RuntimeError('ambiguous profiler host path')
host=prof[0];j=json.loads(info.read_text());start=j['start_info'];host_info=json.loads((host/'info.json').read_text());freq=float(host_info['CPU'][0]['Frequency']);clock=configparser.ConfigParser();clock.read(host/'host_start.log');time_offset=int(clock.get('Host','clock_monotonic_raw'));start_cnt=int(clock.get('Host','cntvct'));cann=ast.literal_eval((host/'start_info').read_text());local_diff=int(cann['collectionTimeBegin'])*1000-int(cann['clockMonotonicRaw'])
if start['syscnt_enable'] is not True or freq != 100.0:raise RuntimeError('unexpected clock mapping')
def timestamp(cnt):return int((cnt-start_cnt)*(1000.0/freq))+time_offset+local_diff
keys={(x['name'],x['start_ns'],x['end_ns']):x for x in TARGETS}
if len(keys)!=12:raise RuntimeError('anchors not unique')
start_set={x['start_ns'] for x in TARGETS}; matches={}; seen=0; size_total=0; hash_=hashlib.sha256();before=p.stat();impl_labels={};storage_labels={}
def label(mapping,value,prefix):
 if not value:return None
 if value not in mapping:mapping[value]=prefix+str(len(mapping)+1)
 return mapping[value]
def metadata(typestr,tensorstr):
 types=typestr.replace(';',';\r\n').strip().split(';\r\n') if typestr else []
 chunks=tensorstr.strip().split(')') if tensorstr else []
 index=0;result=[]
 for i,t in enumerate(types):
  if t in ('None','Scalar','ScalarList','TensorList',''):continue
  raw=chunks[index] if index<len(chunks) else '';index+=1
  parts=raw.split(';')
  if len(parts)!=8:
   result.append({'input_index':i,'status':'unavailable_or_malformed'});continue
  try:
   sizes=[int(v) for v in parts[4].split(',')] if parts[4] else []
   strides=[int(v) for v in parts[5].split(',')] if parts[5] else []
   dtype_size=int(parts[3]);device_type=int(parts[6]);device_index=int(parts[7]);
   if len(sizes)!=len(strides):raise ValueError('different rank')
   result.append({'input_index':i,'status':'available','dtype':parts[2],'element_size':dtype_size,'shape':sizes,'strides':strides,'device_type':device_type,'device_index':device_index,'tensor_impl_label':label(impl_labels,parts[0],'I'),'storage_label':label(storage_labels,parts[1],'S')})
  except (TypeError,ValueError):result.append({'input_index':i,'status':'unavailable_or_malformed'})
 return result
with p.open('rb') as f:
 while True:
  hdr=f.read(6)
  if not hdr:break
  if len(hdr)!=6:raise RuntimeError('truncated outer header')
  typ,length=struct.unpack('<HI',hdr)
  if length<58 or length>16_000_000:raise RuntimeError('invalid outer record length')
  value=f.read(length)
  if len(value)!=length:raise RuntimeError('truncated outer payload')
  hash_.update(hdr);hash_.update(value);size_total+=len(hdr)+length;seen+=1
  start_raw,end_raw,seq,pid,tid,endtid,forwardtid,scope,is_async=struct.unpack('<3q4QB?',value[:58])
  if pid!=746 or tid!=746:continue
  st=timestamp(start_raw)
  if st not in start_set:continue
  end=timestamp(end_raw);fields={};pos=58
  while pos<len(value):
   if pos+6>len(value):raise RuntimeError('truncated nested header')
   t,n=struct.unpack('<HI',value[pos:pos+6]);pos+=6
   if pos+n>len(value):raise RuntimeError('truncated nested payload')
   if t in (2,3,4,5,8):fields[t]=value[pos:pos+n].decode('utf8')
   pos+=n
  name=fields.get(2,'');key=(name,st,end)
  if key not in keys:continue
  target=keys[key];stack=fields.get(8,'');required=target['stack_substring']
  norm=lambda x:re.sub(r'\s+','',x)
  if required not in stack or norm(fields.get(4,''))!=norm(target['shape']):raise RuntimeError('exact time matched but stack/shape mismatch')
  if key in matches:raise RuntimeError('duplicate exact record')
  matches[key]={'label':target['label'],'name':name,'start_ns':st,'end_ns':end,'pid':pid,'tid':tid,'shape_match':True,'stack_match':True,'field5_present':5 in fields,'inputs':metadata(fields.get(3,''),fields.get(5,'')) if 5 in fields else []}
after=p.stat()
if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or size_total!=before.st_size:raise RuntimeError('raw file changed during read')
ordered=[matches[k] for k in keys if k in matches]
print(json.dumps({'scope':'12 exact select/contiguous/copy events; source profiler raw metadata only','raw_path':str(p),'raw_size':before.st_size,'raw_sha256':hash_.hexdigest(),'raw_mtime_ns':before.st_mtime_ns,'record_count':seen,'file_stable_during_read':True,'db_expected_sha256':'95b58de883ee272b7161bce7b213d15673c1925fd165bafbf4deacf7a6b74b23','clock':{'freq':freq,'ratio':1000/freq,'source':'host/info.json,host_start.log,host/start_info; installed ProfilerConfig formula'},'target_count':len(TARGETS),'matched_count':len(ordered),'missing_labels':[x['label'] for k,x in keys.items() if k not in matches],'matched':ordered,'limitations':['I/S labels are same-run equality only, raw pointer addresses are never emitted.','Storage pointer is not per-view data_ptr; no storage_offset field in decoded structure.','profile_memory=false means no allocator lifetime proof; identity/alias does not prove read ordering.','timestamp match links exported host records, not device completion or performance.']},indent=2))
