"""Compare only six real duplicated checkpoint parameters, host-side and bounded."""
import json,sys
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,read,write
sys.path.insert(0,'/Users/wio/work/171-vllm-migration-20261102')
from control import invoke
old=read(O/'checkpoint-all-header-index-v159.json')
dups=old['duplicate_parameter_names'];names={x['shard'] for d in dups for x in d['occurrences']}
headers={x['shard']:x for x in old['headers'] if x['shard'] in names}
assert len(dups)==6 and len(headers)==4
prefix='DUPLICATES = '+repr(dups)+'\nHEADER_RECORDS = '+repr(headers)+'\nINDEX_META = '+repr(old['index_source'])+'\n'
code=prefix+r'''import json,os,socket,hashlib,time,stat,math,traceback
from pathlib import Path
from contextlib import ExitStack
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-171' and os.geteuid()==0
root=Path('/data/tiankuan/wio/DeepSeek-V4.1-Flash-1017-w8a8');out=Path('/data/tiankuan/wio/migration-171-20261102/operator-evidence-v160')
out.mkdir(mode=0o700,exist_ok=True);dest=out/'six-duplicate-parameter-values.json';assert not dest.exists(),'Do not repeat a sealed read;inspect existing receipt instead'
start=time.monotonic();deadline=start+60
def stable(s):return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
idx=root/'quant_model_weights.safetensors.index.json';b=idx.read_bytes()
assert len(b)==INDEX_META['bytes'] and hashlib.sha256(b).hexdigest()==INDEX_META['sha256']
result={'scope':'CURRENT_CHECKPOINT_BYTE_RANGES_ONLY','target':'172.16.10.171','model_directory':str(root),'source_header_cohort':'v159','rows':[],'bytes_read':0,'no_CANN_Torch_import_or_NPU_calls':True,'actual_loaded_or_historical_parameter_values_attested':False,'native_buffer_alias_or_fusion_legality_proved':False,'overall_complete':False}
try:
 with ExitStack() as stack:
  handles={};before={};bodies={};data_starts={}
  for name,row in HEADER_RECORDS.items():
   p=root/name;s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_size==row['source_file_bytes'] and s.st_mtime_ns==row['source_mtime_ns']
   fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW);f=stack.enter_context(os.fdopen(fd,'rb'));before[name]=stable(os.fstat(f.fileno()));assert before[name]==stable(s)
   headsize=int.from_bytes(f.read(8),'little');assert headsize==row['header_bytes'] and headsize<16*1024*1024
   raw=f.read(headsize);assert hashlib.sha256(raw).hexdigest()==row['header_sha256'];body=json.loads(raw)
   handles[name]=f;bodies[name]=body;data_starts[name]=8+headsize
  for dup in DUPLICATES:
   rows=dup['occurrences'];assert len(rows)==2;name=dup['name'];spans=[]
   for entry in rows:
    shard=entry['shard'];actual=bodies[shard][name]
    assert actual['shape']==entry['shape'] and actual['dtype']==entry['dtype']=='BF16' and actual['data_offsets']==entry['data_offsets']
    a,b=actual['data_offsets'];n=b-a;assert n==math.prod(actual['shape'])*2==entry['declared_payload_bytes']
    assert data_starts[shard]+b<=HEADER_RECORDS[shard]['source_file_bytes'];spans.append((shard,data_starts[shard]+a,n))
   assert spans[0][2]==spans[1][2];n=spans[0][2]
   f1,f2=(handles[s[0]] for s in spans)
   f1.seek(spans[0][1]);f2.seek(spans[1][1]);h1=hashlib.sha256();h2=hashlib.sha256();equal=True;first=None;pos=0
   while pos<n:
    assert time.monotonic()<deadline,'60-second bounded host-read deadline'
    amount=min(8*1024*1024,n-pos);x,y=f1.read(amount),f2.read(amount);assert len(x)==len(y)==amount
    h1.update(x);h2.update(y)
    if x!=y:
     equal=False
     if first is None:first=pos
    pos+=amount;result['bytes_read']+=2*amount
   result['rows'].append({'name':name,'dtype':'BF16','shape':rows[0]['shape'],'payload_bytes_each':n,'copies':[{'shard':spans[0][0],'absolute_file_offset':spans[0][1],'sha256':h1.hexdigest()},{'shard':spans[1][0],'absolute_file_offset':spans[1][1],'sha256':h2.hexdigest()}],'direct_bytewise_equal':equal,'first_different_block_start_if_any':first,'selected_index_shard':dup['index_declared_shard'],'actual_runtime_loader_selection_attested':False})
  for name,f in handles.items():assert stable(os.fstat(f.fileno()))==before[name]==stable((root/name).lstat()),('File changed',name)
 result.update(status='PASS_BOUNDED_CONTENT_COMPARISON',all_six_equal=all(r['direct_bytewise_equal'] for r in result['rows']),stable_file_stats_before_after=True,elapsed_seconds=time.monotonic()-start)
except BaseException:
 result.update(status='ERROR',error=traceback.format_exc(),elapsed_seconds=time.monotonic()-start)
dest.write_text(json.dumps(result,ensure_ascii=False,indent=2));result['server_receipt']=str(dest)
print(json.dumps(result,ensure_ascii=False))
'''
d=invoke('target',code,90)
write(O/'six-duplicate-parameter-values-v160.json',d)
print(json.dumps(d,ensure_ascii=False))
assert d['status']=='PASS_BOUNDED_CONTENT_COMPARISON'
