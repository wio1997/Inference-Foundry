"""CPU stdlib-only safetensor SHA inventory; no tensor simulation or NPU calls."""
from pathlib import Path
import hashlib,json,struct,sys
root=Path(sys.argv[1]);nonce=sys.argv[2];folder=root/'layer_probe'/nonce
records=[]
for p in sorted(folder.glob('rank*_layer*.safetensors')):
 meta=json.loads(p.with_suffix('.json').read_text());b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==meta['file_SHA256'];n=struct.unpack('<Q',b[:8])[0];header=json.loads(b[8:8+n]);body=b[8+n:];out={}
 for name in ['input','output','positions']:
  e=header[name];lo,hi=e['data_offsets'];out[name]={'dtype':e['dtype'],'shape':e['shape'],'sha256':hashlib.sha256(body[lo:hi]).hexdigest()}
 records.append({'file':p.name,'rank':meta['rank'],'layer':meta['layer'],'file_SHA256':meta['file_SHA256'],'bytes':len(b),'output_is_TP_partial':meta['output_is_TP_partial'],'tensors':out,'positions':meta['positions'],'input_ids':meta['input_ids']})
assert len(records)==320 and {r['rank'] for r in records}==set(range(8)) and {r['layer'] for r in records}==set(range(40))
rank_equal=[]
for layer in range(40):
 values=[r for r in records if r['layer']==layer];assert len(values)==8
 rank_equal.append({'layer':layer,'input_distinct_raw_BF16_SHA':len({r['tensors']['input']['sha256'] for r in values}),'output_distinct_raw_BF16_SHA':len({r['tensors']['output']['sha256'] for r in values})})
result={'files':records,'all8_actual_ranks_40layers':True,'within_rank_distinct_SHA':rank_equal,'bytes':sum(r['bytes'] for r in records),'scope':'SHA inventory only, no numerical tolerance or modelqualityacceptance. TP-partialoutput marker preserved. Allvectorsremainserver except boundedrank0 forCPUcomparison.'};p=folder/'CPU_safetensors_inventory.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'inventory':str(p),'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':result['bytes'],'rank_equal':rank_equal}))
