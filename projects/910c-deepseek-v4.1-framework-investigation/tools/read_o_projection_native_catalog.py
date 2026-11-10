#!/usr/bin/env python3
"""Inventory installed O-projection native sources and weight headers, stdlib only."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_qkv_native_catalog import REMOTE

INNER=r'''
from pathlib import Path
from hashlib import sha256
import json,re,struct
root=Path('/usr/local/Ascend/cann-9.2.0/opp/built-in/op_impl');names=[]
tokens=['transpose_batch_mat','/mat_mul_v2/','/mat_mul_v3/','/mat_mul/','/matmul/','mm_reduce_scatter','matmul_reduce_scatter']
for p in root.rglob('*'):
 if p.is_file() and p.suffix in ['.h','.hpp','.cpp','.py','.json'] and (any(n in str(p).lower() for n in tokens) or p.suffix=='.py' and ('mat_mul' in p.name or 'matmul' in p.name)):names.append(dict(path=str(p),bytes=p.stat().st_size))
site=Path('/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu');symbols=[];count=0;skipped=[]
needles=['npu_transpose_batchmatmul','aclnnTransposeBatchMatMul','npu_quant_mm_reduce_scatter','npu_mm_reduce_scatter_base','aclnnQuantMatmulReduceScatter','aclnnMatmul']
for p in site.rglob('*'):
 if not p.is_file() or p.suffix not in ['.h','.hpp','.cpp','.yaml','.yml']:continue
 if p.stat().st_size>2000000:skipped.append(str(p));continue
 count+=1;b=p.read_bytes();hits=[dict(line=i,text=l[:900]) for i,l in enumerate(b.decode(errors='replace').splitlines(),1) if any(n in l for n in needles)]
 if hits:symbols.append(dict(path=str(p),sha256=sha256(b).hexdigest(),hits=hits))
model=Path('/data/tiankuan/wio/DeepSeek-V4.1-Flash-1017-w8a8');configs=[]
for p in sorted(model.glob('*.json')):
 if p.name!='config.json' and 'quant' not in p.name:continue
 b=p.read_bytes();assert len(b)<32*1024*1024;d=json.loads(b)
 if p.name=='config.json':
  text=d.get('text_config',d);sel={k:text.get(k) for k in ['model_type','num_hidden_layers','hidden_size','num_attention_heads','head_dim','o_lora_rank','n_groups','quantization_config','torch_dtype']}
 else:sel={k:v for k,v in d.items() if any(x in k for x in ['wo_a','wo_b']) or k in ['model_quant_type','kv_cache_type','fa_quant_type','group_size']}
 configs.append(dict(path=str(p),sha256=sha256(b).hexdigest(),bytes=len(b),selected=sel))
headers=[];tensors=[]
for p in sorted(model.glob('*.safetensors')):
 with p.open('rb') as f:
  first=f.read(8);assert len(first)==8;n=struct.unpack('<Q',first)[0];assert 0<n<128*1024*1024;raw=f.read(n);assert len(raw)==n
 d=json.loads(raw);hits={k:v for k,v in d.items() if re.search(r'\.(wo_a|wo_b)\.(weight|weight_scale|weight_offset|input_scale|input_offset)$',k)}
 headers.append(dict(path=str(p),header_bytes=n,header_sha256=sha256(first+raw).hexdigest(),selected_tensors=len(hits)))
 for k,v in hits.items():tensors.append(dict(name=k,header_file=str(p),dtype=v['dtype'],shape=v['shape'],data_offsets=v['data_offsets']))
print(json.dumps(dict(native_names=names,torch_npu_symbol_files=symbols,symbol_source_files_searched=count,symbol_sources_skipped_large=skipped,configs=configs,safetensors_headers=headers,projection_tensors=tensors,payload_bytes_read=0,scope='Filename inventory/declarations and frozen OA/OB checkpoint metadata only; not loadedbinary/key or numerical proof.')))
'''
if __name__=='__main__':
 r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,INNER],capture_output=True,text=True,timeout=120)
 if r.returncode:raise RuntimeError(r.stderr)
 d=json.loads(r.stdout);d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),new_NPU_work=0,model_loads=0,new_profile=0,service_requests=0,remote_writes=[]);p=Path('audit/execution-search/o-projection-native-catalog.json');p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(native_names=len(d['native_names']),declaration_files=len(d['torch_npu_symbol_files']),headers=len(d['safetensors_headers']),projection_tensors=len(d['projection_tensors']),payload_bytes_read=0)))
