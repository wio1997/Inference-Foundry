#!/usr/bin/env python3
"""Read only fixed checkpoint index and 8-byte+JSON headers; never read tensor payload."""
import hashlib,json,os,stat,struct
from pathlib import Path
ROOT=Path('/data/tiankuan/wio/DeepSeek-V4.1-Flash-1017-w8a8')
INDEX='quant_model_weights.safetensors.index.json'
TARGETS={'layers.1.engram.wkv.weight':'quant_model_weights-engram-extra-1.safetensors',
         'layers.14.engram.wkv.weight':'quant_model_weights-engram-extra-14.safetensors'}
MAX_INDEX=32*1024*1024
MAX_HEADER=16*1024*1024
if ROOT.is_symlink() or ROOT.resolve(strict=True)!=ROOT:raise RuntimeError('model root resolves outside fixed path or is symlink')
def open_fixed(name):
 if Path(name).name!=name or name not in {INDEX,*TARGETS.values()}:raise RuntimeError('filename not allowed')
 p=ROOT/name
 fd=os.open(str(p),os.O_RDONLY | getattr(os,'O_NOFOLLOW',0))
 st=os.fstat(fd)
 if not stat.S_ISREG(st.st_mode):os.close(fd);raise RuntimeError('not a regular file')
 return fd,st

def read_exact(fd,n):
 parts=[]
 while n:
  b=os.read(fd,n)
  if not b:raise RuntimeError('unexpected EOF')
  parts.append(b);n-=len(b)
 return b''.join(parts)
def unchanged(fd,st):
 t=os.fstat(fd)
 if (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)!=(t.st_dev,t.st_ino,t.st_size,t.st_mtime_ns):raise RuntimeError('file changed while reading')
fd,st=open_fixed(INDEX)
try:
 if st.st_size>MAX_INDEX:raise RuntimeError('index exceeds 32 MiB cap')
 b=read_exact(fd,st.st_size);unchanged(fd,st)
finally:os.close(fd)
index=json.loads(b);mapping=index.get('weight_map');assert isinstance(mapping,dict)
for key,shard in TARGETS.items():
 if mapping.get(key)!=shard:raise RuntimeError('checkpoint index target missing or does not match allowlisted shard: '+key)
rows=[]
for key,shard in TARGETS.items():
 fd,st=open_fixed(shard)
 try:
  if st.st_size<8:raise RuntimeError('truncated safetensors file')
  prefix=read_exact(fd,8);n=struct.unpack('<Q',prefix)[0]
  if n==0 or n>MAX_HEADER or 8+n>st.st_size:raise RuntimeError('invalid/excessive safetensors header length')
  hb=read_exact(fd,n);unchanged(fd,st)
 finally:os.close(fd)
 header=json.loads(hb);obj=header.get(key)
 if not isinstance(obj,dict):raise RuntimeError('indexed tensor not found in header: '+key)
 dtype=obj.get('dtype');shape=obj.get('shape');offset=obj.get('data_offsets')
 if not isinstance(dtype,str) or not isinstance(shape,list) or not all(type(d)==int and d>=0 for d in shape):raise RuntimeError('invalid tensor header')
 if not isinstance(offset,list) or len(offset)!=2 or any(type(v)!=int for v in offset) or not 0<=offset[0]<=offset[1]<=st.st_size-8-n:raise RuntimeError('invalid tensor offsets')
 rows.append({'key':key,'shard':shard,'dtype':dtype,'shape':shape,'tensor_payload_bytes_declared':offset[1]-offset[0],
              'shard_size':st.st_size,'header_json_bytes':n,'header_sha256':hashlib.sha256(hb).hexdigest(),
              'matches_profiler_weight_operand_shape_and_dtype':shape==[25600,6144] and dtype=='BF16'})
print(json.dumps({'scope':'169 checkpoint index plus safetensors headers only',
 'index_size':len(b),'index_sha256':hashlib.sha256(b).hexdigest(),'tensors':rows,
 'payload_bytes_read':0,'no_npu_work':True,
 'limits':['Current on-disk checkpoint header and name mapping, not historical loaded tensor, storage identity, tensor values, source/binary or task binding.',
 'Matching shape and dtype alone does not identify trace MatMulV2 weight.']},ensure_ascii=False,indent=2))
