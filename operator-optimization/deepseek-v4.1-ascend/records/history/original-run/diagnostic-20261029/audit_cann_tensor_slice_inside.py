"""Decode one saved CANN tensor-info slice using the installed bean, aggregate only."""
import collections,hashlib,json,pathlib,struct,sys
r=pathlib.Path('/usr/local/Ascend/cann-9.2.0/tools/profiler/profiler_tool/analysis');sys.path.insert(0,str(r))
from msparser.add_info.tensor_add_info_bean import TensorAddInfoBean
from msparser.data_struct_size_constant import StructFmt
from common_func.ms_constant.number_constant import NumberConstant
from common_func.ms_constant.level_type_constant import LevelDataType
p=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binary-binding-draft/second-binding-run/offline-ep8-20261029/dp1_pp0_tp0_dcp0_ep8_rank0_746_20261029050352206_ascend_pt/PROF_000001_20261029050352216_00000746GNOJENRN/host/data/aging.additional.tensor_info.slice_0')
before=p.stat();b=p.read_bytes();after=p.stat();sha=hashlib.sha256(b).hexdigest()
if sha!='a12df64fb3934da72c2d5ded8b7c17a52ec252bdeb5f95b35b2dfe76a1a3dd88':raise RuntimeError('raw slice hash mismatch')
if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('raw file changed')
size=StructFmt.TENSOR_ADD_INFO_SIZE
if size!=256 or len(b)%size:raise RuntimeError('record size or alignment mismatch')
levels=collections.Counter();nums=collections.Counter();bad_magic=bad_level=bad_count=bad_decode=0;worker_tid_matches=0
for i in range(0,len(b),size):
 chunk=b[i:i+size];magic,level=struct.unpack('=HH',chunk[:4])
 if magic!=NumberConstant.MAGIC_NUM:bad_magic+=1;continue
 try:LevelDataType(level)
 except ValueError:bad_level+=1;continue
 try:
  bean=TensorAddInfoBean.decode(chunk)
  if not 0<=bean.tensor_num<=5:bad_count+=1;continue
  levels[bean.level]+=1;nums[str(bean.tensor_num)]+=1
  if bean.thread_id==6857:worker_tid_matches+=1
 except Exception:bad_decode+=1
print(json.dumps({'scope':'one verified original 2 MiB CANN tensor_info slice; installed Bean.decode only; no save, no values','path':str(p),'sha256':sha,'size':len(b),'record_size':size,'total_records':len(b)//size,'magic_fail':bad_magic,'level_fail':bad_level,'tensor_count_fail':bad_count,'decode_fail':bad_decode,'level_counts':dict(levels),'tensor_count_distribution':dict(nums),'matching_worker_tid_records':worker_tid_matches,'limits':['No raw tensor shapes, content, pointers, node IDs or individual timestamps are emitted.','Matching thread alone is not a call identity; no target-correlation claim.','Bean exposes only input/output format, dtype and shape, not stride, storage offset or storage ID.','First aging slice only; other slices and opaque alternate encodings are not characterized.']},indent=2))
