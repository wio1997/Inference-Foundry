#!/usr/bin/env python3
"""Follow selected registry/arch22 branches from pinned entry code, read-only."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_prefix_retirement_identity import OUT,REMOTE
if __name__=='__main__':
 repo='/vllm-workspace/vllm-ascend/';rope=repo+'csrc/attention/inplace_partial_rotary_mul/op_kernel/';scatter=repo+'csrc/moe/scatter_nd_update_sk/op_kernel/arch22/';cann='/usr/local/Ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/'
 names=[repo+'vllm_ascend/utils.py',repo+'vllm_ascend/platform.py',repo+'csrc/attention/inplace_partial_rotary_mul/op_host/inplace_partial_rotary_mul_tiling.h']
 names += [rope+n for n in ['rotate_interleaved_split_s.h','rotate_interleaved_split_bs.h','rotate_interleaved_split_bsn.h','apply_rotary_pos_emb_common.h']]
 names += [scatter+n for n in ['scatter_nd_update_hp.h','scatter_nd_update_common.h','scatter_nd_update_large_index.h']]
 names += [cann+n for n in ['rms_norm/rms_norm_single_row.h','rms_norm/rms_norm_merge_n.h','rms_norm/rms_norm_whole_reduce_sum.h','rms_norm/reduce_common.h','dynamic_quant/dynamic_quant_single_row.h','dynamic_quant/dynamic_quant_multi_row.h']]
 spec=[dict(runtime_path=n,local_sha256=None) for n in names];r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE.replace("'python3','-I','-S'","'python3','-B','-I','-S'"),json.dumps(spec)],capture_output=True,text=True,timeout=120)
 if r.returncode:raise RuntimeError(r.stderr)
 d=json.loads(r.stdout)
 for row in d['sources']:
  body=row.pop('content',None)
  if body is not None:
   p=OUT/'cache-epilogue-selected-source'/row['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body);assert sha256(p.read_bytes()).hexdigest()==row['runtime_sha256'];row['local_path']=str(p)
 d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),scope='Source snapshots only;selectedcall/tiling/body ranges separate, not everytemplatequalified.');(OUT/'cache-epilogue-selected-sources.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(requested=len(spec),received=sum(not r.get('missing') for r in d['sources']),missing=[r['runtime_path'] for r in d['sources'] if r.get('missing')])))
