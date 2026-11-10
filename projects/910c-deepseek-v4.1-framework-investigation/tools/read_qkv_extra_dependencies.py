#!/usr/bin/env python3
"""Read exact compiled option catalog and quant method selectors, no framework import."""
from hashlib import sha256
from pathlib import Path
import json,subprocess
from read_qkv_native_catalog import REMOTE

INNER=r'''
from pathlib import Path
from hashlib import sha256
import json
root=Path('/vllm-workspace/vllm-ascend/vllm_ascend/quantization');source=[];searched=0
for p in root.rglob('*.py'):
 searched+=1;b=p.read_bytes()
 if len(b)>1000000:continue
 text=b.decode()
 if 'def get_quant_method(' in text or 'quant_model_description.json' in text:
  source.append(dict(runtime_path=str(p),sha256=sha256(b).hexdigest(),content=text))
kernel=Path('/usr/local/Ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe/kernel/ascend910_93/ops_nn/quant_batch_matmul_v3')
for name in ['QuantBatchMatmulV3_ND_NZ_int8_int8_bf16_high_performance.json','QuantBatchMatmulV3_ND_NZ_int8_int8_fp32_bf16_high_performance.json']:
 p=kernel/name
 if not p.is_file():source.append(dict(runtime_path=str(p),missing=True));continue
 b=p.read_bytes();source.append(dict(runtime_path=str(p),sha256=sha256(b).hexdigest(),content=b.decode()))
p=Path('/data/tiankuan/wio/DeepSeek-V4.1-Flash-1017-w8a8/config.json');b=p.read_bytes();c=json.loads(b);t=c['text_config'];cfg=dict(path=str(p),sha256=sha256(b).hexdigest(),selected_text_config={k:t[k] for k in ['num_hidden_layers','hidden_size','num_attention_heads','head_dim','q_lora_rank','rms_norm_eps','compress_ratios','kv_source_layer_ids','index_source_layer_ids','num_nextn_predict_layers','dspark_block_size']},model_dtype=c.get('dtype'),quantization_config=c.get('quantization_config'))
print(json.dumps(dict(sources=source,selector_pyfiles_searched=searched,model_config=cfg,scope='Compiled option list not active invocation tiling; model metadata no payload.')))
'''
if __name__=='__main__':
 r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,INNER],capture_output=True,text=True,timeout=120)
 if r.returncode:raise RuntimeError(r.stderr)
 d=json.loads(r.stdout);out=Path('audit/execution-search')
 for row in d['sources']:
  c=row.pop('content',None)
  if c is not None:
   p=out/'qkv-extra-source'/row['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(c);assert sha256(p.read_bytes()).hexdigest()==row['sha256'];row['local_path']=str(p)
 d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),new_NPU_work=0,new_profile=0,model_loads=0,service_requests=0,remote_writes=[]);(out/'qkv-extra-dependencies.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=[r['runtime_path'] for r in d['sources']],config=d['model_config'])))
