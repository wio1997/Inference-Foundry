#!/usr/bin/env python3
"""Check local frozen model dimensions, Engram index keys and quantization labels."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[2]
M=Path('/Users/wio/work/169-vllm-ascend-reproduce-172/source172/model-metadata')
paths={'config':R/'evidence/model-config.json', 'reference_index':M/'quant_model_weights.safetensors.index.json',
 'quant_description':M/'quant_model_description.json',
 'engram_layer':R/'sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram/layer.py',
 'engram_gate':R/'sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram/common.py'}
hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
config=json.loads(paths['config'].read_text())['text_config'];w=json.loads(paths['reference_index'].read_text())['weight_map'];q=json.loads(paths['quant_description'].read_text())
rows=[{'layer':i,'key':f'layers.{i}.engram.wkv.weight',
       'indexed_shard':w.get(f'layers.{i}.engram.wkv.weight'),
       'quant_description_label':q.get(f'layers.{i}.engram.wkv.weight')} for i in (1,14)]
N=(config['engram_max_ngram_size']-1)*config['engram_n_heads']*config['engram_head_dim']
K=(config['hc_mult']+1)*config['hidden_size']
assert (N,K)==(6144,25600) and all(v['indexed_shard'] and v['quant_description_label']=='FLOAT' for v in rows)
print(json.dumps({'source_sha256':hashes,'logical_projection_input_dim':N,'logical_projection_output_dim':K,
 'index_mapping':rows,'index_limits':['local reference index is name to shard, not tensor shapes or dtype',
 '169 index bytes and historical process object are not independently proved by this script',
 'FLOAT label is not BF16 dtype proof','no device or service actions']},indent=2))
