#!/usr/bin/env python3
"""Install/restore temporary W8A8 linear call-site probe for Run132."""
import argparse,hashlib,json
from pathlib import Path
SOURCE=Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/quantization/methods/w8a8_dynamic.py')
BACKUP=Path('/tmp/extreme_run132_w8a8_dynamic.orig')
ANCHOR='        quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(x, dst_type=self.act_quant_type)\n'
MARKER='        # EXTREME_RUN132_QUANT_CALL_PROBE\n'
INSERT='''        # EXTREME_RUN132_QUANT_CALL_PROBE
        _probe_dir = __import__('os').environ.get('EXTREME_QUANT_CALL_DIR')
        if _probe_dir and x.ndim >= 2 and x.shape[0] == 96:
            import json as _json
            import os as _os
            _prefix = getattr(layer, 'prefix', '<none>')
            try:
                _x_ptr = x.data_ptr()
                _q_ptr = quantized_x.data_ptr()
                _scale_ptr = pertoken_scale.data_ptr()
            except Exception:
                _x_ptr = _q_ptr = _scale_ptr = -1
            _weight = getattr(layer, 'weight', None)
            _weight_1 = getattr(layer, 'weight_1', None)
            _weight_2 = getattr(layer, 'weight_2', None)
            _rec = {
                'pid': _os.getpid(), 'prefix': _prefix,
                'x_shape': list(x.shape), 'x_dtype': str(x.dtype), 'x_ptr': _x_ptr,
                'q_shape': list(quantized_x.shape), 'q_ptr': _q_ptr,
                'scale_shape': list(pertoken_scale.shape), 'scale_ptr': _scale_ptr,
                'weight_shape': list(_weight.shape) if _weight is not None else None,
                'weight_1_shape': list(_weight_1.shape) if _weight_1 is not None else None,
                'weight_2_shape': list(_weight_2.shape) if _weight_2 is not None else None,
                'chunk_size': getattr(layer, '_chunk_size', 0),
            }
            _os.makedirs(_probe_dir, exist_ok=True)
            with open(_os.path.join(_probe_dir, f'pid{_os.getpid()}.jsonl'), 'a', encoding='utf-8') as _f:
                _f.write(_json.dumps(_rec) + '\\n')
'''
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=('install','restore'));p.add_argument('--record',type=Path,required=True);a=p.parse_args()
 if a.action=='install':
  assert not BACKUP.exists() and MARKER.encode() not in SOURCE.read_bytes()
  orig=SOURCE.read_bytes();src=orig.decode()
  assert src.count(ANCHOR)==1
  patched=src.replace(ANCHOR,ANCHOR+INSERT).encode()
  BACKUP.write_bytes(orig);SOURCE.write_bytes(patched)
  entry={'action':'install','source':str(SOURCE),'original_sha256':sha(orig),'patched_sha256':sha(patched)}
 else:
  assert BACKUP.exists() and MARKER.encode() in SOURCE.read_bytes()
  orig=BACKUP.read_bytes();SOURCE.write_bytes(orig);BACKUP.unlink()
  entry={'action':'restore','source':str(SOURCE),'restored_sha256':sha(SOURCE.read_bytes())}
 a.record.parent.mkdir(parents=True,exist_ok=True)
 a.record.write_text(json.dumps(entry,indent=2)+'\n');print(json.dumps(entry))
if __name__=='__main__':main()
