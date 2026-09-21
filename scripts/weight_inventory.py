import glob
import json
import struct
from collections import defaultdict
from pathlib import Path

root = Path('/data/yxy/DeepSeek-V4-Flash-0731-w4a8')
bits = {'F64': 8, 'F32': 4, 'F16': 2, 'BF16': 2, 'I64': 8, 'I32': 4, 'I16': 2, 'I8': 1, 'U8': 1, 'BOOL': 1}
groups = defaultdict(int)
tensors = 0
for file in sorted(glob.glob(str(root / '*.safetensors'))):
    with open(file, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))
    for name, meta in header.items():
        if name == '__metadata__':
            continue
        size = bits[meta['dtype']]
        for dim in meta['shape']:
            size *= dim
        if name.startswith('mtp.'):
            group = 'mtp_routed_experts' if '.ffn.experts.' in name else 'mtp_other'
        elif '.ffn.experts.' in name:
            group = 'target_routed_experts'
        elif '.ffn.shared_experts.' in name:
            group = 'shared_experts'
        elif '.ffn.' in name:
            group = 'other_ffn'
        elif '.self_attn.' in name or '.attn.' in name:
            group = 'attention'
        else:
            group = 'other'
        groups[group] += size
        tensors += 1
output = {'file_count': len(glob.glob(str(root / '*.safetensors'))), 'tensor_count': tensors,
          'groups_bytes': dict(groups), 'total_bytes': sum(groups.values()),
          'target_selected_weight_bytes_approx': sum(v for k, v in groups.items() if not k.startswith('mtp_')) - groups['target_routed_experts'] * (1 - 6/256),
          'mtp_selected_weight_bytes_approx': groups['mtp_other'] + groups['mtp_routed_experts'] * 6/256}
print(json.dumps(output, indent=2))
