#!/usr/bin/env python3
"""Apply a version-checked DSA metadata initialization patch; no NPU calls."""
import hashlib
import pathlib

TARGET = pathlib.Path('/opt/vllm-ascend-dsnext/vllm_ascend/attention/dsa_v41.py')
BASE_SHA = '47cfb57d13143ed56f7eb0c76511f80d28af3deeb6477c1380c9845345782ef2'
ANCHOR = b'''class DeepseekV41MetadataBuilder(AttentionMetadataBuilder[DeepseekV41Metadata]):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
'''
ADDITION = b'        self._init_reorder_batch_threshold(supports_spec_as_decode=True)\n'

def patched(base):
    assert hashlib.sha256(base).hexdigest() == BASE_SHA, 'Unexpected source SHA256'
    assert base.count(ANCHOR) == 1, 'Unexpected metadata builder layout'
    out = base.replace(ANCHOR, ANCHOR + ADDITION, 1)
    compile(out, str(TARGET), 'exec')
    return out

def main():
    backup = pathlib.Path('/workspace/deploy/backups/dsa_v41.py.original')
    current = TARGET.read_bytes()
    if backup.exists():
        desired = patched(backup.read_bytes())
        if current == desired:
            print('DSA patch already applied'); return
        assert current == backup.read_bytes(), 'Source differs from backup and patch'
    else:
        desired = patched(current)
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes(current)
    TARGET.write_bytes(desired)
    print('DSA patched SHA256=' + hashlib.sha256(desired).hexdigest())

if __name__ == '__main__':
    main()
