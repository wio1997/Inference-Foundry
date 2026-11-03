#!/usr/bin/env python3
"""Compare archived CANN 9381/9382 INI values and captured profiler DeviceInfo.

Read-only: uses task-local snapshots, does not access a device or network.
"""
from pathlib import Path
import configparser
import hashlib
import json

root = Path(__file__).resolve().parents[2]
sku_path = root / 'evidence/actual-sku-and-cp.json'
profile_path = root / 'evidence/profile-device-capacity-info.json'
skus = json.loads(sku_path.read_text())
profiles = json.loads(profile_path.read_text())
parsed = {}
for name in ('Ascend910_9381.ini', 'Ascend910_9382.ini'):
    text = skus[name]['text']
    config = configparser.ConfigParser(interpolation=None, strict=True)
    config.read_string(text)
    parsed[name] = config
    print(name, 'stored_text_sha256=' + hashlib.sha256(text.encode()).hexdigest(),
          'sections=' + str(len(config.sections())))
a, b = parsed.values()
diff = []
for section in sorted(set(a.sections()) | set(b.sections())):
    left = dict(a[section]) if a.has_section(section) else {}
    right = dict(b[section]) if b.has_section(section) else {}
    for key in sorted(set(left) | set(right)):
        if left.get(key) != right.get(key):
            diff.append((section, key, left.get(key), right.get(key)))
print('config_differences=', json.dumps(diff, ensure_ascii=False))
entries = profiles['device_info']
print('profile_entry_count=', len(entries), 'saved_errors=', len(profiles['errors']))
print('profile_device_ids=', sorted(v['id'] for e in entries for v in e['device_info']))
print('profile_core_vector_frequency_values=', sorted({(v['ai_core_num'], v['aiv_num'], v['aic_frequency']) for e in entries for v in e['device_info']}))
keys = {k for e in entries for k in e['all_deviceinfo_keys']}
print('DeviceInfo_keys_indicating_soc_chip_model_sku=', sorted(k for k in keys if any(t in k.lower() for t in ('soc', 'chip', 'model', 'sku'))))
for path in (sku_path, profile_path):
    print(path.name, 'sha256=' + hashlib.sha256(path.read_bytes()).hexdigest())
