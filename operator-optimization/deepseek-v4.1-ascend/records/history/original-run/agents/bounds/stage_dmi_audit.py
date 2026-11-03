#!/usr/bin/env python3
"""Review saved stage DMI and ZCode receipt without contacting host or NPU."""
import hashlib
import json
from pathlib import Path

p=Path(__file__).resolve().parent
z=Path('/Users/wio/work/173-sglang-matched-172-benchmark')
files={
 'result':p/'stage-dmi-readonly.json',
 'query_script':p/'stage_dmi_readonly.py',
 'transport_wrapper':p/'run_stage_dmi.py',
 'exit_code':z/'operator-ceiling169-v2-086-dmi.exit-code',
 'verified':z/'operator-ceiling169-v2-086-dmi.verified.json',
 'official_catalog':p/'isoftstone-catalog-search.txt',
}
d=json.loads(files['result'].read_text());v=json.loads(files['verified'].read_text())
assert files['exit_code'].read_text().strip()=='0'
assert v['model_verified'] and 'deepseek-flash' in v['observed_models']
assert hashlib.sha256(files['query_script'].read_bytes()).hexdigest()==d['script_sha256']
assert d['no_device_work'] and not d['npu_or_network_access'] and not d['serial_or_uuid_read']
assert '软通华方超强A800T A3' in files['official_catalog'].read_text()
out={'query_status':'SUCCESS','verified_zcode_model':'deepseek-flash','container_visible_dmi_fields':d['fields'],
 'host_is_bare_metal_proven':False,
 'model_name_comparison':'DMI ChaoqiangA800T A3 and official directory 超强A800T A3 appear consistent by transliteration; not a verified one-to-one SKU mapping.',
 'independent_complete_oem_specs_found':False,'huawei_800t_spec_applies_proven':False,
 'actual_physical_hard_limits':'UNKNOWN','no_npu_operations_in_this_audit':True,
 'input_sha256':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in files.items()}}
(p/'stage-dmi-audit.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(out,indent=2,ensure_ascii=False))
