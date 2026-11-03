#!/usr/bin/env python3
"""Hash and check bounded official web snapshots; no network/device access."""
import hashlib
import json
from pathlib import Path

p=Path(__file__).resolve().parent
records={
 'atlas_800i_a3':{
  'file':'atlas-800i-a3-web.txt',
  'url':'https://e.huawei.com/cn/products/computing/ascend/atlas-800i-a3',
  'required':['8.96 POPS INT8','3.2TB/s','双向784GB/s'],
  'scope':'Atlas 800I A3 whole server, eight 910 processors; product-page headline, not verified as this host',
 },
 'atlas_800t_a3':{
  'file':'atlas-800t-a3-web.txt',
  'url':'https://e.huawei.com/cn/products/computing/ascend/atlas-800t-a3',
  'required':['12.0 POPS INT8','3.2TB/s','双向784GB/s'],
  'scope':'Atlas 800T A3 whole server, eight 910 processors; product-page headline, not verified as this host',
 },
 'soc_identity_cann91':{
  'file':'soc-name-a3-web.txt',
  'url':'https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/Utils-API/platform_info/PlatformAscendC/GetSocVersion.md',
  'required':['Chip Name_NPU Name','Atlas A3'],
  'scope':'CANN9.1.X A3 model naming rule; applies to the locally queried board0 conditional on A3 category',
 },
}
for rec in records.values():
 q=p/rec['file']; data=q.read_bytes(); text=data.decode('utf-8')
 assert all(v in text for v in rec['required']),q
 rec['sha256']=hashlib.sha256(data).hexdigest()
 rec['retrieved_utc']='2026-10-28'
 rec['publisher']='Huawei / Ascend'
 rec['actual_169_hard_capacity_certified']=False
out={'sources':records,'vendor_800i_a3_document_07':{
 'url':'https://e.huawei.com/cn/documents/products/computing/5f99170196404c5f90a7d0638c613652',
 'state':'Landing page indicates registration form; content not accessed; page link label says 04 but destination title says 07.'
},'actual_169_physical_hardware_upper':'UNKNOWN','no_new_device_work':True}
(p/'vendor_sources_9382.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(out,indent=2,ensure_ascii=False))
