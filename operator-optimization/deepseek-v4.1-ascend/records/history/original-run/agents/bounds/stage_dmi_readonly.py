#!/usr/bin/env python3
"""Read non-identifying DMI product labels inside the existing stage container.

No shell commands, network, hardware device API, serial number, UUID or writes.
Empty/unavailable fields are explicitly UNKNOWN. Output is a single JSON object.
"""
import json
from pathlib import Path

root=Path('/sys/class/dmi/id')
allowlist=('sys_vendor','product_name','product_version','board_vendor','board_name')
fields={}
for name in allowlist:
    try:
        value=(root/name).read_text(encoding='utf-8',errors='replace').strip()
        value=' '.join(value.split())[:160]
        fields[name]={'status':'KNOWN' if value else 'UNKNOWN', 'value':value or None}
    except FileNotFoundError:
        fields[name]={'status':'UNKNOWN','value':None,'reason':'NOT_PRESENT'}
    except PermissionError:
        fields[name]={'status':'UNKNOWN','value':None,'reason':'PERMISSION_DENIED'}
    except OSError as exc:
        fields[name]={'status':'UNKNOWN','value':None,'reason':'OS_ERROR','errno':exc.errno}
print(json.dumps({'scope':'container-visible DMI labels; no proof values identify physical host if virtualized',
                  'fields':fields,'serial_or_uuid_read':False,'npu_or_network_access':False},
                 ensure_ascii=False,sort_keys=True))
