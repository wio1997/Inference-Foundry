#!/usr/bin/env python3
"""Inspect installed public SDK signatures only; no store or metadata service start."""
import json
from pathlib import Path

import memcache_hybrid as mc
import memfabric_hybrid as mf


def main():
    result = {}
    for name in ('DistributedObjectStore', 'KeyInfo', 'LocalConfig', 'MetaConfig', 'MetaService'):
        cls = getattr(mc, name)
        result[name] = {'doc': cls.__doc__, 'methods': {}}
        for method in dir(cls):
            if not method.startswith('_'):
                obj = getattr(cls, method)
                result[name]['methods'][method] = getattr(obj, '__doc__', None)
    for module in (mc, mf):
        path = Path(module.__file__).parent/'VERSION'
        result[module.__name__+'_version'] = path.read_text()
    print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(result, indent=2)+'\nGLM53_RESULT_JSON_END')


if __name__ == '__main__':
    main()
