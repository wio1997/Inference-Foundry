#!/usr/bin/env python3
"""Decode and hash-check the public lossless window archive to a new output file."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import zlib

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('archive',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args()
    x=json.loads(a.archive.read_text());assert x['encoding']=='zlib+base64'
    c=base64.b64decode(x['data']);assert hashlib.sha256(c).hexdigest()==x['compressed_sha256']
    raw=zlib.decompress(c);assert hashlib.sha256(raw).hexdigest()==x['uncompressed_sha256']
    assert len(raw)==x['uncompressed_bytes']
    with a.output.open('xb') as f:f.write(raw)
    decoded=json.loads(raw)
    if isinstance(decoded,dict):
        count=len(decoded['windows']) if 'windows' in decoded else {k:len(v) for k,v in decoded.items() if isinstance(v,list)}
    else:
        count=len(decoded)
    print(json.dumps(dict(windows=count,output=str(a.output))))
