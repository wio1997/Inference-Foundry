"""Apply the formal cache scope fix to the pinned, site-local G1332 preflight."""

import argparse
from hashlib import sha256
from pathlib import Path


SOURCE_SHA256 = '83d7c1fde7bd61aa19d4e69936a0f33f65f468bc56b2e762b60240a0270cf7d1'


def render(source):
    assert sha256(source).hexdigest() == SOURCE_SHA256
    rendered = source.decode()
    replacements = (
        ('import urllib.request\n',
         'import urllib.request\n\nfrom verify_SG_matched_profile_cache_g1428 import validate_formal_cache_scope\n'),
        (" and profile['cold_cache_counter_verified']\n",
         "\n    assert profile['dataset_sha256']==run_manifest['formal_dataset_sha256']['32768']\n"),
        ("    assert fg['profile_epoch']==1317\n",
         "    assert fg['profile_epoch']==1317\n    formal_cache_scope=validate_formal_cache_scope(profile,fg)\n"),
        ("        'SGLang_formal_audit_SHA256':",
         "        'SGLang_formal_cache_scope':formal_cache_scope,\n        'SGLang_formal_audit_SHA256':"),
        ('G1332_SG_PROFILE_GATE_JSON ', 'G1428_SG_PROFILE_GATE_JSON '),
    )
    for previous, replacement in replacements:
        assert rendered.count(previous) == 1, previous
        rendered = rendered.replace(previous, replacement)
    return rendered.encode()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    source = render(args.source.read_bytes())
    compile(source, str(args.output), 'exec')
    with args.output.open('xb') as stream:
        stream.write(source)
