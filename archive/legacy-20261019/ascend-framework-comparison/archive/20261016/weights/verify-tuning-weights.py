# Weight completeness verification vs authoritative evidence/source-manifest.json (coordinator).
# Validates: all 193 file paths+sizes, 183 safetensors (incl optional/quarot) headers/offsets,
# index refs, and reports root-182 subtotal. Full-payload sha256 done separately (sha-verify).
import json, struct
from pathlib import Path

T = Path('/data/wio/sglang-glm53-33-41-20261013')
dest = Path('/data/wio/glm53-w8a8-tuning-local')
man = json.loads((T / 'evidence' / 'source-manifest.json').read_text())
src = {f['path']: f['bytes'] for f in man['files']}
assert len(src) == 193, f'manifest files {len(src)}'
st_src = {k: v for k, v in src.items() if k.endswith('.safetensors')}
assert len(st_src) == 183 and sum(st_src.values()) == 764188259784, 'manifest safetensors totals'
root_src = {k: v for k, v in st_src.items() if '/' not in k}
assert len(root_src) == 182 and sum(root_src.values()) == 764037264744, 'manifest root-182 totals'

dst = {}
for p in dest.rglob('*'):
    if p.is_file():
        dst[str(p.relative_to(dest))] = p.stat().st_size
missing = sorted(set(src) - set(dst)); extra = sorted(set(dst) - set(src))
assert not missing, f'missing: {missing[:10]}'
assert not extra, f'extra: {extra[:10]}'
mism = [k for k in src if dst[k] != src[k]]
assert not mism, f'size mismatch: {mism[:10]}'

st_total = 0
for k, size in sorted(dst.items()):
    if not k.endswith('.safetensors'):
        continue
    with (dest / k).open('rb') as f:
        ln = struct.unpack('<Q', f.read(8))[0]
        assert 0 < ln < 100_000_000, f'{k}: bad header length {ln}'
        header = json.loads(f.read(ln))
    spans = sorted(v['data_offsets'] for kk, v in header.items() if kk != '__metadata__')
    assert spans and spans[0][0] == 0, f'{k}: first span must start at 0'
    for a, b in zip(spans, spans[1:]):
        assert a[1] == b[0], f'{k}: non-contiguous spans'
    assert 8 + ln + spans[-1][1] == size, f'{k}: size mismatch 8+{ln}+{spans[-1][1]} != {size}'
    st_total += 1

index = json.loads((dest / 'quant_model_weights.safetensors.index.json').read_text())
refs = set(index['weight_map'].values())
assert refs.issubset({k for k in dst if '/' not in k and k.endswith('.safetensors')}), 'index refs outside shards'

out = dict(
    path=str(dest), manifest='evidence/source-manifest.json',
    file_count=len(dst), total_bytes=sum(dst.values()),
    safetensors=len(st_src), safetensors_bytes=sum(st_src.values()),
    root_safetensors=len(root_src), root_safetensors_bytes=sum(root_src.values()),
    safetensors_headers_validated=st_total,
    index_weight_map_refs=len(refs), index_refs_all_present=True,
    names_sizes='all 193 relative paths + sizes match source manifest',
    validation='names/sizes + all safetensors headers/offsets + index refs (this script); full-payload sha256 via sha-verify-weights.sh (separate step)')
(T / 'weights-local.verified.json').write_text(json.dumps(out, indent=2) + '\n')
(T / 'weights-local.complete').write_text(str(sum(st_src.values())) + '\n')
print(json.dumps(out))
