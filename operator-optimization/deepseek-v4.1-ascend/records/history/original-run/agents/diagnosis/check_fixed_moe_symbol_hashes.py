#!/usr/bin/env python3
"""Match fixed source-derived public operator symbols against stage193 hashes; CPU only."""
import hashlib, json
from pathlib import Path
root = Path(__file__).resolve().parents[2]
stage = root / 'agents/diagnosis/draft-mc2-gap-api-names.json'
source_base = root / 'agents/fusion/frozen-source/vllm_ascend/ops/fused_moe'
checks = [
    ('_C_ascend::moe_gating_top_k_hash', source_base / 'router/fused_topk_router.py', 'torch.ops._C_ascend.moe_gating_top_k_hash('),
    ('_C_ascend::dispatch_ffn_combine', root / 'sources/vllm-workspace/vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py', 'torch.ops._C_ascend.dispatch_ffn_combine('),
]
def sha(data): return hashlib.sha256(data).hexdigest()
raw = stage.read_bytes()
assert sha(raw) == 'ef21672afb40e0ff7f0d26126c38cad2f1faa7ea93d3fafeb5f317c4f2404621'
rows = [x for x in json.loads(raw)['rows'] if x['name_withheld']]
assert len(rows) == 12 and {x['opId_local'] for x in rows} == {402, 405, 408}
result = []
for symbol, path, call in checks:
    content = path.read_bytes()
    lines = content.decode().splitlines()
    loc = [i for i, line in enumerate(lines, 1) if call in line]
    assert len(loc) == 1
    digest = sha(symbol.encode())
    matching = [r for r in rows if r['name_sha256'] == digest]
    assert len(matching) == 3 and {r['opId_local'] for r in matching} == {402,405,408}
    result.append({'symbol':symbol,'symbol_sha256':digest,'source_relative':str(path.relative_to(root)),'source_sha256':sha(content),'source_line':loc[0],'matched_rows':len(matching),'opId_local':sorted(r['opId_local'] for r in matching)})
matched = {x['symbol_sha256'] for x in result}
unmatched = [x for x in rows if x['name_sha256'] not in matched]
assert len(unmatched) == 6 and len({r['name_sha256'] for r in unmatched}) == 1
print(json.dumps({'stage193_sha256':sha(raw),'matches':result,'remaining_rows':len(unmatched),'remaining_distinct_hashes':1,'remaining_hash':unmatched[0]['name_sha256'],'limitations':['Hash match identifies trace text subject to collision resistance; source symbol occurrence does not establish loaded binary, arguments, request identity, runtime branch, CPU exclusivity or device causality.','Other name unresolved; no raw trace or path text extracted.']},indent=2))
