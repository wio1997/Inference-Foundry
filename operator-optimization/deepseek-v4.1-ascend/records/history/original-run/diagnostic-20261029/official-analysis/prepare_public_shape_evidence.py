"""Select already measured signatures for public arithmetic checks; no device work."""
import hashlib
import json
from pathlib import Path

here = Path(__file__).resolve().parent
run = here.parents[1]
source = here / 'operator-shapes.json'
raw = json.loads(source.read_text())
selected = []
for g in raw['groups']:
    t, s, w = g['type'], g['input_shapes'], g['window']
    choose = w == 2 and (
        (t == 'QuantBatchMatmulV3' and '3,1280;1024,80' in s)
        or (t == 'TransposeBatchMatMul' and '24,1,4096' in s)
        or (t == 'MatMulV2' and '3,6144;25600' in s)
        or (t == 'HcPre' and s.startswith('"3,4,5120;'))
        or t in ('VllmAscendQuantLightningIndexerV2',
                 '_prepare_indexer_indices_kernel', '_engram_host_uva_gather_dequant_kernel')
        or (t == 'DispatchFFNCombine' and s.startswith('"3,5120;'))
    )
    choose |= w == 1 and t == 'DispatchFFNCombine' and s.startswith('"816,5120;24,')
    if choose:
        selected.append(g)
out = {
    'schema_version': 1,
    'run_id': raw['run_id'],
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'scope': 'Selected exact input signatures from the same 32 CSVs; pooled ranks/repeated calls, not independent experiments or wall time. No runtime tensor identity or output-shape proof.',
    'selection': 'Later small-row Q_b/output/Engram/mHC/MoE and QLI/helper/UVA signatures; initial M816 MoE. All 220 groups remain in private analysis.',
    'groups': selected,
}
path = run / 'publication-draft/official-operator-shapes.json'
path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'groups':len(selected), 'bytes':path.stat().st_size}))
