"""CPU-only diagnostic inputs: same 32 prefixes, one fresh fifth suffix each.

Executed only in the frozen tokenizer image through a separately reviewed launcher.
No model calls; no copying of text/token arrays to controller or Git.
"""
import hashlib, importlib.util, json
from pathlib import Path

SOURCE_SHA = '19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d'
CANONICAL_SHA = '9bdba915c372eb2c23da018a18c2c9653ded9c5012d875db8b496effb5983b67'

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--canonical', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--generator-reference', required=True)
    parser.add_argument('--generator-reference-sha256', required=True)
    parser.add_argument('--tokenizer', required=True)
    args = parser.parse_args()
    reference = Path(args.generator_reference)
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == args.generator_reference_sha256
    spec = importlib.util.spec_from_file_location('canonical_generator', reference)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    tokenizer_path = Path(args.tokenizer)
    assert hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() == SOURCE_SHA
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    canonical = Path(args.canonical)
    raw = (canonical/'manifest.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == CANONICAL_SHA
    manifest = json.loads(raw)
    assert manifest['concurrency'] == 32
    output = Path(args.output)
    output.mkdir(exist_ok=False)
    rows = []
    for index in range(32):
        prefix = json.loads((canonical/f'prefix-{index:02d}.json').read_text())
        text = (canonical/f'prefix-{index:02d}.txt').read_text()
        assert tokenizer.encode(text, add_special_tokens=False).ids == prefix['input_ids']
        assert len(prefix['input_ids']) == 74400 and original.id_sha(prefix['input_ids']) == manifest['prefixes'][index]['ids_sha256_le_u32']
        suffix, suffix_ids = original.exact_text(tokenizer, prefix['seed']+4*4999+1, 5600, 'Elder revision 5: evaluate only this new inventory appendix.\n')
        ids = tokenizer.encode(text+suffix, add_special_tokens=False).ids
        assert ids == prefix['input_ids']+suffix_ids and len(ids) == 80000
        lcps = []
        for variant in range(1,5):
            old = json.loads((canonical/f'request-p{index:02d}-v{variant}.json').read_text())
            old_meta = next(x for x in manifest['formal'] if x['prefix_id']==index and x['variant']==variant)
            assert original.id_sha(old['input_ids']) == old_meta['ids_sha256_le_u32']
            lcps.append(original.common_prefix(ids, old['input_ids']))
        assert lcps == [74400]*4, 'Fresh fifth suffix must diverge exactly at prefix boundary'
        record = dict(prefix_id=index, variant=5, input_tokens=80000, requested_output_tokens=600,
                      prefix_sha256_le_u32=original.id_sha(prefix['input_ids']),
                      ids_sha256_le_u32=original.id_sha(ids),
                      text_sha256_utf8=hashlib.sha256((text+suffix).encode()).hexdigest(),
                      lcp_against_previous_four=lcps, input_ids=ids)
        name = f'request-p{index:02d}-v5.json'
        (output/name).write_text(json.dumps(record)+'\n')
        (output/name.replace('.json','.txt')).write_text(text+suffix)
        rows.append({k:v for k,v in record.items() if k!='input_ids'} | {'file':name})
    result = dict(project_id='glm53', diagnostic_only=True, performance_certificate_allowed=False,
                  concurrency=32, requests=32, warmups=0, logical_prefix_tokens=74400,
                  canonical_manifest_sha256=CANONICAL_SHA, tokenizer_sha256=SOURCE_SHA,
                  generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  reference_generator_sha256=args.generator_reference_sha256,
                  fresh_suffix_lcp_exact=True, rows=rows)
    data = (json.dumps(result,indent=2)+'\n').encode()
    (output/'manifest.json').write_bytes(data)
    print(json.dumps({'project_id':'glm53','manifest_sha256':hashlib.sha256(data).hexdigest(),'manifest':result,'model_requests':0,'SDK_init_calls':0}))

if __name__=='__main__':
    main()
