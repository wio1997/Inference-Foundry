"""Generate synthetic, independent text/ID inputs inside the frozen runtime.

Code preparation only until a separately reviewed ZCode launcher executes it.
All writes belong under a new project run/artifacts directory; never Git payloads.
"""
import hashlib
import itertools
import json
from pathlib import Path
import random
import struct

TOKENIZER_SHA = '19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d'
WORDS = ('amber basin cedar delta ember forest garden harbor island juniper kettle lantern '
         'meadow north orchard pebble quartz river silver timber upland valley willow xenon '
         'yellow zephyr atlas bridge circle drift engine field grain horizon ivory journey '
         'kernel lattice matrix network orbit prism ridge signal tunnel unit vector wave').split()


def id_sha(ids):
    return hashlib.sha256(struct.pack('<'+'I'*len(ids), *ids)).hexdigest()


def common_prefix(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def exact_text(tokenizer, random_seed, target, first, delimiter=''):
    rng = random.Random(random_seed)
    lines = [first]
    ids = []
    while len(ids) < target + 512:
        for _ in range(200):
            name = ' '.join(rng.sample(WORDS, 6))
            lines.append(f'{rng.getrandbits(48):012x}: {name}; units {rng.randrange(100,99999)}; '
                         f'level {rng.randrange(10,999)}; note {rng.choice(WORDS)} {rng.choice(WORDS)}.\n')
        source_text = ''.join(lines)
        encoded = tokenizer.encode(source_text, add_special_tokens=False)
        ids = encoded.ids
    # Use original character offsets; tokenizer JSON decoders need not be invertible.
    # The resulting literal text must still re-encode exactly under the same encoder.
    candidates = [0] + [delta for distance in range(1, 65) for delta in (-distance, distance)]
    seen = []
    for delta in candidates:
        boundary = encoded.offsets[target + delta - 1][1]
        text = source_text[:boundary] + delimiter
        actual = tokenizer.encode(text, add_special_tokens=False).ids
        seen.append((delta, len(actual)))
        if len(actual) == target:
            assert tokenizer.encode(text, add_special_tokens=False).ids == actual
            return text, actual
    raise RuntimeError(f'Cannot construct exact text boundary; target={target}, observed={seen[:16]}; reject, never normalize')


def generate(output, tokenizer_path, concurrency, seed=20261017):
    assert concurrency in (16,32,64)
    tokenizer_path = Path(tokenizer_path)
    assert hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() == TOKENIZER_SHA
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    output = Path(output)
    assert '/data/wio/inference-foundry/glm53/runs/' in str(output.resolve())
    output.mkdir(parents=True, exist_ok=False)
    prefixes=[];requests=[]
    for prefix_index in range(concurrency):
        prefix_seed=seed+prefix_index*100003
        prefix_text,prefix_ids=exact_text(tokenizer,prefix_seed,74400,
            f'{random.Random(prefix_seed).getrandbits(128):032x} independent inventory record\n','\n\n')
        prefix={'prefix_id':prefix_index,'seed':prefix_seed,'token_count':len(prefix_ids),
                'ids_sha256_le_u32':id_sha(prefix_ids),
                'text_sha256_utf8':hashlib.sha256(prefix_text.encode()).hexdigest(),
                'input_ids':prefix_ids}
        prefixes.append(prefix)
        (output/f'prefix-{prefix_index:02}.json').write_text(json.dumps(prefix)+'\n')
        (output/f'prefix-{prefix_index:02}.txt').write_text(prefix_text)
        variants=[]
        for variant,label in enumerate(('Aster','Birch','Cedar','Dahlia')):
            suffix_text,suffix_ids=exact_text(tokenizer,prefix_seed+variant*4999+1,5600,
                f'{label} revision {variant+1}: evaluate only this new inventory appendix.\n')
            full_text=prefix_text+suffix_text
            ids=tokenizer.encode(full_text,add_special_tokens=False).ids
            assert ids==prefix_ids+suffix_ids, 'BPE boundary changed; reject dataset'
            assert len(ids)==80000
            record={'request_id':f'glm53-C{concurrency}-p{prefix_index:02}-v{variant+1}',
                    'prefix_id':prefix_index,'variant':variant+1,
                    'input_tokens':80000,'requested_output_tokens':600,
                    'ids_sha256_le_u32':id_sha(ids),'text_sha256_utf8':hashlib.sha256(full_text.encode()).hexdigest(),
                    'input_ids':ids}
            variants.append(ids)
            name=f'request-p{prefix_index:02}-v{variant+1}.json'
            (output/name).write_text(json.dumps(record)+'\n')
            (output/name.replace('.json','.txt')).write_text(full_text)
            requests.append({k:v for k,v in record.items() if k!='input_ids'}|{'file':name})
        assert len({ids[74400] for ids in variants})==4
        assert all(common_prefix(a,b)==74400 for a,b in itertools.combinations(variants,2))
    cross_lcp=[common_prefix(a['input_ids'],b['input_ids']) for a,b in itertools.combinations(prefixes,2)]
    assert max(cross_lcp)<=16, 'Independent-prefix LCP gate failed'
    summary={'project_id':'glm53','schema':1,'concurrency':concurrency,'seed':seed,
             'tokenizer_sha256':TOKENIZER_SHA,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'warmup_requests':concurrency,'formal_requests':4*concurrency,'input_tokens':80000,
             'output_tokens':600,'logical_prefix_tokens':74400,'same_prefix_variant_lcp':74400,
             'independent_prefix_lcp':{'min':min(cross_lcp),'max':max(cross_lcp),
                                       'all_pairs':cross_lcp},
             'id_hash_encoding':'SHA256 of uint32 little-endian token IDs',
             'prefixes':[{k:v for k,v in p.items() if k!='input_ids'} for p in prefixes],
             'formal':requests,'qualification':'INPUTS_ONLY_NOT_SERVICE_TOKENS_OR_PERFORMANCE'}
    (output/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--tokenizer',required=True);parser.add_argument('--concurrency',type=int,required=True)
    args=parser.parse_args();result=generate(args.output,args.tokenizer,args.concurrency)
    print(json.dumps({k:v for k,v in result.items() if k not in ('prefixes','formal')}))
