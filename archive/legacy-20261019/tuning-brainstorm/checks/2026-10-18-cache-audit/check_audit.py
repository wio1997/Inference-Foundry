# coding: utf-8
"""Finite source-derived checks for CACHE-010/011/020/023, not engine tests.

Use --fetch-source beside source-version.json to retrieve missing fixed files.
No torch, model, accelerator or server is used. Dependencies outside the extracted
methods are simple stand-ins; passing does not validate complete engine behavior.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
import random
from array import array
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
options = argparse.ArgumentParser(description=__doc__)
options.add_argument('--fetch-source', action='store_true')
args = options.parse_args()
version = json.loads((ROOT / 'source-version.json').read_text())
needed = ['radix_cache.py', 'unified_radix_cache.py', 'common.py']
for record in version['files']:
    if record['file'] not in needed:
        continue
    target = ROOT / record['file']
    if not target.exists():
        if not args.fetch_source:
            raise SystemExit('Missing fixed source; use --fetch-source: ' + str(target))
        with urlopen(record['url'], timeout=30) as response:
            target.write_bytes(response.read())
    assert hashlib.sha256(target.read_bytes()).hexdigest() == record['sha256']


def definition(file, name, cls=None):
    tree = ast.parse((ROOT / file).read_text(encoding='utf8'))
    body = tree.body if cls is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    return copy.deepcopy(next(n for n in body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name))


def compile_defs(defs, namespace=None):
    namespace = {} if namespace is None else namespace
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future] + defs, type_ignores=[]))
    exec(compile(tree, '<fixed-source-check>', 'exec'), namespace)
    return namespace


# 010: move only token_ids slicing after the existing disabled-cache return.
old = definition('unified_radix_cache.py', 'checkpoint_into_tree', 'UnifiedRadixCache')
old.name = 'before_checkpoint'
new = copy.deepcopy(old); new.name = 'after_checkpoint'
assignment = next(n for n in new.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'token_ids')
new.body.remove(assignment)
disabled = next(n for n in new.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'self.disable')
new.body.insert(new.body.index(disabled) + 1, assignment)
ns = compile_defs([old, new])

class SliceCounter:
    def __init__(self): self.calls = 0
    def __getitem__(self, key): self.calls += 1; return []

disabled_cases = 0
for finished in [False, True]:
    for component_count in [0, 1, 3]:
        outputs = []
        for name in ['before_checkpoint', 'after_checkpoint']:
            log = []
            components = [SimpleNamespace(cleanup_after_caching_req=lambda *a, **kw: log.append(kw)) for _ in range(component_count)]
            req = SimpleNamespace(finished=lambda: finished, full_untruncated_fill_ids=SliceCounter())
            result = ns[name](SimpleNamespace(disable=True, _components_tuple=components), req, up_to=128)
            outputs.append((result, log, req.full_untruncated_fill_ids.calls))
        assert outputs[0][:2] == outputs[1][:2]
        assert outputs[0][2] == 1 and outputs[1][2] == 0
        disabled_cases += 1

# 011: actual RadixKey slicing + first-page key vs direct child_key_at.
ns = compile_defs([definition('radix_cache.py', 'RadixKey')], {'array': array})
RadixKey = ns['RadixKey']; rng = random.Random(1011)
split_cases = 0
for page in [1, 2, 4, 128]:
    for bigram in [False, True]:
        for namespace, salt in [(None, None), ('lora', None), ('lora', 'tenant')]:
            for _ in range(100):
                logical = page * rng.randint(2, 12)
                raw = array('q', range(logical + int(bigram) + page * 2))
                key = RadixKey(raw, namespace, is_bigram=bigram, limit=logical + int(bigram), cache_salt=salt)
                offset = page * rng.randint(1, logical // page - 1)
                assert key[offset:].child_key(page) == key.child_key_at(offset, page)
                split_cases += 1

# 020: compile the production nested target_reached closure with its setup.
original = definition('unified_radix_cache.py', '_evict_components', 'UnifiedRadixCache')
target = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == 'target_reached')
index = original.body.index(target)
original.body = original.body[:index + 1] + [ast.Return(value=ast.Name(id='target_reached', ctx=ast.Load()))]
original.name = 'target_before'
candidate = copy.deepcopy(original); candidate.name = 'target_after'
target_new = next(n for n in candidate.body if isinstance(n, ast.FunctionDef))
donor_if = next(n for n in target_new.body if isinstance(n, ast.If) and 'component_type == ComponentType.FULL' in ast.unparse(n.test))
assert not donor_if.orelse
donor_if.orelse = [ast.Return(value=ast.Constant(value=False))]
ns = compile_defs([original, candidate], {'ComponentType': SimpleNamespace(FULL=0, MAMBA=2)})

class Capacity:
    def __init__(self, initial):
        self.value = initial; self.reads = 0; self.prepares = 0
        self.token_to_kv_pool_allocator = SimpleNamespace(mamba_full_cache_donor=lambda: self)
    def _component_available_size(self, ct): self.reads += 1; return self.value
    def full_tokens_before_mamba_recheck(self, target): return 4
    def prepare_mamba_allocation(self, target): self.prepares += 1; self.value = target

capacity_cases = saved_reads = 0
for ct in [0, 1, 2]:
    for target_ct in [0, 1, 2]:
        for target_size in [1, 4, 8]:
            for initial in range(10):
                for reclaimed in [0, 4]:
                    caches = [Capacity(initial), Capacity(initial)]
                    outputs = []
                    for method, cache in zip(['target_before', 'target_after'], caches):
                        fn = ns[method](cache, {}, {0: reclaimed, 1: reclaimed, 2: reclaimed}, {ct: (target_ct, target_size)})
                        outputs.append(fn(ct))
                    assert outputs[0] == outputs[1]
                    assert caches[0].prepares == caches[1].prepares and caches[0].value == caches[1].value
                    assert caches[1].reads <= caches[0].reads
                    saved_reads += caches[0].reads - caches[1].reads
                    capacity_cases += 1

# 023: actual range classification; direct hand-off when dead interval is empty.
old = definition('common.py', 'free_kv_row_segments'); old.name = 'ranges_before'
new = copy.deepcopy(old); new.name = 'ranges_after'
interval = next(n for n in new.body if isinstance(n, ast.Assign) and 'dead_lo, dead_hi' in ast.unparse(n.targets[0]))
guard = ast.parse('if dead_hi == dead_lo:\n    allocator.free_segments(segments)\n    return').body[0]
new.body.insert(new.body.index(interval) + 1, guard)
ns = compile_defs([old, new])

class Segment:
    def __init__(self, data): self.data = list(data)
    def numel(self): return len(self.data)
    def __getitem__(self, item): return Segment(self.data[item])

class Allocator:
    page_size = 4
    def __init__(self): self.freed = []
    def free_segments(self, segments): self.freed += [('both', pos, tuple(t.data)) for t, pos in segments if t.numel()]
    def free_full_segments(self, segments): self.freed += [('full', pos, tuple(t.data)) for t, pos in segments if t.numel()]

range_cases = 0
for _ in range(600):
    segments = [(Segment(range(start, start + rng.choice([0, 4, 8]))), start) for start in [0, 8, 16]]
    lo, end = rng.choice([0, 4, 8, 12, 16]), rng.choice([0, 4, 8, 12, 16, 24])
    outputs = []
    for name in ['ranges_before', 'ranges_after']:
        allocator = Allocator()
        ns[name](allocator, segments, swa_evicted_seqlen=end, swa_dead_lo=lo)
        outputs.append(allocator.freed)
    assert outputs[0] == outputs[1]
    range_cases += 1

result = {'commit': version['commit'], 'status': 'PASS',
          'CACHE-010': {'disabled_branch_cases': disabled_cases},
          'CACHE-011': {'split_key_cases': split_cases},
          'CACHE-020': {'capacity_cases': capacity_cases, 'total_removed_metadata_reads_in_cases': saved_reads},
          'CACHE-023': {'range_cases': range_cases},
          'limits': ['Extracted source + lightweight dependencies, not full SGLang correctness tests.',
                     'Only the listed local behaviors were compared; no model, asynchronous device work or timing benchmark.',
                     'Capacity getter is assumed pure between the two reads; donor preparation remains observable and tested.']}
(ROOT / 'validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
print(json.dumps(result, ensure_ascii=False, indent=2))
