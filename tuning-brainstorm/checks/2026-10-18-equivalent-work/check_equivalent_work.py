# coding: utf-8
"""Narrow state/algorithm checks, not full SGLang tests or a performance benchmark.

Run beside the three sources pinned in source-version.json. No server or torch
import is needed. The production lock method and RadixKey class are extracted
with AST; surrounding tree/storage objects are explicitly lightweight stand-ins.
"""
from __future__ import annotations

import ast
import argparse
import copy
import hashlib
import json
import random
from array import array
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--fetch-source", action="store_true", help="Download missing pinned public source files.")
options = parser.parse_args()
version = json.loads((ROOT / "source-version.json").read_text())
for record in version["files"]:
    source_path = ROOT / record["file"]
    if not source_path.exists():
        if not options.fetch_source:
            raise SystemExit("Missing pinned source; rerun with --fetch-source: " + str(source_path))
        with urlopen(record["url"], timeout=30) as response:
            source_path.write_bytes(response.read())
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == record["sha256"], source_path


def load_definition(filename, class_name, method_name=None):
    module = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    if method_name is None:
        return cls
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)


def evaluate(definitions, namespace):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future] + definitions, type_ignores=[]))
    exec(compile(tree, "<pinned-source-definition>", "exec"), namespace)


original = load_definition("full.py", "FullComponent", "acquire_component_lock")
original.name = "acquire_reference"
candidate = copy.deepcopy(original)
candidate.name = "acquire_candidate"
moves = 0
for block in ast.walk(candidate):
    if not isinstance(block, ast.While):
        continue
    zero_test = next((n for n in block.body if isinstance(n, ast.If) and ast.unparse(n.test) == "cd.lock_ref == 0"), None)
    discard = next((n for n in block.body if ast.unparse(n) == "self.tree_core.evictable_device_leaves.discard(cur)"), None)
    if zero_test is not None and discard is not None:
        block.body.remove(discard)
        zero_test.body.append(discard)
        moves += 1
assert moves == 1, "Source changed; review the proposed transformation."
release = load_definition("full.py", "FullComponent", "release_component_lock")
release.name = "release_reference"
scope = {}
evaluate([original, candidate, release], scope)


class CountedSet(set):
    calls = 0

    def discard(self, item):
        self.calls += 1
        super().discard(item)


class Node:
    def __init__(self, node_id, parent):
        self.id, self.parent = node_id, parent
        self.children = {}
        self.component_data = [SimpleNamespace(value=[0] * 4, lock_ref=0)]
        self.aux_ref = 0
        if parent is not None:
            parent.children[node_id] = self


class LockTree:
    def __init__(self, evicted=(), auxiliary=False):
        # root -> P1 -> P2 -> P3 -> P4 -> A/B/C
        self.nodes = []
        for i, parent in enumerate([None, 0, 1, 2, 3, 4, 4, 4]):
            self.nodes.append(Node(i, None if parent is None else self.nodes[parent]))
        self.root_node = self.nodes[0]
        for i in evicted:
            self.nodes[i].component_data[0].value = None
        if auxiliary:
            self.nodes[2].aux_ref = self.nodes[7].aux_ref = 1
        self.component_evictable_size_ = {0: sum(len(n.component_data[0].value or []) for n in self.nodes[1:])}
        self.component_protected_size_ = {0: 0}
        self.evictable_device_leaves = CountedSet(n for n in self.nodes if self.eligible(n))

    def eligible(self, node):
        return (node is not self.root_node and node.component_data[0].value is not None
                and node.component_data[0].lock_ref == 0 and node.aux_ref == 0
                and all(c.component_data[0].value is None for c in node.children.values()))

    def _update_evictable_leaf_sets(self, node):
        if self.eligible(node):
            self.evictable_device_leaves.add(node)
        else:
            self.evictable_device_leaves.discard(node)

    def snapshot(self):
        assert {n.id for n in self.evictable_device_leaves} == {n.id for n in self.nodes if self.eligible(n)}
        return (tuple(n.component_data[0].lock_ref for n in self.nodes),
                tuple(sorted(n.id for n in self.evictable_device_leaves)),
                self.component_evictable_size_[0], self.component_protected_size_[0])


def component(tree):
    return SimpleNamespace(tree_core=tree, component_type=0,
                           reclaimable_tokens=lambda n: len(n.component_data[0].value))


lock_events = 0
toy_discards = None
for fixture, (evicted, auxiliary) in enumerate([((), False), ((5,), False), ((5, 6), False), ((), True)]):
    before, after = LockTree(evicted, auxiliary), LockTree(evicted, auxiliary)
    ca, cb = component(before), component(after)
    active = []
    rng = random.Random(20261018 + fixture)
    sequence = [(True, leaf) for leaf in [5, 6, 7]]
    for _ in range(1000):
        if active and (len(active) >= 24 or rng.random() < .47):
            sequence.append((False, active.pop(rng.randrange(len(active)))))
        else:
            leaf = rng.choice([5, 6, 7])
            active.append(leaf)
            sequence.append((True, leaf))
    # The three initial holders were not in the random generator's list.
    sequence += [(False, leaf) for leaf in active + [5, 6, 7]]
    for step, (take, leaf) in enumerate(sequence):
        if take:
            ra, rb = SimpleNamespace(delta=0), SimpleNamespace(delta=0)
            scope["acquire_reference"](ca, before.nodes[leaf], ra)
            scope["acquire_candidate"](cb, after.nodes[leaf], rb)
            assert ra.delta == rb.delta
        else:
            scope["release_reference"](ca, before.nodes[leaf], None)
            scope["release_reference"](cb, after.nodes[leaf], None)
        assert before.snapshot() == after.snapshot(), (fixture, step)
        lock_events += 1
        if fixture == 0 and step == 2:
            toy_discards = {"reference": before.evictable_device_leaves.calls,
                            "candidate": after.evictable_device_leaves.calls}
    assert all(n.component_data[0].lock_ref == 0 for n in before.nodes[1:])


key_scope = {"array": array}
evaluate([load_definition("radix_cache.py", "RadixKey")], key_scope)
RadixKey = key_scope["RadixKey"]


def build_chain(tokens, page, bigram, namespace, salt, rng):
    root = SimpleNamespace(id=0, children={})
    node, start, logical_len = root, 0, len(tokens) - int(bigram)
    while start < logical_len:
        stop = min(logical_len, start + page * rng.randint(1, 3))
        key = RadixKey(tokens[start:stop + int(bigram)], namespace, is_bigram=bigram, cache_salt=salt)
        child = SimpleNamespace(id=node.id + 1, children={}, key=key)
        node.children[key.child_key(page)] = child
        node, start = child, stop
    return root


def suffix_walk(root, key, page):
    node, trace, copied = root, [], 0
    while len(key):
        child = node.children.get(key.child_key(page))
        if child is None:
            break
        matched = child.key.match(key, page_size=page)
        trace.append((child.id, matched))
        node = child
        key = key[matched:]
        copied += len(key.token_ids)
        if matched < len(child.key):
            break
    return trace, tuple(key.raw_token_ids()), copied


def offset_walk(root, key, page):
    node, trace, offset = root, [], 0
    while offset < len(key):
        child = node.children.get(key.child_key_at(offset, page))
        if child is None:
            break
        matched = child.key.match_at(key, offset, page_size=page)
        trace.append((child.id, matched))
        node, offset = child, offset + matched
        if matched < len(child.key):
            break
    # A persistent tail still materializes once; ownership is not made a view.
    tail = key if offset == 0 else key[offset:]
    return trace, tuple(tail.raw_token_ids()), 0 if offset == 0 else len(tail.token_ids)


cursor_cases = 0
rng = random.Random(7)
for page in [1, 2, 4, 128]:
    for bigram in [False, True]:
        for namespace, salt in [(None, None), ("adapter", None), ("adapter", "tenant")]:
            for _ in range(100):
                logical_len = page * rng.randint(1, 12)
                tokens = array("q", range(logical_len + int(bigram)))
                root = build_chain(tokens, page, bigram, namespace, salt, rng)
                query = tokens[:]
                if rng.random() < .5:
                    query[rng.randrange(len(query))] = -999
                limit = rng.randrange(len(query) + 1) if rng.random() < .5 else None
                key = RadixKey(query, namespace, is_bigram=bigram, limit=limit, cache_salt=salt).page_aligned(page)
                reference, candidate_result = suffix_walk(root, key, page), offset_walk(root, key, page)
                assert reference[:2] == candidate_result[:2], (page, bigram, namespace, salt, limit)
                assert candidate_result[2] <= reference[2]
                cursor_cases += 1

# Fixed illustrative chain: 65,536 tokens through 64 matched 1,024-token nodes.
tokens = array("q", range(65536))
root, node = SimpleNamespace(id=0, children={}), None
node = root
for i in range(64):
    key = RadixKey(tokens[i * 1024:(i + 1) * 1024])
    child = SimpleNamespace(id=i + 1, children={}, key=key)
    node.children[key.child_key(128)] = child
    node = child
query = RadixKey(tokens)
old, new = suffix_walk(root, query, 128), offset_walk(root, query, 128)
assert old[:2] == new[:2]
result = {
    "source_commit": json.loads((ROOT / "source-version.json").read_text())["commit"],
    "status": "PASS",
    "lock_state_events": lock_events,
    "lock_toy_acquire_loop_discard_calls": toy_discards,
    "cursor_cases": cursor_cases,
    "cursor_toy": {"tokens": 65536, "nodes": 64, "element_bytes": tokens.itemsize,
                   "reference_suffix_payload_bytes": old[2] * tokens.itemsize,
                   "candidate_suffix_payload_bytes": new[2] * tokens.itemsize},
    "limits": ["Lightweight surrounding objects; not complete SGLang tests.",
               "Cursor checks match traces and final suffix only; insertion side effects, barriers, ownership and split/eviction interleaving remain unverified.",
               "Operation/payload counts only; no latency, throughput or real workload speedup measurement.",
               "Suffix counts exclude page alignment, key comparison slices, persistent-node materialization and tensor-index operations."]
}
(ROOT / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
