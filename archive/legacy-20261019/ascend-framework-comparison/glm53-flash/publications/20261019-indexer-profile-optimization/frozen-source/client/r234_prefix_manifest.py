"""Load exactly prefix000/001 from the accepted R139/private-C32 dataset.

The loader verifies the accepted manifest SHA and every prefix id SHA; it
never retokenises and never fabricates a prefix.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

MANIFEST_SHA256 = "c70d5bf944e92e0d4b5673a45f53022ce723278cb9c627a48ef82af78833798b"
PREFIXES = ("C32-prefix000", "C32-prefix001")
WARM_TOKENS = 74400
VOCAB = 154856


def digest_ids(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def load_prefixes(root):
    root = pathlib.Path(root)
    raw = (root / "manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError("private-C32 manifest SHA mismatch")
    manifest = json.loads(raw)
    if manifest.get("concurrency") != 32 or manifest.get("actual_full_input_tokens") != 80000:
        raise ValueError("not the accepted private-C32 dataset")
    by_name = {row["logical_prefix_id"]: row for row in manifest.get("rows", [])}
    if set(by_name) != {"C32-prefix%03d" % i for i in range(32)}:
        raise ValueError("unexpected logical prefix set")
    result = []
    for name in PREFIXES:
        row = by_name[name]
        if row.get("private_file") != name + ".json":
            raise ValueError("noncanonical prefix file " + name)
        payload = json.loads((root / row["private_file"]).read_text())
        ids = payload.get("prefix_ids")
        if not isinstance(ids, list) or len(ids) != WARM_TOKENS:
            raise ValueError("prefix length is not 74400: " + name)
        if any(type(v) is not int or not 0 <= v < VOCAB for v in ids):
            raise ValueError("non-native Flash token id in " + name)
        actual = digest_ids(ids)
        if actual != row["prefix_ids_sha256"]:
            raise ValueError("prefix id SHA mismatch " + name)
        result.append({"name": name, "ids": ids, "input_tokens": len(ids),
                       "prefix_ids_sha256": actual})
    if result[0]["ids"] == result[1]["ids"]:
        raise ValueError("the two cold prefixes must be independent")
    return result


def load_tokenizer_identity(root):
    root = pathlib.Path(root)
    out = {}
    for name in ("tokenizer.json", "tokenizer_config.json"):
        path = root / name
        out[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out
