#!/usr/bin/env python3
"""Extend the source denominator without replacing the earlier frozen ledger."""
from pathlib import Path

here = Path(__file__).resolve().parent
code = (here/'build_source_ledger.py').read_text()
old = "SNAPSHOTS=[ROOT/'frozen-source-snapshot',ROOT/'sampler-source-supplement/frozen-source-snapshot']"
new = old[:-1] + ",ROOT/'contract-source-supplement/frozen-source-snapshot']"
assert code.count(old) == 1
assert code.count("OUT=ROOT/'coverage-ledger'") == 1
code = code.replace(old, new).replace("OUT=ROOT/'coverage-ledger'", "OUT=ROOT/'coverage-ledger-v2'")
assert not (here/'coverage-ledger-v2').exists(), 'Do not overwrite a historical denominator'
# Execute only the local AST indexer; never import the frozen target modules.
exec(compile(code, str(here/'build_source_ledger.py'), 'exec'), {'__file__':str(here/'build_source_ledger.py'), '__name__':'__main__'})
