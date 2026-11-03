#!/usr/bin/env python3
"""Include seven actual initialization dependencies; keep prior ledgers intact."""
from pathlib import Path
here=Path(__file__).resolve().parent
code=(here/'build_source_ledger.py').read_text()
before="SNAPSHOTS=[ROOT/'frozen-source-snapshot',ROOT/'sampler-source-supplement/frozen-source-snapshot']"
after=before[:-1]+",ROOT/'contract-source-supplement/frozen-source-snapshot',ROOT/'initialization-source-supplement/frozen-source-snapshot']"
assert code.count(before)==1 and code.count("OUT=ROOT/'coverage-ledger'")==1
assert not (here/'coverage-ledger-v3').exists()
code=code.replace(before,after).replace("OUT=ROOT/'coverage-ledger'","OUT=ROOT/'coverage-ledger-v3'")
exec(compile(code,str(here/'build_source_ledger.py'),'exec'),{'__file__':str(here/'build_source_ledger.py'),'__name__':'__main__'})
