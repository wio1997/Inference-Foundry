#!/usr/bin/env python3
"""Validate expanded MoE references/open-state invariants; not model acceptance."""
from pathlib import Path
here=Path(__file__).resolve().parent
code=(here/'verify_recovery_checkpoint.py').read_text()
old="[('tail-branch-review-delta.json','branch_id'),('dspark-branch-review-delta.json','id')]"
new="[('tail-branch-review-delta.json','branch_id'),('dspark-branch-review-delta.json','id'),('initialization-branch-review-delta.json','id'),('moe-branch-review-delta.json','id')]"
assert code.count(old)==1 and code.count("source_files']==566")==1 and code.count('total==23')==1
code=code.replace(old,new).replace('coverage-ledger-v2','coverage-ledger-v3').replace("source_files']==566","source_files']==573").replace('total==23','total==47')
code=code.replace("'DSPARK-CONTRACT-REVIEW.md']:","'DSPARK-CONTRACT-REVIEW.md','CROSS-ITERATION-REVIEW.md','INITIALIZATION-AND-TAIL-REVIEW.md','MOE-COUNTER-CONTRACT.md']:")
code=code.replace('recovery-checkpoint-verification.json','recovery-checkpoint-v4-verification.json')
assert not (here/'recovery-checkpoint-v4-verification.json').exists()
exec(compile(code,str(here/'verify_recovery_checkpoint.py'),'exec'),{'__file__':str(here/'verify_recovery_checkpoint.py'),'__name__':'__main__'})
