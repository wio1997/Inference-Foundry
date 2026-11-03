#!/usr/bin/env python3
"""Use the identified same-image stage after the baseline-source read failed."""
import datetime
import json
from pathlib import Path
W=Path(__file__).resolve().parent
readiness=json.loads((W/'owned-environment-readiness.json').read_text())
assert readiness['stage_running'] and not readiness['model_or_npu_load_started']
assert (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(readiness['utc'])).total_seconds()<90
code=(W/'read_installed_dump_contract.py').read_text()
old="'wio-vllm-v41-repro169-20261028'"
assert code.count(old)==1
code=code.replace(old,"'wio-vllm-v41-repro169-stage-20261028'")
work=W/'stage-dump-contract';work.mkdir(exist_ok=False)
exec(compile(code,str(W/'read_installed_dump_contract.py'),'exec'),{'__file__':str(work/'read_installed_dump_contract.py'),'__name__':'__main__'})
