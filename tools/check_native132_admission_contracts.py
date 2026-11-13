"""Check host admission ownership only; no model or hardware simulation."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import ast
import hashlib
import importlib.util
import json

W=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,default=W/'work/fair-advantage-20261111/native-admission-cohort132/DeepInfer')
parser.add_argument('--output',type=Path,default=W/'evidence/migration103-20261112/native132-host-contracts.json')
a=parser.parse_args();D=a.source.resolve()
p=D/'deepinfer/model_executor/models/glm5_next/full_model/native_admission_cohort.py'
spec=importlib.util.spec_from_file_location('cohort_contract',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
checks=[]
x=m.NativeAdmissionCohort(False)
assert x.allow_admission(['a'],3,10)
x.after_step(['a'],True,11)
assert x.allow_admission(['a'],4,12) and not x.decoding
checks.append('disabled_mode_keeps_admission_open')
x=m.NativeAdmissionCohort(True)
assert x.allow_admission([],0,1)
x.admitted('a',1,2);x.after_step(['a'],False,3)
assert x.allow_admission(['a'],2,4)
x.admitted('b',3,5)
checks.append('prefill_phase_can_fill_existing_queue_without_wait_timer')
x.after_step(['a','b'],True,6)
assert not x.allow_admission(['a','b'],2,7)
assert not x.allow_admission(['b'],1,8)
x.after_step(['b'],True,9)
assert x.counts['cohorts_started']==1
checks.append('decode_freezes_admission_until_all_original_members_leave')
try:
    x.allow_admission(['b','intruder'],1,10)
except RuntimeError:
    pass
else:
    raise AssertionError('ownership mismatch not rejected')
checks.append('unexpected_cohort_member_fails_closed')
assert x.allow_admission([],2,11)
x.admitted('c',7,12);x.after_step(['c'],True,13)
assert not x.allow_admission(['c'],1,14)
assert x.allow_admission([],0,15)
assert x.counts['cohorts_started']==x.counts['cohorts_finished']==2
assert x.counts['queue_wait_ns_max']==5
checks.append('terminal_boundary_reopens_with_balanced_cohort_counts')
x=m.NativeAdmissionCohort(True);x.after_step([],True,10)
assert not x.decoding and x.allow_admission([],1,11)
checks.append('empty_or_prompt_completed_request_does_not_hold_empty_cohort')
http=D/'deepinfer/model_executor/models/glm5_next/full_model/sdk_http.py'
entry=D/'examples/glm5_full_model_offline.py'
for f in (p,http,entry):
    ast.parse(f.read_text())
h=http.read_text()
assert h.index('admission_allowed =')<h.index('while (admission_allowed and')<h.index('for pointer in llm.step():',h.index('for state in active.values():'))
assert 'admission.after_step(active, bool(llm.scheduler.decode_step)' in h
assert "'continuous_batching': not admission.enabled" in h
assert 'admission.allow_admission(active, waiting.qsize(), time.monotonic_ns())\n            if admission.enabled:' in h
checks.append('integration_uses_real_scheduler_phase_and_terminal_cleanup')
text=entry.read_text()
assert "p.add_argument('--native-admission-cohort', action='store_true'" in text
assert "a.native_admission_cohort and not (a.serve" in text
checks.append('flag_off_by_default_and_bound_to_existing_serving_recipe')
q={'created_at_utc':datetime.now(timezone.utc).isoformat(),'checks_passed':len(checks),'checks':checks,
   'scope':'host metadata/state-machine and source wiring only;not model arithmetic,performance or NPU simulation',
   'new_model_loads':0,'NPU_validated':False,
   'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
   'files_sha256':{str(f.relative_to(D)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (p,http,entry)}}
a.output.write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'checks_passed':len(checks),'NPU_validated':False}))
