"""Complete fixed current-instance controls before allowing finite service timing."""
from pathlib import Path
import datetime,hashlib,importlib.util,json,subprocess,sys,time
r=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('transport',r/'run_prefix_count_service_v1.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
start=time.monotonic()
while not(r/'rope-step-reuse-correctness-outcome-v1.json').exists():
 assert time.monotonic()-start<300;time.sleep(5)
q=json.loads((r/'rope-step-reuse-correctness-outcome-v1.json').read_text());assert q['completed'] and q['all_selected_pair_and_owned_copy_bitwise']
details={};sources=[]
if q['unequal_cases']:
 control=m.collect(r/'control_rope_step_reuse_quality_v1.py','rope_step_reuse_quality_controls_v1','rope-step-reuse-quality-controls-v1',timeout=300)
 assert control['completed'] and all(z['numeric_pass'] for z in control['rows'])
 details=control['membership'];p=next((r/'rope-step-reuse-quality-controls-v1').glob('*/metrics.json'));sources.append(dict(path=str(p.relative_to(r)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
proceed=all(v['both_initial_and_repeat_candidate_in_observed_native_set'] for v in details.values())
out=dict(proceed_finite_formal=proceed,reviewed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),initial_full64_equal_cases=q['quality_full64_equal_cases'],cases=26,finite_controls=details,sources=sources,scope='Current exact loaded instance, fixed inference parameters and streaming request format. Both initial and repeated candidate full64 sequences must be covered by current native outputs. No general QA, harmless-rounding explanation or service gain claim.')
p=r/'rope-step-reuse-quality-reviewed-v1.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
if proceed:subprocess.run([sys.executable,str(r/'run_rope_step_reuse_formal_v1.py')],check=True)
