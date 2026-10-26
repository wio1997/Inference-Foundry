import hashlib,json,pathlib
paths=['AGENTS.md','MISSION.md','PROJECT_STATE.md','PERFORMANCE_MAP.md','ACHIEVABLE_BOUND.md','RESULTS.md','runtime/fixed_serving.py','scripts/bench.py','evidence/20260926_loop074_refill/run339/findings.md','evidence/20260926_loop074_refill/run340/findings.md']
base='evidence/20260927_loop079_identity/'
paths += [base+p for p in ['run462/astra_hardware_certificate_review.md','run487/findings.md','run508/findings.md','run510/astra_terminal_abi_source_review.md','run517/witness_source_locations_sha256.json','run519/astra_v3_25_review.md','run520/astra_compulsory_census_plan.md']]
out=pathlib.Path(base+'run525')
manifest={p:{'sha256':hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest(),'bytes':pathlib.Path(p).stat().st_size} for p in paths}
(out/'review_inputs_sha256.json').write_text(json.dumps({'scope':'reviewed evidence/source; not expansion of every raw leaf','inputs':manifest},indent=2)+'\n')
for p in sorted(out.iterdir()):
 print(hashlib.sha256(p.read_bytes()).hexdigest(),p)
