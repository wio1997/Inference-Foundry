from pathlib import Path

root=Path('/data/wio/Inference_Foundry')
blocks={
'PROJECT_STATE.md': '''
## Run570 parked-row Resource census

The source/admission-pinned Run566 A0 diagnostic has all8 identical parking boundaries across measured cohorts5–8. Run570 and independent Astra review find per rank 115,776 current physical Target input rows, 98,496 rows while slots remain active and 17,280 (14.92537%) rows after slots park, under the current fixed96 Target-8 evaluation class. Source position increment/rewind/freeze and Runtime staged-count joins passed; service remained stopped. This is current diagnostic geometry, not compulsory work, available wall saving or formal Run99 W₀. It sharpens the conditional Resource numerator only; exact-board capacity, complete necessary work/traffic and all8 Scheduling still open. Formal Current571.681tok/s and all finite Bound endpoints remain null. See Run570 findings, summary and PK-071.
''',
'PERFORMANCE_MAP.md': '''
## Run570 observed parked-row distinction

Run566 A0 all8 row census separates 98,496 active-slot rows from 17,280 parked-slot rows within 115,776 current physical Target inputs/rank over four diagnostic cohorts. The parked share 14.92537% is a current geometry observation. Whether variable-active-row execution reduces compute/HBM or critical-path wall requires compatible Graph/layout, same fixed W₀ and all8 mixed-resource measurement; no such gain or Product Bound is inferred. The Resource gap remains dominated by formal W₀ necessary-work/traffic and matching C⁺/B uncertainty.
''',
'ACHIEVABLE_BOUND.md': '''
## Run570 conditional active-row calibration

Run570 source-pinned A0 diagnostic row census gives 115,776 physical, 98,496 active and 17,280 parked Target input rows/rank (14.92537% parked) across measured cohorts5–8, with all8 exact agreement. A parked slot's `num_computed_before` first follows raw acceptance increments, then rewinds to initial+960 and freezes; pre-park counts match Runtime staged output. This supports a conditional current Target-8 active-row numerator, not a mathematical compulsory-operation count, a proportional memory-traffic saving or a Schedule with lower wall time. Run99's formal W₀ is a different untraced trajectory. All finite strict Resource/Hardware, Scheduling/Execution and Product endpoints and numerical Current→limit distance remain null; Current Formal571.681tok/s. See Run570 findings, summary and PK-071.
''',
'RESULTS.md': '''
## Run570 offline row census (2026-09-28)

No new model run. All8 admitted Run566 A0 diagnostic traces imply per-rank 115,776 current Target input rows, including 98,496 active-slot and 17,280 parked-slot rows. Astra independently verified source formulas and output files. This is not formal Run99 W₀, compulsory work, a measured saving or E2E result. Formal Current remains571.681tok/s; finite strict Bound endpoints remain unknown.
''',
}
for name,block in blocks.items():
 p=root/name;s=p.read_text();marker=block.strip().splitlines()[0]
 assert marker not in s
 p.write_text(s.rstrip()+'\n\n'+block.strip()+'\n')
print('updated',','.join(blocks))
