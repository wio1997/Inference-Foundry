# Run280 — RoPE matched comparison, confounded work

COMPLETED 2026-10-07T12:04:34.845069Z; spec `0fee2cd3c9ade2468f57e8126fcbd7f50415aa8df4434c48d732c5b315a2bccd`, 94 source pins. Same resident workers, H6/H5/target FULL fixed, H11 off. Ten frozen requests: priming warm; A1/B1/A2/B2 each warm plus natural23 EOS; retained old warm. No recovery/reload or favorable retry.

| Complete request | D wall s | D SSE TPOT ms | PD wall s | drafts / draft tokens / accepted / position0 |
| --- | ---: | ---: | ---: | --- |
| A1 original | 0.902544510 | 27.263910136 | 1.260371573 | 11/11/11/11 |
| B1 dense | 0.879895843 | 26.565069818 | 1.246023832 | 11/11/11/11 |
| A2 original | 0.895238559 | 27.293254136 | 1.409098966 | 11/11/11/11 |
| B2 dense | 0.933466052 | 28.908099409 | 1.292461917 | 12/12/11/11 |

All four exact23 IDs/EOS/usage, zero invalid draft tokens and all16 transfer pass. Ten unique cache salts; complete-request P and D local58 queries/0 hits, P external58/0, D external58/58. The D external hits describe PD transfer reuse, not the formal93% prefix-hit workload.

D savings +22.648667/−38.227493ms; A drift7.305951ms. PD savings +14.347741/+116.637049ms include P savings −8.300925/+154.864542ms, with PD A drift148.727393ms. Work differs in B2 and D improvement is not repeated. **INCONCLUSIVE; measurement comparison inadmissible; H12 off/not added to research stack.** No posthoc work normalization, cherry-picking or unchanged rerun.

[Independent final reduction](measurement_reduced.json) distinguishes raw integrity=true from admissible comparison=false. Frozen `reduce_matched_raw.py` was preserved unchanged despite its overly broad metadata flag; `independent_final_reduction.py` applies accurate work/admission flags. Inputs were fetched once with `FETCHED_TERMINAL_INPUTS.json` hashes. Terminal transition9, healthy/idle same all16, H6/H5/FULL retained, H11/H12 off, no recovery. Current/fullAPI/formal SLA remain open; no code gain claimed.
