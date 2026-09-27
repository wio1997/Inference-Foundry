# Run513 — independent V3.23 model review

2026-09-27. **PASS for the exact final bytes below.** V3.23 correctly carries the narrowly scoped Run508 observation and Run509 interval-service qualification without promoting any Bound. **All19 proof nodes remain false; all finite Algorithm/Resource, Hardware/Resource, Scheduling/Execution and Product endpoints remain null. Formal Current stays571.681 tok/s.** Run510 is intentionally outside this model's evidence cutoff.

| Reviewed artifact | SHA256 |
| --- | --- |
| `scripts/extreme_bound_calibration_v3_23.py` | `96d04506cea5779f4677487140146652243a9a085e195f5e9f6486d163c6fd57` |
| `run512/bound_calibration_v3_23.json` | `f7e9fc401b5e678305e321a4fbefc0d64640c9ae4c60eabba43a7d96ae736c88` |

`astra_model_audit.py` produced33 passing checks, `audit_results.json`, a byte-identical `rebuild.json` and `input_sha256_manifest.json` for21 reviewed inputs. Every recorded input hash was rechecked unchanged at audit end. Analysis was CPU-only; this reviewer changed no service, device, hot path, TaskCtl, generator or model.

## Scope and mathematical review

**Run508:** the added record preserves170 attention keys, empty params/handles/events, zero source-inferred zipped task-update/event-record iterations for40 selected calls in the fresh all8/five-cohort acquisition. The implementation is correctly named `AscendMLAImpl.update_graph_params`. Its promotion applies only to that selected loop body, under Run511's source/stable-list assumptions. Stream-context cost and other private work remain null. Typed four-output last writers, native model-terminal→callerR1 completion, necessary path duration and removable Product time remain null. Run502 structure and Run487 timing remain separate historical observations. No inferred event count is represented as observed native event completion.

The pinned Run511 review preserves60 completed1024-token requests with no client-recorded error and8 server ERROR lines in post-parent-shutdown context. The model relies on that bounded acquisition admission; it does not claim the server log is error-free or retroactively admit invalid Run507. Exact output length remains a scoped output-contract check, not a full model-semantic oracle.

**Run509:** for the same declared legal implementation class and full formal49,152-token interval, a proved in-window W_minus and a matching cumulative service envelope `W <= C_plus*T` imply `T >= W_minus/C_plus`. If only `W <= C_plus*T+B` is certified, the correct relaxation is `T >= max(0,W_minus-B)/C_plus`; **W_minus>B** is necessary for a positive floor. A finite `TPS <= 49152/T_star` requires a positive certifiedT_star. The new formula states these conditions correctly. No B, C_plus or actual necessary work has been invented.

The inherited8,388,608-operation BF16 projection remains a conventional-class candidate, not a certified unavoidable numerator. External retained-token lineage, formal-window freshness and true fresh evaluation remain open. Exact-board capacity, permitted-engine scope and cumulative service/boundary applicability remain open. A complete43-layer work census, full dependency graph or attainable schedule is not required for the first loose strict subset ceiling; matching necessary work and a genuine interval-applicable capacity envelope are required. Attainable Engineering performance remains a separate legal-schedule/formal-calibration claim.

## Review finding resolved in final bytes

The initial draft stated B in the ladder/formula but omitted the new condition from existing proof-node and Product scope ledgers, and its amended formula still cited only Run416. The final revision now:

- cites Run509 in the formula provenance;
- adds cumulative interval service/B to `matching_true_C_plus` and the hardware capacity certificate;
- adds W_minus>B to `strict_scope_units_join`;
- adds the same interval/boundary condition to Product strict-ceiling scope checks.

These are additions to existing unresolved obligations. The node set and certification values remain unchanged. No unresolved admission blocker remains in the reviewed final all-null model.

## Reproduction and challenge checks

The audit checks every V3.23 and inherited V3.22 pinned evidence hash, independently rebuilds the final JSON byte-for-byte, recomputes all19 proof-node states with the DAG validator, checks frozen contract/Current invariance, and recursively verifies all10 occurrences of the six recognized endpoint fields are null. It separately checks the new C_plus/B and Scheduling unknown fields.

All eight new pinned-input hash mismatches reject. Disposable **SYNTHETIC_ONLY** fixtures also reject zero keys, positive zip count, nonempty event list, missing summary row, wrong UUID, wrong rank count, failed final admission and failed cleanup final gate, even after their synthetic hashes are updated to reach semantic checks. The null-endpoint guard rejects each recognized endpoint key when nested with a numerical value; DAG wrong-contract and cycle mutations reject. Simple exact-rational examples exercise positive, zero and negative W_minus−B cases without supplying real capacity values.

The model remains a **human-reviewed obligation ledger**. `validate_dag` resolves declared statuses/dependencies, not evidence truth; its `missing` strings are review obligations, not executable hardware proofs. `require_null_endpoints` checks named keys rather than an arbitrary numeric schema. The production generator's exact hash pins bind the admitted reviewed data, but future new evidence or renamed fields still require source and scope review. This PASS certifies the present static all-null model, not a general automatic permission to promote future endpoints.
