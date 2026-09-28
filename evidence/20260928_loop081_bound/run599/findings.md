# Run599 — source-typed Target-tail / Draft-context dependency split

This read-only gate pins 12 current source files and six evidence inputs, including Run597/598's admitted new W₀ and the historical R09/R21 reports at commit `db3beef223e0b5acd81ccccb444600c1b22aac8a`. The script checks source locations and hashes; Astra independently reviewed the actual call/data flow. Predicate strings are locators, not a proof of dependency absence or device readiness.

The conservative structural frontier is:

```text
Target forward and conditional FlashComm output gather
 ├─ hidden → Target LMHead → greedy acceptance → raw rejection count
 │                                     └─ masked/preserved last token → query seed
 └─ aux + old Target positions/slots → Draft combine → three-layer context KV
                                                 ↓
                       both branches join before Draft query consumer
```

`FixedTargetAdapter` exposes aux after its binding forward and before `compute_logits`. The binding forward includes any FlashComm output gather; a Python return is **not** a device-ready timestamp. The Draft's `combine_hidden_states` takes aux before its acceptance-dependent input preparation. The installed copy/expand kernel copies all context positions and slots before it reads rejected counts to build query positions and slots. Context KV projection reads combined hidden, old Target positions and per-layer context slots. `advance_state` masks parked counts for output progress while DSpark preparation receives **raw** acceptance counts, so query geometry and seed must remain separate in the DAG. Context stores may alias later query stores; context-store→query ordering remains required.

Run597 contains **49,523 active-slot prefix-decision witness positions**, including371 Host overshoot positions. For n≤7 active output tokens, the witness is n−1 matching draft predictions plus the first mismatch/recovery; for n=8, seven matches plus the bonus. This supports the observed active greedy output only. It does **not** count necessary fresh Target model evaluations, materialized full-vocabulary logits, parked raw acceptance, or Draft context/KV consumers. Formal Run99 has 49,470 /49,508 /49,471 staged active positions in its three repeats, but no same-state cycle basis; these remain non-Bound workload columns. Applying Run569 full Target-body coefficients to these counts would be unjustified.

Historical R09 measured an old DP2×TP4/EP8 Draft context phase with no repeated call, but its absolute Host span does not transfer to current TP8. R21's old Compressor/W4A8 overlap regressed under resource contention; current Run579–580 similarly rejects one coarse GMM/HCCL overlap, without ruling out this different split. Existing Run246/593 profiler scopes lack the typed producer→consumer ready/completion witness. Current Run99 Product residual was already quantified in Run153/238; this review does not repeat it.

**Next falsifiable measurement:** one all8, three-adjacent-cycle fixed-W₀ cut with real stream event timestamps for Target forward/gather completion, logits, acceptance, combined context, last context KV store, first Draft query consumer, Draft completion and next Target producer. Record Host submission, buffer storage/lifetime and Graph identity. Compare a no-event control with an event-probed arm under identical algorithm/output ledger; reject timing if perturbation is comparable to the prospective window, if any writer/consumer cannot be joined, or if clocks are subtracted across ranks. This measures one Scheduling window; it does not itself supply the complete all8 critical-path Bound or justify an overlap implementation. Resource compulsory work/traffic and cumulative mixed `C⁺/B` remain parallel gaps.

Formal Current **571.681 tok/s**. Finite Resource/Hardware, Scheduling/Execution and Product E2E endpoints, plus numeric Current→credible-limit gap, remain **null**.
