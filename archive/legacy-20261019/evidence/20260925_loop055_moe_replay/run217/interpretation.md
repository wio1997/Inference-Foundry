# Run217 distinct-input MoE graph replay

The legal 48-request warmup and two 12-request, 1024-token cohorts completed 72/72. All 48 rank/cohort Runtime records passed; Host mirror remained exact and FULL target handoff was used. Production continued to use eager MoE outputs.

The single layer0 first88 prefill MoE graph captured on all eight ranks in A and replayed on all eight in B. Each rank had a different A/B input hash, and both graph output hashes changed across A/B. In each phase the shared output was bit-exact against same-input eager. Routed output maximum absolute graph/eager difference was 0.0078125; the eager self-control maximum was 0.015625. A few individual ranks had graph difference 0.0078125 versus eager self 0.00390625, so this is numerical-noise parity, not per-rank dominance or exact equality.

B diagnostic median wall times were eager 4.225 ms, input refresh 0.139 ms, graph replay 1.621 ms, and refresh plus replay 1.763 ms for the one layer. These measurements include synchronization and coexist with hashes and extra eager controls. They establish a plausible local saving, not a stage or E2E performance result. A/B client throughput is also diagnostic and must not be compared to formal Stock or Extreme figures.

Next: test whether replacing only this one layer in serving, after a captured graph has warmed, preserves full Runtime correctness and lowers low-overhead prefill stage wall time. Include matched eager controls and account for graph capture amortization and output alias ownership. Reject this path if stage saving is absent or numerical differences break correctness; do not extrapolate one-layer wall time to 43 layers.
