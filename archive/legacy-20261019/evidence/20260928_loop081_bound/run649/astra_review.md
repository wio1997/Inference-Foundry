# Astra High independent Run649 reuse review

Read-only review of Run602/606. **Run606 is admissible as a conditional same-W₀ Product/output/workload ledger; no new live acquisition is needed solely for P0.** The prior claim that P0 itself was missing was too strict.

The reviewer rehashed all Run606 Product manifest references: 281 SHA references, 280 normalized unique files; `client_admission.json` is referenced by relative and absolute path. The manifest covers 64 Runtime, 32 Basis, 32 worker Product, 48 API, four Scheduler, 98 measured/warmup client files, one Basis admission and two references to the same client admission. All referenced bytes matched. Independently recomputing 48 request ID chains gives pre IDs + accepted Runtime bulk IDs = Scheduler post IDs = concatenated API delta IDs, with 564 + 48,588 = 49,152. Runtime histories cover 297/309/310/298 = 1,214 cycles, and the same diagnostic Product wall is 83.39575357083231 s. Run602 is a separate 1,189-cycle/91.483702 s case.

Admission is limited to **final output, Runtime algorithm trajectory and diagnostic Product wall**. Basis does not certify initial KV generation or full ordinary prefill/seed work. Actual inputs/dispatch are sparse checkpoints; writer-ready, side-stream join, data versions and fixed primitive costs remain open. Synchronous diagnostic collection perturbs performance, so Run606 wall/589.383 diagnostic TPS cannot be transferred to Run99's formal 571.681 tok/s.

Priority: instantiate a parameterized Run606 whole-Product schedule case now; acquire only cost/ready edges at sensitive Target→DSpark→metadata→next Target transitions and an observer transfer bridge. Reuse admitted ID and cycle ledgers.
