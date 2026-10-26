# GLM-5.3-Flash Ascend company submission — 2026-10-26

This is a source/evidence publication round, not a new tuning run. The containing Foundry commit is the publication identity; actual measured model commits are separately fixed in the source manifest and compact results. No new Ascend/model/profiler execution was performed.

- [Complete per-service numeric results and limitations](results.md)
- [Structured compact evidence](compact-evidence.json):47 original service/attempt records,26 canonical performance services,86 within-service blocks; no best-only filtering.
- [Source reconstruction check](tests/source-reconstruction.json): four temporary-index Git applications and exact48/52/53/54 runtime checkpoints; not model verification.
- [Publication CPU correction history](tests/offline-check-history.json)
- [Runtime patch provenance](../../../../manifests/source-provenance.json)
- [Native/source reproduction boundary](../../../../reproduction.md)

Original round IDs, model/controller/evaluator identities, negative/inconclusive results and raw-source hashes remain separate. The R3 profile automatic stop/export correction is preserved; repeated STOP500 is not called capture failure. R2, initial numerical failures, preliminary seed mismatches, R4 limited C4 gain, R5 small observations and R6 mixed tradeoffs are retained.

This public package does not include private raw prompts/SSE/server traces, connection configuration, authentication material, complete environments, weights or prebuilt native libraries. Source reconstruction is available from the included patches and public official base. Exact prior fused-B execution additionally needs the controlled SO whose strict hash guard remains intact; no bit-identical fresh rebuild has been established. Original private raw evidence and complete native/code-generation closure are not publicly reproduced, so full raw benchmark/causal/precision revalidation requires further authorized access and work.
