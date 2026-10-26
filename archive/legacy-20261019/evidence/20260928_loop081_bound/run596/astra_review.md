# Run596 independent Astra short review

## Verdict: SCOPED PASS

Extraction/replay separation is real. `extract_host_events` reads historical full-trace positions before replay; `compact_basis` retains initial state and accepted/count/next-draft histories. `replay` advances state from only that basis and explicit events. The optional oracle supplies assertions only: no oracle value enters reconstructed state.

Independently executed all **32 rank/cohort cases with oracle=None**, then with oracle assertions. Both paths pass and match the summary: 1,206 cycles, 115,776 physical rows, **98,496 active rows**, 17,280 parked rows. Do not multiply logical work by eight replicated ledgers.

## Evidence and negatives

All 64 trace/runtime hashes match. Independently checked rank/cohort identity, runtime pass, cycle coverage and [1024]*12 output counts for all cases. Basis + events + request IDs hash identically across all eight ranks per cohort, stronger than the main script aggregate-only consensus.

The full unmodified negative fixture passes first. Missing/shifted events fail oracle position; duplicate fails event uniqueness; wrong anchor fails anchor validation. All four independent reruns reject for the intended reason. No truncated-fixture false pass remains. Missing/shifted tests use the full trace oracle; they do not prove an oracle-free collector detects every missing or mistimed event from aggregates alone.

## Scope limits

Events still originate from old trace position drops, not actual Host instrumentation. Extraction allows at most one rewind per slot; replay requires initial+960 anchor after >=1024 staged samples. This is the admitted fixed-geometry class, not a generic scheduler. Final-cycle parking without a following Target is not extractable and is unnecessary for this input replay; this is not a complete Host control ledger.

Full Draft/KV ledger false, Run99 same-state false and strict endpoints null are correct. Target input reconstruction and next-draft lineage do not prove prefill/seed/KV provenance, Draft context/query/Markov dependencies, fresh semantic necessity, cache dedup, compulsory traffic or attainable runtime. Current Formal stays 571.681.

The revised main now checks runtime rank/cohort/pass/generated/cycle/request identity and full all8 basis/event/request SHA equality. The former summary is preserved as summary_pre_source_join.json. I rechecked the new script, all 64 hashes, all 32 oracle-free/oracle-checked replays and four negatives; numeric results remain unchanged. Pin the imported helper and validate standalone entry schemas in the future collector.

## Identity and execution

Only CPU imports and existing file reads were executed; no service/NPU/device query. Parent source and summary unchanged. Machine-readable checks: `astra_checks.json`.

- script_sha256: `5ec0efa5abfbf3cb275f1eec71ebc66fdd78d6552eed2626f19ff104745c4437`.
- imported_run594_sha256: `ffc6832018fa7097037e4f615ca63bd59eeb522541b474041794036aec419fe6`.
- summary_sha256: `9b29ea967d9951bc90dfa72662488070a3648b5b3f9396be06f2caece62678f1`.
