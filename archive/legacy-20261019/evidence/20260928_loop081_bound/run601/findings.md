# Run601 — Product handoff and delivery source sites

This read-only gate pins nine currently installed source files and five existing evidence summaries. The exact-source anchors identify where a single future W₀ must be joined; they do not certify absence of other producers, asynchronous completion, or a time Bound.

```text
request admission/prefix hit → ordinary residual-prefill execute(s)
  → Target _model_forward / sample → DSpark _propose / seed / context-KV / query
  → final 96-token execute entry → Extreme handoff input clones/binds
  → FixedCohortServing first Target → Runtime cycles / existing final sync
  → ModelRunnerOutput → KVDeliveryScheduler bulk append/stop
  → EngineCoreOutput/RequestOutput → API stream output-token accounting
  → ASGI SSE yield → client SSE receive/usage/DONE
```

The ordinary runner invokes `self._model_forward`, then proposes Draft tokens after sampling; the proposer combines Target hidden states, prepares DSpark inputs, stores three-layer Draft context KV and runs Draft. The last ordinary proposal's Host return alone does not identify all 12 handoff slots' producers. `draft_token_ids_copy_stream` waits for the default stream before copying token IDs **when that branch is active**; the common async branch can skip it. This ordering is limited to the covered copy path and does **not** prove all Target/DSpark KV or metadata side-stream writers complete at Python return. At the final 96-token execute entry, input preparation/reorder/correction occurs before the installed Extreme branch binds cloned input tokens/positions/seq lengths/slots into `FixedDecodeState`; it starts serving before entering the ordinary Target forward. `build_extreme_runtime` return is Host construction, not a device-ready or initial-cache-freshness certificate.

The installed scheduler's `CachedRequestData.num_output_tokens` adds `req.num_output_tokens` and `req.num_output_placeholders`. The KVDelivery scheduler appends the Extreme bulk output through its special path, checking stop after each token and truncating the accepted bulk prefix. API chat streaming counts `len(output.token_ids)` before yielding serialized SSE, including token-ID deltas whose text may be suppressed; the existing instrumented client captures received payload and DONE timestamps. SSE yield is not socket delivery. These are distinct ownership/clock boundaries. Neither Run341's cached count nor Run597's Runtime retained1024 tells how many pre-handoff tokens reached the client or which initial generated-token prefix the bulk path owns.

Minimum next live acquisition: reuse Run597's compact full-cycle basis and Run341's request/call chronology in one new W₀, then record (a) scheduler actual prebulk token IDs/count and placeholder count, postbulk accepted IDs; (b) API per-output token IDs/count and yield time, joined to existing client receive time; (c) all8 ordinary prefill/DSpark producer Host and participating-stream events, final handoff input/state identity and first Target submission. Writer-stream identities and existing waits must be audited; current-stream events alone stay **partial**. Preallocate marks, defer serialization to existing after-sync points when possible, charge any publication work, and compare observer control before interpreting durations. The fixed DSpark7 algorithm and acceptance/cycle trajectory are measurements, not an optimization target.

Resource companion: capture the actual initial prefix/cache generation, Target/Draft KV/state and any retained activation/result availability; then join one sparse required consumer/freshness witness to the full basis. This is a declared execution-class necessity study, not a universal arithmetic or HBM traffic proof. Exact-board cumulative compute/HBM/HCCS `C⁺/B` certificates remain open.

Formal Current571.681 tok/s. Strict finite Resource/Hardware, Scheduling/Execution and Product E2E endpoints and numeric Current→credible-limit gap remain null. See Astra's Product and Resource reviews adjacent to this run.
