# GLM-5.3-Flash SGLang B103 persistent storage C2 checkpoint

This version passes the affected native C2 PD regression on the fixed B103 image and owned nodes46/47. Four real two-sequence cohorts produce eight complete PD pairs. All output IDs, text, natural EOS and repeats match the frozen reference; selected-token log probability differences remain within0.25 (maximum0.192514509 against R080).

The D index caches and FP32 tails have independent exact logical storage before native SDK registration. Actual model compression and scoring use those same registered pointers. No full cache mirror or per-step cache copyback is used; nine small transient inputs are still compacted. Compression/indexer Triton kernel mathematics are unchanged. The sole runtime source difference from R170 is a diagnostic C2 capture predicate accepting actual overlapping positions24/25; C2 batching is chosen by the native scheduler after the test readiness gate joins requests.

Evidence:1120 SDMA packets,19968 ranges,2261755392 bytes,4352 imported KDA layer boundaries,8704 exact convolution/SSM hashes and704 live FP32 tails. Both native services and containers exit0,32 native SDK destroys and320 ordered shutdown phases complete, and all8 cards on each node are idle. The P writer stops before D cleanup is permitted.

Scope is bounded eager C2 with24/24/32/2379 token inputs and context4096. Diagnostic CPU metadata checks, state hashes and ABI audits remain. Graph, MTP,80K/600, vision, arbitrary service traffic and performance are unqualified. This source snapshot does not establish a performance baseline. The original failed runs and numerical thresholds remain unchanged.

The manifest binds all44 runtime overlays and exact argv/configuration. Acceptance is independently derived from actual request and archive receipts in the R176 run. Raw model data and private long-token datasets are excluded.
