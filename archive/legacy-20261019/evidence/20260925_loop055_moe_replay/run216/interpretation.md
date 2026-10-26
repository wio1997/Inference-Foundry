# Run216 single-layer MoE shadow graph result

The corrected wrapper actually armed on A/B. Legal frozen Extreme warmup48+A12+B12 requests all72 succeeded with max_tokens1024. Six observed Runtime cohorts passed on all8 ranks with exact Host mirror and FULL target graph. The service stopped and borrowed source SHA values were restored.

For model.layers.0.mlp.experts at local11x4096 BF16, all8 ranks captured one shadow NPUGraph during A and replayed it during B. The normal eager custom-op output was returned to the model. No DSA/KV operation was captured. A capture wall was46-54ms/rank and graph-related memory delta634368 bytes/rank; B replay diagnostic wall including synchronization was0.694-1.617ms/rank, but excludes input refresh and includes perturbing controls. These are not product-stage savings.

On all8 ranks and both A/B, shared output was bit-identical to eager. Routed output was not bit-exact under eager self-replay: maximum absolute eager-versus-eager difference was0.0078125; graph-versus-eager maximum was also0.0078125. One rank/phase eager self difference was0.00390625 while graph difference was0.0078125, so equality to the self-noise maximum is not universal at rank granularity. This passes only a bounded numerical feasibility screen; full acceptance/state correctness is untested.

The run did not save hashes of A/B input tensors or graph output, so it does not directly demonstrate that B carried distinct values from A. The next Run must save input/output hashes and verify changed inputs change graph results while staying within eager self-replay error. Then measure a low-overhead one-layer A/B/A stage with the buffer copy charged to graph, before any wider capture bank or formal E2E.
