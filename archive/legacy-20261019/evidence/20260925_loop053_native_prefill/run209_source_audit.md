# Run209: first88 prefill metadata producers and write ownership

Read-only source audit against the borrowed vllm-ascend tree, following Run208 first eager prefill fingerprints. No service, patch, graph capture or performance claim.

## Dynamic input producers

- worker/model_runner_v1.py:1248 computes fresh per-KV-group slot mapping from request block table and positions. Lines 4734-4895 derive attention metadata from scheduled tokens, query/sequence lengths, per-group block tables and slots; lines 4950-5007 build DSA group metadata.
- attention/context_parallel/dsa_cp.py:283-347 converts slot mappings and builds DSA metadata from request query lengths, positions/RoPE, sequence lengths and block tables. Lines 667-815 derive CP-local metadata, CPU maxima, SAS and QLI inputs. Lines 929-1045 run SAS/QLI metadata operators and copy results into reused fixed buffers.
- Run208 confirms the reused addresses are not constant values: per rank 107/107 observed tensor layouts and addresses match A/B, while eight selected small integer hashes differ. The changes include SWA/compressor slot mapping, SAS/QLI metadata and attention block table.

## Stateful writes and capture boundary

- dsa_cp.py:1522 writes SWA KV through scatter. Lines 1550-1574 invoke compressor with state cache and state block table, then scatter compressed KV. For c4, lines 1677-1740 also invoke indexer compressor, update indexer state, K cache/full cache and scale cache. These are real destination changes for the next request and later tokens.
- dsa_cp.py:1047-1064 rejects prefill in build_for_graph_capture. worker/model_runner_v1.py:4950-4977 routes DSA metadata through normal build even when for_cudagraph_capture is requested. No existing prefill replay contract supplies refreshed dynamic metadata and reversible KV/compressor/indexer writes.
- A same-shape graph needs all dynamic input buffers refreshed before replay and exact capture warmup write-set restoration. Run208 fingerprints cover only selected reachable fields to depth7; they do not prove the full input ABI, all tensor values, or side effects.

## Sol decision

The first88 shape is repeatable, but the current evidence does not isolate a >=0.5s/cohort removable Host segment or a bounded safe graph-capture implementation. A whole-forward prefill graph would need a new ownership contract across runner, DSA-CP and multiple cache families, with capture warmup restoration and same-state differential testing. This is disproportionate to the unproven savings, so pivot Loop053 from prefill graph feasibility. Preserve the fingerprint/source map for a later architecture design if a measured prefill critical-path bound justifies it.

Next revisit the target cycle, where real-weight GMM compute and exposed communication were both candidate gaps in Run116. Separate profiled kernel sums from product-exposed time; design a low-overhead discriminator before selecting an implementation.
