# Run261: H6 versus H6 + H5

Frozen verdict: POSITIVE. 20 requests, same workers/common library; H6 mode1 throughout. Four natural-EOS 23-token complete PD requests (58 external KV tokens), eight timed 2334/8 short requests. New model correctness and all-rank witnesses passed.

| pair | D TPOT ms | D wall s | D saving s | P contribution s | PD wall s |
| --- | --- | --- | --- | --- | --- |
| A1->B1 | 190.358075 to 177.462912 | 4.508800573 to 4.194096501 | 0.314704072 | 0.090187829 | 4.986372054 to 4.581480153 |
| A2->B2 | 190.970420 to 177.535348 | 4.525812973 to 4.211829431 | 0.313983542 | 0.090886479 | 4.989539434 to 4.584669413 |

Frozen max within-mode D wall drift: 0.017732930s. Complete cumulative work matched: True; short work matched: True. Exact raw inputs are hashed in comparison_reduced.json and download_identity.json. SSE grouping is not an ordered ModelRunner step trace.

H5 incremental D TPOT reduction on H6 is 6.77/7.04%; D wall reduction 6.98/6.94%. Both D savings exceed frozen drift. All four complete signatures are 11/11/11/11/0; all eight short signatures 4/4/4/4/0. IDs, EOS, external KV and all 16 cache/helper/source/mapping witnesses pass. H6+H5 retained, MC2/event1, H4 off. PD wall reduction 8.12/8.11% includes separately reported P contribution around 90ms, not credited to H5.

Current=None and formal product promotion pending full API, official SLA, stability and full-stack E2E. Research gain is retained under the user override. No cumulative stock-to-stack gain measured; independent percentages are not added. Controller completed, phase0. No profile, scan or kernel change.
