# Run260: H6 versus H6 + H4

Frozen verdict: INCONCLUSIVE. 20 requests, same workers/common library; H6 mode1 throughout. Four natural-EOS 23-token complete PD requests (58 external KV tokens), eight timed 2334/8 short requests. New model correctness and all-rank witnesses passed.

| pair | D TPOT ms | D wall s | D saving s | P contribution s | PD wall s |
| --- | --- | --- | --- | --- | --- |
| A1->B1 | 189.681528 to 191.119885 | 4.496287060 to 4.503991249 | -0.007704189 | 0.052647897 | 4.986417754 to 4.941474046 |
| A2->B2 | 189.960167 to 200.370573 | 4.506058690 to 4.771715174 | -0.265656484 | 0.025140077 | 4.976747848 to 5.217264255 |

Frozen max within-mode D wall drift: 0.267723925s. Complete cumulative work matched: False; short work matched: True. Exact raw inputs are hashed in comparison_reduced.json and download_identity.json. SSE grouping is not an ordered ModelRunner step trace.

H4 short medians improve 6.06/9.10%, but complete B1 has 12 drafts/11 accepted versus A 11/11; B2 matches cumulative work and is slower by 0.265656s. H4 stays off, H6 retained. No post-hoc normalization or retry.

Current=None and formal product promotion pending full API, official SLA, stability and full-stack E2E. Research gain is retained under the user override. No cumulative stock-to-stack gain measured; independent percentages are not added. Controller completed, phase0. No profile, scan or kernel change.
