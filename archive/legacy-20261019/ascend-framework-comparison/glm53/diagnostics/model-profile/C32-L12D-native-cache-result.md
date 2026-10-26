# C32 L12D native-P cache result

Single canonical C32 cohort: 128 formal requests, 32 warm logical pairs excluded
from the clock, 80,000 input tokens, 600 output tokens, 320 backend HTTP attempts.
All three rows below are historical or new measurements on that same protocol, and
all three fail aggregate SLA.

| Loop / framework | Output TPS | Full clock (s) | TTFT P50 (s) | TPOT P50 (ms) | SLA pass |
| --- | ---: | ---: | ---: | ---: | --- |
| SGLang L10 (qualified historical) | 308.9149589657814 | 248.61211078 | 29.852490126155317 | 36.55125868222202 | false |
| SGLang L12D native P (accepted) | 389.8740711045039 | 196.98668286 | 17.757243875414133 | 36.65860450806373 | false |
| vLLM C32 (accepted baseline) | 530.8125387945174 | 144.68384672 | 21.16201964020729 | 19.15265580107503 | false |

Sources: L10 `results/sglang-C32-L10-target-graph-v1-measurement-v1.json` SHA
`f2eecea1a50786af063d16acca32dab3fcb35293c18d2476d3428cd58ce7ea2d`; L12D
`results/sglang-C32-L12D-native-HiCache-v1-measurement-annotated-v2.json` SHA
`bab30fb0e0cdec779eb26e833142549e3bdd0b45b4b841f3fe9ba42a519173d7` (source v1
`39e31dc3cfb97b5d59553dce69137b01cfcf6160c1ae95837ad6a6fd9a0ffa3d`); vLLM
`results/vllm-formal-C32-v2-measurement-v1.json` SHA
`5933a65bdd6d90514e6fe975c08df68bf18a3f2a655a4279942e05c2507fd596`.

Relative changes from the new native-P row:

| Reference | Output TPS | TTFT P50 | TPOT P50 |
| --- | ---: | ---: | ---: |
| L10 | +26.20757259854496% | -40.51670798525417% | +0.2936857162019413% |
| vLLM C32 | -26.551457885694774% | -16.089087065792963% | +91.40219972003096% |

Native P kept TPOT stable against L10 while reducing TTFT P50. The largest
remaining gap is D decode, where vLLM is still materially faster. Aggregate SLA is
false for both TTFT and TPOT. This is one cohort against qualified historical L10,
not global optimality and not a universal causal or precision claim.

The cache paths differ: SGLang records MC G2H followed by native H2D, while vLLM
uses direct HBM `batch_get_into_layers`. No blanket capability or precision parity
is claimed. The D-only L13 two-step diagnostic is planning-only and
not runtime-authorized.
