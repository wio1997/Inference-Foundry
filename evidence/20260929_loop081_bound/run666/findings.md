# Run666 — metadata Graph OFF/ON/OFF diagnostic screen

Same 48 measured request bodies and order; each arm completed warmup48 + measured48, 96 server POSTs, 64/64 FULL Target Graph runtime rows passed. ON captured/replayed metadata Graph on all measured cycles. Controller and source restoration gates passed.

| arm | diagnostic output tok/s | measured cycles | rank0 runtime ms/cycle |
|---|---:|---:|---:|
| OFF_A | 590.544 | 1222 | 56.532 |
| ON | 607.644 | 1221 | 55.240 |
| OFF_B | 590.437 | 1166 | 56.802 |

Per measured cohort, ON rank0 ms/cycle is below both controls:
- cohort 5: OFF_A 56.624, ON 55.150, OFF_B 56.639 ms/cycle
- cohort 6: OFF_A 56.328, ON 55.224, OFF_B 56.725 ms/cycle
- cohort 7: OFF_A 56.548, ON 55.307, OFF_B 56.650 ms/cycle
- cohort 8: OFF_A 56.651, ON 55.284, OFF_B 57.192 ms/cycle

Diagnostic ON improves client output by about 2.9% against both controls, but cycle counts differ due to acceptance variation. This is a positive screen, not a promoted formal TPS result. Run667 tests the same candidate under the frozen Run99 bench.py protocol, three measured repeats per arm.
