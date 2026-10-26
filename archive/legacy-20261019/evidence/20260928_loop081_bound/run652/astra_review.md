# Astra High independent Run652 review

**SCOPED PASS** for rank-local Host marker versus same-current-stream Event interval inequality. The reviewer independently rehashed and recomputed64 packet pairs, checked event generation and issue order. Min/median/max Host5.790/6.265/7.001ms; Event0.798/0.873/4.054ms; paired Host−Event2.325/5.195/5.598ms. Host interval exceeds Event interval64/64. The reducer now explicitly checks upstream and packet status plus ON run tag, as requested by the reviewer; its numerical JSON SHA remains `8bb68bb7…`.

Physical statement: Host reaches the successor marker about6ms later, but the two markers execute less than1ms apart on the same current stream in the median sample. The Host interval does not become an equal exposed interval on that stream. If clock rates are accurate, the difference reflects a decrease in the Host-marker→device-Event delay between the two ends. Already queued device work is one consistent mechanism, **not proved to be the full cause**: async queueing, Event.record call delay, device waits and scheduling may contribute. The5.195ms difference is not idle or removable overhead.

Each packet's Event elapsed times use one common first Event across all sampled cycles; they are not reset each cycle, and subtraction is valid only within the same packet/stream. Extraction queries Event completion and later `elapsed_time` uses sync APIs after the existing terminal drain; observer initialization/export can still perturb Product wall. The result belongs only to Run638 ON and does not cover other streams, true producer-ready, collective completion or Run606/Run99 timing.

Read-only independent review; no live run.
