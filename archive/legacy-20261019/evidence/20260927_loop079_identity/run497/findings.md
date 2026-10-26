# Run497 — CPU first-position lemma

Run495 Astra High design selected a minimal Resource witness. The pinned-source CPU script executed the actual `greedy_accept` function for all 128 seven-position equality patterns and observed every leading accepted length 0–7. In all cases, sampled position 0 equalled Target argmax position 0. Inactive count, wrong-column, Runtime-prefix and conditional Scheduler-prefix negative controls passed. No NPU workload or service was started.

This establishes a source-level first-position identity for the frozen greedy path. It does not establish that a chosen token is a new external output in the formal interval: exact request/generation, Scheduler pre-append count, clipping, raw-token publication and client receipt remain unjoined. A single conventional 4096×1024 BF16 `wo_a` group corresponds to 8,388,608 operations only under the declared ordinary dense and fresh-evaluation class. The exact-board maximum C⁺ remains missing. Therefore no finite Resource/Hardware, Scheduling or Product endpoint is promoted; Formal Current stays 571.681 tok/s.

Artifact `first_position_cpu.json` pins five source hashes and reports the 128 cases and explicitly open evidence gates. Script `scripts/loop079_first_position_cpu_run497.py` SHA256 de4e48b90dcd5635df2baf319c7cf143a6120238d25d4452f7014869c8ff72dc.
