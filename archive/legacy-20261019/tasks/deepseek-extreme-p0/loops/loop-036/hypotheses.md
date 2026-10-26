# 假设记录

## H001 - ACTIVE

The fixed c12 target metadata updater performs a per-cycle NPU-to-host max().item synchronization to choose max_local_seq_len, contributing materially to the measured 8.673 ms metadata stage. A conservative cohort-fixed bound can remove this synchronization while preserving exact target metadata and long-run KV/DSpark semantics.
