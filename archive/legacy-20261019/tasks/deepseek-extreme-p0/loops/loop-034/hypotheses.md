# 假设记录

## H001 - ACTIVE

For the frozen greedy 48x32K-to-1024 c12 workload, each 12-request cohort can remain inside the Extreme-owned decode loop until its exact per-request output limit, drain device outputs once, and return one bulk completion frame to the serving control plane; this preserves request outputs while keeping ModelRunner and Scheduler off the per-cycle hot path.
