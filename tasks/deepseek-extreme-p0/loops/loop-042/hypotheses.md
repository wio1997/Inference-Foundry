# 假设记录

## H001 - ACTIVE

A persistent inter-rank phase offset entering each c12 target cycle makes early TP ranks wait in the first reduce-scatter; synchronizing or pacing launch without changing token semantics can remove exposed wait from the cohort critical path, provided the late-rank arrival itself can move earlier.
