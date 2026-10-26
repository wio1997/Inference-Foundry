# Independent Astra read-only review

Independent raw packet recount: 32 packets from cohort5–8 × rank0–7, 64 adjacent transitions. Previous `draft_commit_after` current-stream Event query true counts are cycle_begin 0/64, prepare_target 0/64, target_before 13/64. Dynamic latest Target Host rank by transition is `[6,6,6,6,7,7,4,4]`, with target_before query true in 6/8. Host marker precedes query, so true proves completion only by query end, not before Host marker submission. Latest Host rank is not necessarily device critical rank. No exposed idle, numeric Bound or TPS gain follows.

Reviewer independently checked Run660 root cause as inference tensor `_version` access, Run661 client 48/48 both phases, source/script byte equality after cleanup, service stopped and eight NPUs idle.
