# Run408 review corrections — schema 2 and proposed cleanup

No borrowed source was patched. No service or NPU action was taken. The active
A0 runner was not edited.

Helper now separates first_clone (source/output pointers, clone_pre/post Host
timestamps and lineage) from the first actual numeric consumer (downstream_pre/
post, input pointer and direct_mirror/tracked_clone lineage). The existing
build_parallel_draft_seq_lens_cpu clone is reference/data propagation only;
its following existing out[:num_reqs].add_(query_len) is the first arithmetic
consumer if that branch runs. Otherwise an actual DSA max/item site can record
numeric consumption. Clone no longer suppresses those numeric markers.
Only .add_/.max_item sites with verified pointer lineage yield downstream_observed.
Schema 2 prevents old clone-only schema 1 records being silently accepted.

CPU tests explicitly prove clone-only -> downstream false, clone -> numeric
still records, and mislabeled clone-as-numeric is rejected. Original wait/copy
counts and the compound overwrite expression remain unchanged. New validation
is in validation_revision2.json; source_check.json reflects the new generated
patch. Any source_check_sol.json used by a runner must be regenerated/reviewed
before B install: the old generated SHA should fail closed.

## Cleanup proposal (not applied)

scripts/run_loop078_count_marker_cleanup.patch is a unified diff against the
active scripts/run_loop078_count_marker_arm.sh. A0 may be reading that shell
file; do not apply until its process has exited. Inspect current runner SHA
against cleanup_patch_review.json, then apply only after A0 completion:

    git apply --check scripts/run_loop078_count_marker_cleanup.patch
    git apply scripts/run_loop078_count_marker_cleanup.patch
    bash -n scripts/run_loop078_count_marker_arm.sh

The existing cleanup swallows stop and restore failures. In addition,
99_stop_service.sh itself masks its docker stop errors and does not enforce
its memory-release condition. The proposal therefore records stop exit and
independently requires no VLLM/vllm processes plus exactly eight parsed devices
below 6144 MB with no VLLM rows before allowing restore. Failure to establish
this state preserves installed sources/backups and exits nonzero. This strict
parser can conservatively reject unfamiliar npu-smi output; it must never turn
unrecognized output into permission to restore live source.

cleanup_status.txt exposes original run exit, stop exit, stop-verification exit,
restore exit, SHA command exit, before/after SHA comparison and final exit.
source_after.sha256 is recorded even if restoration is refused. Existing run
failure codes are preserved; otherwise cleanup failure changes final exit to1.

Proposed shell passes bash -n. Five CPU stubbed cleanup cases pass: stop failure
never restores; verification failure never restores; restore failure propagates;
success exits0; original run error remains visible. No mocked test called actual
stop_service, docker, or npu-smi. See cleanup_selftest.json. Active runner SHA
remained unchanged throughout proposal/tests.

Cleanup proposal follow-up: /proc inspection now reads status/State and skips
Z (zombie) entries before classifying VLLM/vllm processes as live. Unknown process
state fails closed. The separate npu-smi process/HBM gate is unchanged; zombie
filtering in /proc never suppresses evidence of actual device occupancy. Five
CPU fixture cases pass (zombie-only, live worker mixed with zombie, live vllm,
unrelated process, missing state). See cleanup_zombie_selftest.json. Patch remains
unapplied and the active runner SHA remains unchanged.
