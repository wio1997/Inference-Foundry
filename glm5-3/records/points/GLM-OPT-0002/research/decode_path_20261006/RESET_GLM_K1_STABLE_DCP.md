# Scoped MTP stable DCP table — Research Reset

2026-10-07. One active hypothesis: remove native GLM K1 draft eager host submission through a supported single ACL graph. H6/H5 remain the validated research baseline; target FULL/bucket2 and all parallel/async settings stay fixed. Formal Current/API/80K input/600 output/93% shared-prefix and observed cache contract remain open.

Run272 actually captured the graph and returned exact short outputs, but all16×2 real mode1 rows rejected replay. The only signature difference is DCPContext.block_table.data_ptr; both runtime steps also differ from each other. Its complete request was not executed. No timing or graph correctness claim follows. H6/H5, mode0, all16 healthy/idle were retained; no recovery reload was necessary. Original raw/spec stay immutable.

Source cause: FULL `_adjust_tensor` pads the input table into temporary storage, then the SFA-DCP builder retains a local view. The existing persistent clone is bound only after building metadata. Dummy step0 captures the original runner table. The decode SFA consumer reads the retained local table. Reuse Run272 raw and actual builder/consumer source; no new profiling or parameter scan is needed.

Minimal patch v5 stages the current table into the existing persistent clone before both dummy step0 and runtime metadata builders. Preserve logical request rows and column stride, copy current values, skip exact self-copy, leave other models/K/native eager behavior intact, keep every pointer guard and owned output. Exact NoopOffloader scope and single-wrapper correction remain. No kernel, arithmetic, communication or fence removal.

Actual container CPU Torch and SFA local-view AST verify five independently allocated padded inputs update the same captured view, plus unpadded/alias/shape rejection. Actual context/observer CPU tests pass. Independent review references `MTP_DCP_BLOCK_TABLE_REVIEW_272.md`; final diff review and device evidence are separate requirements.

Distinguishing diagnostic Run273: one owned D replacement, normal four fixed small standard-PD requests (mode0 short, mode1 short, mode1 complete natural23 EOS, retained-mode warm). At most one owned failure recovery plus one recovery warm. P stays protected. Unique per-request salt, exact tokens/content/EOS, scheduler counters, all16 transfer/capability/changed positions and lengths, same captured pointers, actual replay branch plus completed output are required. No profile, scan or large workload.

Decision table: all-rank real replay + exact short + exact complete EOS + terminal health passes → independently reduce correctness, then prepare reviewed same-worker matched A/B/A/B. Pointer rejection, numerical mismatch or service failure → preserve first raw; no performance claim or automatic repeat; attribute source before any new diagnostic. H6/H5 stay active. Only this storage contract evidence is currently missing; capture alone is insufficient.
