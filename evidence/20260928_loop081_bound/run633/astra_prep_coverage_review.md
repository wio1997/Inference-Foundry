# Run633 independent Astra High preparation Host-call coverage review

**SCOPED PASS for the32 rank/cohort Host interval ledgers in Run606's own diagnostic W0.** This is a preparation accounting refinement; it does not establish device service, a removable gap, a critical path or a numerical Bound. No service/NPU workload was run, and only this review report was written.

Final generator SHA256: `55fd7b78c6ae299970e69d52a556613e9d3c8768e812fd4589bb4bcf92bbdf6c`.

Final `prep_host_coverage.json` SHA256: `1bac88d687cbd16b9a92123d07f7af2140c845ab356c0eb8954183e1ea0feea0`.

## Independent reproduction and ownership

I ran the final main with writes intercepted in memory and reproduced the saved JSON byte-for-byte. I separately reconstructed all32 interval unions directly from raw records with a conventional sorted-interval merge, without calling the reducer's sweep. Every union, complement and total span matches exactly in integer ns. The class-signature durations sum to each rank's first-own→built interval. Raw product files and client admission are rehashed against the Product admission manifest; all pass. Rank/cohort/run tag/namespace match the expected identity.

The collection patch stores `execute_mark_count = len(self._p602_marks)`, which counts **all marks**, not a filtered execute-only list. Source `scripts/loop081_product_patch_run602.py` SHA256 is `a80aa76e2396a0da4f5fbb881d64dc84a0bcc177e08facda417c7e88dcce25e7`; assignments occur in both proposal and Target call collection. Initial Run633 indexed the filtered list, relying on an implicit property. In the admitted raw data all marks before every call are execute_entry, so its original numerical output happened to be correct. The final reducer explicitly enforces this property and indexes the raw all-marks list.

I independently checked all **640 call rows**: each lies between its identified execute marker and the next marker, and every proposal's request-ID multiset matches that execute's new+cached operands. The Target wrapper has no independent request-ID payload, so its ownership is inherited through this source-pinned execute occurrence and temporal containment; the report must not upgrade it to a separate token/activation lineage certificate.

Each rank has40 Target/proposal pairs across the four files. **37 pairs belong to the current file's cohort**, with9/11/7/10 pairs for cohorts5–8. The other3 pairs belong to the predecessor cohort, one at the beginning of each later file. The three no-request execute markers per rank have no Target/proposal calls. Thus the prior40-pair count remains a valid current issuance count across these files but is not40 own-cohort preparation pairs. Final handoff execute entries are included in marker counts and need not have these ordinary calls.

## Negative challenges and corrections

The first revision accepted four in-memory ownership corruptions: count1→3 reassignment of a prior call to own, invalid proposal request IDs, changed namespace, and changed packet rank with matching marker ranks. Sol added exact expected rank/namespace, all-marks indexing, call containment and proposal Counter equality.

Eight final independent negative cases reject: invalid owner index reassignment; alien proposal IDs; duplicate proposal ID; wrong namespace; changed packet/marker rank; a non-execute marker before a call; call end crossing the next marker; and a raw SHA mismatch. Structural mutations were applied after parsing in memory, separately from hash drift, and did not modify evidence. These test the named cases, not arbitrary future schemas.

## Host coverage interpretation

The selected window starts at **that rank's** first execute with own-cohort operands and ends at **that rank's** Host runtime_built marker. It is not the allrank maximum-built envelope from Run631; small differences from that ledger are expected. Prior-cohort calls are preserved in whole-file call counts, but their intervals are outside this selected own window in the current data. Mixed-owner classification remains possible in the reducer; none occurs in the admitted records.

| Cohort | Own-first→built range across ranks s | Target/proposal Host union range s | Uncovered Host range s |
|---:|---:|---:|---:|
| 5 | 3.397282–3.407244 | 2.916275–3.129531 | 0.277714–0.484006 |
| 6 | 3.296470–3.307725 | 2.759560–2.998766 | 0.306202–0.538919 |
| 7 | 2.450091–2.461317 | 2.081848–2.238069 | 0.221597–0.368645 |
| 8 | 3.121760–3.133296 | 2.567882–2.812521 | 0.320775–0.556505 |

These ranges are descriptive; extrema can belong to different ranks, so table endpoints must not be arithmetically combined. Each actual row satisfies union+complement=span. Host call overlap is0 in all32 rows. That describes the Python wrapper intervals and does **not** prove no device compute/communication overlap. A Target Host return can precede Graph/native completion, a proposal wrapper can contain waiting or other work, and asynchronous device work can run outside the wrappers.

Uncovered0.22–0.56s per rank/cohort is therefore unclassified Host-envelope time, not Host overhead, idle time, required time or recoverable latency. Built is Host construction, not full seed/KV/state readiness. Current-stream event elapsed fields are correctly omitted from service accounting because they are not absolute side-stream completion or disjoint phase costs. The timestamps used here are Host monotonic marks in the admitted namespace, not device timestamps.

Run606's observer remains perturbed. No durations transfer to Run99, no ranks or cohorts are summed into a Product critical path, and no comparison with another W0 becomes an intervention result. All Resource/Scheduling/Product endpoints and the numeric Current-to-limit gap remain null. Formal Current571.681tok/s is unchanged. No remaining correction is required for this narrow current Host-coverage scope.
