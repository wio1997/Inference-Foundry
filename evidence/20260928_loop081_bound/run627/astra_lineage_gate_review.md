# Run627 independent Astra High lineage-gate review

Reviewed generator SHA256 `c2e098b4ac9f44f76d4c267e37d38f62e4c741f931826dc30726813d852c7dad`; output `09fa2f7e40ef12678465e43b2766a62a6870afd8f196bcaafebc3873d93f5ec0`.

**Verdict: corrections required before calling this fail-closed schema preflight.** Pure conditional logic only; no live packet, service or device experiment. All18 supplied fixtures and saved JSON were independently reproduced exactly using only the read/compute portion. Existing endpoint limits are correctly null.

## Reproduced false PASS cases

Starting from BASE, the following independently tested mutations all return PASS_CURRENT_ELIGIBLE_LINEAGE:

| Mutation | Problem |
|---|---|
| Set query writer native_seq=10, same as selected context writer, but keep a different destination | Native-order collision is checked only among selected-row hits; two different calls share one same-stream canonical ordering token. |
| Append a second selected call_id at seq40, after reader seq30 | Duplicate selected-call identity is filtered away before uniqueness check. Call identities must be unique over the whole packet, not only before the reader. |
| snapshot_post_context_equals_pre_reader=0 | Only singleton False is recognized; malformed false-like input silently passes. |
| snapshot_post_context_equals_pre_reader="False" | Same type-validation hole. |
| Nonselected writer cache_identity=None | Malformed writer identity is treated as a different allocation and skipped. |

**Required fix:** separate complete packet schema validation from semantic filtering. Validate every writer/reader identity and type first; enforce globally unique call IDs and unique canonical native order keys within the declared acquisition/process/device/stream scope (including records after the reader and off-selected-row writes). A canonical order key is not an unqualified raw task_id that can wrap or repeat across Model/process epochs. Reject malformed optional snapshot fields; missing/None may be an explicitly supported “not collected” state, because authenticated write-order proof need not require equal byte snapshots. Do not let early UNKNOWN/NO_WITNESS paths implicitly admit a malformed packet.

## Actual slot ABI mismatch

The synthetic base accepts [-1,-1] as its sole invalid pair. Actual non-A5 `format_dsa_slot_mapping` is stack(flat//32, flat%32), so flat-1 gives **[-1,31]**. That pair currently raises ValueError. This is a conservative rejection, not false lineage PASS, but it prevents describing the helper as accepting the actual formatted ABI.

Do not silently normalize arbitrary negative pairs or treat index rejection as absence of a write. Either pin/admit the precise installed scatter invalid-index rule and record raw plus normalized representations, or return explicit unsupported/UNKNOWN for raw negative forms until that adapter is established. The installed scatter source computes linear indices and filters against per-core index ranges, but this review does not certify the loaded binary/tiling path. The fixed block32 rule must be explicit in the admitted geometry/schema; a caller-provided capacity alone does not specify block size.

## Alias, eligibility and generation certificate boundary

The helper trusts five Boolean certificate flags; it does not call the Run619 oracle or bind actual metadata to its output. Pinning the Run619 synthetic fixture JSON is not an actual-reader certificate. This is acceptable only under the output's explicitly conditional pure-helper scope.

For later real admission, cache_identity must mean a canonical allocation-lifetime and byte-coordinate identity, not a layer/prefix/tensor-owner label. Different views of the same allocation must be normalized to the selected byte row, and all overlapping alias writers covered. Differing labels must not cause a real alias writer to disappear. Need a same-acquisition join across actual descriptor/view, lifetime generation, writer indices and row version, reader metadata generation and oracle output. The reader eligibility list must be the admitted branch-specific set: dense window/table or sparse prefix/terminator, with execution coverage separately established. These are not authenticated by True flags or nonempty strings.

Duplicate destinations within any admitted writer are conservatively UNKNOWN; that is safe even if later a specific loaded scatter variant proves deterministic winner semantics. Equal post-context/pre-reader bytes cannot identify the last writer when an overwrite writes identical bytes. Conversely a current lineage can pass without Resource “freshness”; reuse does not invalidate a well-defined current write/read relationship.

## Correct meaning / final limits

After structural fixes, PASS may mean: **under independently supplied authenticated certificate premises, this selected context invocation is the last writer to this canonical row before this eligible reader in the admitted stream order.** It is not physical-first-read, actual value contribution, compulsory HBM transfer, fresh unavoidable arithmetic, a mandatory separate scatter, unavoidable serial runtime architecture, or critical-path duration. NO_CONTEXT_WITNESS is selected-row-local, not a claim that no dependency exists globally.

All strict Resource/Scheduling/Product bounds and numerical Current-to-Bound gap remain null; Formal571.681 tok/s unchanged. No live-readiness promotion follows from additional synthetic fixtures.

## Corrected revision — final scoped verdict

**SCOPED PASS for the pure CPU conditional-lineage helper; NOT LIVE-READY.** This supersedes the initial schema rejection above.

Final generator SHA256: `be45423ca4c979784c3d1263421865e77d2070dc93e232b0f2e1aed9c85ea688`.
Final JSON SHA256: `c363d6ee96a9bb3f4bcd68fbd6b536eb80907b0e9fb13dfcbe57a37124b3010f`.

Independently reproduced all25 fixtures and the entire JSON byte-for-byte, executing only source read/compute portions. Re-ran all five originally reported adversarial mutations; each now raises schema rejection: disjoint-row native ordinal collision, selected call repeated after reader, snapshot0, snapshot string "False", and another writer's cache_identity=None.

Additional independent checks:

- Malformed writer identity still rejects when native_flow_verified=False: no evidence-return bypass of schema validation.
- Actual formatter [-1,31] returns UNKNOWN; no unsupported sentinel is silently discarded into a PASS.
- A later same-cache selected-row writer gives NO_CONTEXT_WITNESS even when byte-snapshot parity is True; equal values cannot hide a write generation.
- Reversing the writer-list presentation leaves the result unchanged; ordering uses authenticated native sequence, not input list order.
- A newly added later same-cache alias-normalized writer to the selected row also gives NO_CONTEXT_WITNESS.

The full schema pass precedes evidence classification, checks every writer cache token and call ID, and enforces (stream,sequence) uniqueness against the reader and all writers. Snapshot type is now strict; absent/None or False is conservatively UNKNOWN. Requiring True parity is stricter than necessary for an abstract authenticated order proof but cannot create a false PASS.

No remaining false PASS was found **within the stated abstract premises**. Important limits still apply: block size32 and canonical physical-row coordinates are the supported model; acquisition/process/device namespace, actual same-stream execution order, alias normalization/lifetime, completeness of overlapping writers, and the actual metadata-generation→oracle eligibility/coverage join must be supplied by a separate authenticated adapter. A nonempty different cache label cannot by itself certify disjoint storage. This helper still does not enforce or construct that adapter, and it must not be wired directly to raw packets with self-declared True flags.

The single-reader format also does not certify first reader or exclude earlier consumers. PASS certifies only the named eligible reader's selected current last writer under the external premises. Physical reads, semantic contribution, Resource freshness, compulsory traffic, mandatory physical operations, cross-stream reorder legality and Scheduling duration remain unproved. All numerical endpoints remain null; Formal Current571.681 tok/s unchanged. No further toy expansion is required for this scoped helper before progressing to concrete adapter/source evidence.
