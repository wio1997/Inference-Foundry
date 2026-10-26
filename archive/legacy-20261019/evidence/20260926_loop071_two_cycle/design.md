# Loop071 candidate: real adjacent-cycle state and critical-path gate

## Motivation and prior

Run307/308 show a local private Graph owner16 saving for one c4 layer, but Run311 does not establish Product correctness or exposed wall time. Run313/314 isolate metadata behavior at one real prestate; synthetic start shifts are not a persistent Runtime trajectory. Historical PK-008 (R21/R28) warns that overlapping Compressor work can lose to resource contention; PK-009/010 delimit the current owner's local and live evidence. Search these entries again before changing scheduling or Graph ownership.

## Smallest causally useful experiment

Use a frozen real two-cycle c12 TP8 Target entry, with original AllGather, qsl, collective ordering, request slots and acceptance decisions. Obtain a complete mutable-state inventory before intervening: Target c4/c128/SWA/indexer typed current write domains on every shared backing, Draft KV and MTP state, recurrent state, page and prefix epochs, output/acceptance/parking buffers, and Python-side metadata cache keys. Check storage alias by byte interval and typed view, not only tensor pointer. Keep all ranks on the same arm sequence.

Construct separate A→A and B→B private branches from the exact same captured prestate. Restore the full inventoried mutable state **between** branches, never between the two steps within a branch. Run A→A twice first and require exact typed state and decision equality to validate restoration. B uses owner16 only in the one c4 layer; every other layer and step remains original. Compare at step 1 and step 2 after metadata generation, producer/scatter, QLI, Sparse, Target output, acceptance and state advance. Fail at the first typed divergence, including parked slots and cross-layer aliased backing domains. If an untracked write or unsupported replay shape appears, mark the fixture invalid.

`runtime/assets.py::RuntimeAssets.snapshot_pages(strict=True)` is a candidate recovery primitive, not proof of complete byte coverage: its typed `index_select` can miss adjacent scale bytes on a shared backing (for example, a 4096-byte indexer FP32 view within a 4160-byte page). Audit against Run299/301 descriptors and use backing-level byte intervals where typed coverage is incomplete. The page union must cover both steps, including Draft context writes and up to eight token advances per request.

Only after exact parity: measure paired A→A/B→B/A→A elapsed intervals on the actual FULL Graph/rank rendezvous path, from a common predecessor through the next collective and whole Target tail, with complete metadata/Host submission included. Require all-eight-rank and A/A controls; consider repeated formal 48×32K→1024 c12 E2E only if a stable exposed advantage survives. Useful tokens/cycle and serving residual remain separate terms.

An A/B Python flag does not switch an already captured FULL Graph. Use independently captured graph identities and audit actual dispatch keys for both branches. If full Target branch capture is too costly, a layer2 shadow across real cycle0→1 can narrow state semantics, but it must replay neighboring layers' writes to shared backing between layer2 calls. Incomplete closure is `unknown`, not a synthetic-shift correctness pass.

## Bound interpretation

Owner16 is an implementation allocation, not the compulsory Resource floor: the 12 requests create 16 request-rank memberships at the current query partition, and split requests may admit other halo/unique-owner schedules with different compute/communication costs. Do not extrapolate one-layer microseconds to all21 c4 layers or TPS. A failed B branch is a candidate-specific result; it cannot certify a hardware or Scheduling-aware ceiling. The main open resource calibrations remain compulsory weight/activation/KV bytes for the same route, joint FULL Graph attainable capacity, TP8 rank arrival/link costs, and acceptance/parking efficiency.
