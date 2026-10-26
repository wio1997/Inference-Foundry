# Run280 — CP c4 immediate, one layer, invalid capture gate

The SHA-guarded patch was installed from base 27cbbf92a02f16301aad2a35091e258b8459dc77744bd8294f2f11a4c126004e to candidate 665349d9e25fb0cd47fadd4777b340eddbf65a002e93214ed95f32020dc585ce. All eight worker logs selected `model.layers.2.self_attn.attn` in `immediate` mode during Graph capture. Capture then stopped with `CP fork KV storage alias: [['main.state', 'main.kv']]`.

This audit result is an invalid cross-branch safety gate: `audit_disjoint` compares members of its private set with each other; the two reported tensors are both in the main branch. No main-versus-indexer conflict was reported before the exception. No service health, cohort, timing, or numerical correctness result exists from Run280. The service was stopped and borrowed CP source restored to base SHA. The replacement audit in patch v2 compares each main tensor separately against indexer tensors, preserving conservative cross-branch interval checks.

The launcher was interrupted once the capture failure was established; its remote host process was then terminated so its cleanup trap could stop the service and restore source. The TaskCtl Run280 verdict is invalid, correctness invalid.
