# Run663 — invalid RoPE verifier, no candidate correctness verdict

Zcode ran the guarded Run663 controller. The first warmup cohort raised at cycle0 on all eight ranks with rotary target_cos/sin mismatch; controller exited1, no measured phase. Stop, idle and exact source/script restoration passed.

**Correction:** the diagnostic compared entire active RoPE buffers against make_scratch buffers. The active buffers are allocated with max_num_batched_tokens rows in vllm_ascend/ops/rope_dsv4.py, while make_scratch truncates target_cos/sin to 96 rows. torch.equal returns false for different shapes without testing values. Thus the all8 rotary mismatch is a **verifier shape false positive**; no correctness verdict about Graph replay follows. Run664 inherited the same verifier bug and is invalid for the same reason. Compare only the first 96 written rows, assert shapes and inspect actual differing elements in a fresh Run.

Service stopped, eight NPUs idle and borrowed source SHA restored. Do not use either Run663 or Run664 for performance or candidate correctness decisions.
