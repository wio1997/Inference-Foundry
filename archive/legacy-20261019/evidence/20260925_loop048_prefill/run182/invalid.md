# Run182 invalid instrumentation registration

The temporary DSA and MoE wrappers used `*args, **kwargs`. vLLM `direct_register_custom_op` calls PyTorch `infer_schema`, which rejects varargs and varkwargs. Service startup failed during import before model loading, warmup or benchmark; no phase measurements exist. The run was stopped explicitly, all three borrowed source files were restored to their original hashes, and NPU memory returned to idle levels. Run183 will use the original typed signatures and preflight custom-op imports before service startup.
