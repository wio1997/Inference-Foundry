# Four-op workspace owner, g5

Baseline is the frozen `candidate_extension_g4/`. The only compiled change is `include/op_api_common_213.h`, in `EXEC_NPU_CMD`. Eight selected private APIs, exact-library resolution, current Torch-NPU storage descriptor access, API parameters, math, launch stream, setup and namespace remain unchanged.

Previously `auto workspace_tensor` lived only inside the allocation `if` block. Its Tensor owner was destroyed before constructing the custom-handler closure. Only `workspace_addr` survived, which is a raw pointer and owns no allocation. This is a C++ lifetime gap even if `OpCommand.Run()` executes the handler immediately; asynchronous host dispatch can widen it. Device completion and allocator reuse are separate questions.

G5 declares the Tensor outside the block, assigns its allocation there, and captures it by value in the handler. The outer Tensor keeps storage owned through `cmd.Run()`. Any owning copy of the handler also owns a Tensor reference until that handler copy is destroyed. Zero-workspace calls capture an undefined Tensor and retain the previous null pointer behavior. No synchronization, event, CPU fallback, shape, dtype, API or arithmetic change is introduced.

Current `OpCommand` queue/Run/Sync and allocator stream/graph contracts must be reviewed separately. Merely retaining a Tensor until host API return is not proof of arbitrary cross-stream safety or NPUGraph memory-pool lifetime. The adapter already obtains its ACL stream from the current NPU stream and allocates the workspace without changing streams. No new cross-stream use is introduced by this change. Actual installed header evidence is recorded in `host_contract_review.json`. `OpCommand.h:182` explicitly says Tensor lifetime must cover Run; its Run/Sync bodies are absent. `NPUCachingAllocator.h:120-123` distinguishes requested/completed free and other-stream record_stream; lines269-273 distinguish active capture from private-pool routing. The installed headers establish these interfaces and comments, but current Run/Sync/queue and allocator free/event/graph-pool implementation sources are missing; those implementation-level guarantees remain unverified.

## Target CPU build

Run only in the parent's current Torch/Torch-NPU2.13 aarch64 image, after copying this directory to the task mount:

```bash
cd /task/compat/engram/attention/candidate_extension_g5
SGLANG_V41_CANN_ROOT=/usr/local/Ascend/cann-9.2.0 MAX_JOBS=1 \
python setup.py build_ext \
  --build-lib /task/compat/four_op_g5/lib \
  --build-temp /task/compat/four_op_g5/build
```

`setup.py`, C++ registration and namespace intentionally stay byte-identical to g4. The built DSO still registers `sglang_v41_a3_g4` and works with existing `a3_donor_bindings_g4.py` by changing only `SGLANG_V41_CANDIDATE_EXTENSION` to the new g5 build artifact. Load exactly one of g4/g5 per fresh process. Do not install over the system extension or load both versions into a single process.

Use the parent's existing four-op CPU schema/owner smoke and bounded operator/graph probes on the new DSO. The source patch alone does not establish hardware correctness, performance or the cause of prior failures. The separate legacy broad-registration issue is not changed here.
