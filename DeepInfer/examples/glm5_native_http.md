# GLM-5.3-Flash native HTTP transport

`glm5_full_model_offline.py --serve` exposes `/health`, `/v1/models` and streaming
`/v1/completions`. The existing external launcher still owns device mapping,
rendezvous, claims, deadlines and cleanup. Replace `--cases` or `--tokens` with:

```
--serve --host 127.0.0.1 --port 18080 --served-model-name glm53flash
```

Use the already supported model/rank/world-size/local-device/init-method/output-dir/
go-file arguments. For fully resident TP8 set `--world-size 8 --weight-offload none`.
`--serve-max-requests 3` bounds an integration run; zero leaves the service running
until its launcher's deadline. All ranks receive the same configuration; only rank
zero starts HTTP, while other ranks execute the native Worker collective loop.

The transport supports a string or single token-ID vector as `prompt`, greedy
`temperature=0`, bounded `max_tokens`, `ignore_eos`, and streaming usage. It rejects
unsupported sampling and endpoints. HTTP threads only queue and stream; the original
engine thread performs model operations. A disconnected client does not interrupt
collectives: the active request drains before its state is released. A model error
after state mutation is fatal to the bounded process rather than silently reused.

**Current limitation:** one active model request, with a bounded pending HTTP queue.
This is not continuous batching. Do not present concurrent queued HTTP requests as
batched device execution. Context, output and device limits are launcher inputs.

An eight-rank 910C integration run completed three real HTTP requests, including
64 forced output tokens, with native worker exit codes all zero and resource
release. That integration result is not an AISBench performance claim. Formal
engine comparison, batching work and the 1.5–2x performance target remain pending.
