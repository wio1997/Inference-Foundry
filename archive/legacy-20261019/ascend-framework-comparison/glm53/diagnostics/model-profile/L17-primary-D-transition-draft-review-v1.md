Primary review while the local D lifecycle draft is being prepared; no transition is approved yet.

1. Meta is a Docker container on node34 only. D36/D41 stop/precheck plans cannot run local `docker inspect META_ID`: that container does not exist on those Docker daemons. Qualify Meta ownership/liveness through the controller's source-pinned node34 meta-before/meta-after plans. D stop records may retain the expected controller Meta ID as a static reference, but must not fabricate a successful local Meta check.

2. Model stdout/stderr normally append shutdown messages during docker stop. Preserve the complete pre-stop byte prefix and the complete final files; do not assert full before/after hashes are identical. After stop, require each final size >= pre-stop size and hash exactly the first pre-stop byte count to prove that the old bytes were retained; report final full-file size/SHA separately. Profile metadata should remain exactly identical because profiling has ended. Do not truncate, rotate, overwrite or delete old files.

3. Resource closure requires supervisor provider_exit_code==0 as well as provider CLOSE_COMPLETE rc0, stopped owned container/Pid0 and all8 NPU cards empty. Clean-model exit/timeout diagnostic remains separate. Retain raw SDK copy return codes and model shutdown additions. All futures must have their raw result saved even when another node fails; no automatic retry or broad cleanup.

4. New mixed capacity/identity selectors must keep P native candidate fixed independently from candidate/rollback D selection; see L17-mixed-identity-and-capacity-review-v1.md. Actual fresh D IDs remain pending the native READY+mount audit. No model-start or rollback execution is approved by drafting these plans.
