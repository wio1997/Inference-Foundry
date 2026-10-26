# Run170 invalid before execution

The shell redirected `bash scripts/run_loop046_admission_trace.sh` to `evidence/20260925_loop046_prefill/run170/driver.log` before creating that directory. Exit code 1; the runner never started, scheduler source was never patched, no service or NPU work ran. TaskCtl records the invalid attempt. Run171 reuses the prepared opt-in scheduler trace with the output directory created first.
