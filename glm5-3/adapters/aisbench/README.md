# Task-local AISBench adapter

Derived from the deployed prefix-cache driver, preserving generated data, API sampling, prefix warmup and formal workload. `prefix_bench.py` replaces the shell pipeline with `runtime/phase_runner.py`: captures the real child exit and stops immediately on failure before parsing metrics. Candidate source originated in execution-fixture-20261001T083737Z; that directory did not contain test evidence at recovery.

Run with `PYTHONPATH=<repo>/glm5-3/runtime:<this directory>:<aisbench work path>`. The deployed wrapper accepts `AISBENCH_TEST_PY`, so the live driver can remain untouched. New Runs must pin adapter/runtime/deployed-wrapper identities and record phase exit artifacts. A successful CLI is not proof all requests passed; validate details and SLO separately.
