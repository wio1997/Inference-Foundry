# Input token cache candidate

Uninstalled design for repeated text requests. Cache only immutable input token IDs, keyed by the fully rendered text and every actual scalar tokenizer encoding option. Never cache model outputs, EngineInput, SamplingParams or mutable request fields. Build fresh prompt lists and retain all input validation. Bound entries and estimated retained Python-object bytes. Fixed tokenizer instance/configuration is required; replace cache if it changes. Offset paths must bypass integration entirely.

Run670 is isolated worker1/4 formal E2E. This file is not imported by the active service. Candidate admission requires actual patched tokenizer parity, mutation/option/eviction/concurrency checks, renderer integration review, then frozen formal E2E. No performance claim.
