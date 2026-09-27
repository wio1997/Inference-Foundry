# Run446 — archived source recheck

2026-09-27. **ACCEPT unchanged.** Stdlib-only checks on temporary copies; no service/NPU actions.

The decompressed installed_allgather_rootinfo_test.cc.gz is byte-identical to the original raw source and has the previously reviewed SHA256 8a5339c7310160c8028dc4d481b9a44a9326979124c8ccf7b5b31732b9015d15. The generator hashes decompressed bytes directly, preserving vendor whitespace.

Regeneration reproduces updated Run445 byte-for-byte. All 19 certifications remain false, all registered numeric endpoints are null, and Current remains 571.681 tok/s.

Negative controls reject modified decompressed source, corrupt gzip, each of the six registered numeric endpoint keys under proof_dag, an added rank-failure line, zero duration and inconsistent bandwidth. No guard regression was found. Original Run446 scope limitations continue to apply.

Updated SHA256:

- Generator: 84edd4f96d8610af47b9cfe9fa5e06a3b649eb1fae03a5bd195c627505c0c7db
- Run445 JSON: d95dcca019ade92769c11bb0786dd0425518a32234888f86aad30115c1f8618e
- Source gzip: 4deb1d9ffc1732bf36b9e04702a618d022cc692ca7aa80e2a16f5e9e3920a973
