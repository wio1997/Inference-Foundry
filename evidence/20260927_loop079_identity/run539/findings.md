# Run539 — post-cleanup admission CPU gate

2026-09-27. `loop079_formal_ledger_final_admit.py` SHA256 `ea093143aa603e79543d23da57cb9f21db9acfac1f34294756147d249eefe7f7`; selftest SHA256 `09757ccc70f858c0d33bec66add1581a5c99d07b30681e87d6a247dce3d28eea`.

Host and target-container CPU tests each exited 0 with two positive cases, including independent post-cleanup replay, and eight negative cases. The gate requires successful run, stop, idle verification, restore, original source SHA, all diagnostic script SHA, 96 client/server request admission, warmup barrier and exact POST count. It rejects helper drift and mismatched install/restore manifests. This test does not launch service or certify a Bound.
