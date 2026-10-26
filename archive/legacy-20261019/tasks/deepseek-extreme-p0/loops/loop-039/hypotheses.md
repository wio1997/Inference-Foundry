# 假设记录

## H001 - ACTIVE

The target graph contains 86 grouped-matmul kernels per c12 cycle summing about 9.97 ms in synchronized traces; a lower-cost backend or better fusion for the W4A8 MoE grouped-matmul path can remove at least 5 ms from the 46.56 ms unprofiled target stage without changing model state or outputs.
