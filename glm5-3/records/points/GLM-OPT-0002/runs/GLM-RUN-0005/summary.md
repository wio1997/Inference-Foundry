# GLM-RUN-0005 — Native capability E2E

7 new successful inference requests, 1296 output tokens; native400 excluded from inference, one8192-budget request intentionally cancelled. Reused Run4 local P/D128+128. Stream/nonstream, chat/completion, legal n2 sampling, variable open arrivals, active drain and generation readd passed. Cancel lease released then native P running1→0 within observed1s; both engines waiting/running0, temporary gateway8002 stopped.

P-only arrival during D drain:32 outputs, first output0.654s/TPOT239.775ms; D drain512 first output0.827s/TPOT21.879ms. Different configs/workloads, not a code-speedup comparison. Shows P complete decode needs a different deployment from eager/MTP1.

INCONCLUSIVE performance; functional diagnostic passed. No formal KEEP or stable capacity claim.
