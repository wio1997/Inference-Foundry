#!/usr/bin/env python3
"""Append the reviewed Loop079 bound update once to the project maps."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRIES = {
    "ACHIEVABLE_BOUND.md": """
## Loop079 Run421–425 terminal output ledger

Clean exact60 diagnostic Run421 joined all60 terminal Scheduler bulk appends to all8 FULL Graph Runtime request IDs. Scheduler had already generated 584 tokens (1–33/request) at the terminal bulk call; Runtime submitted61,440, Scheduler admitted60,856 and clipped584. Handoff-time generated `g_i^H` and API-published `p_i^H` are not measured, so neither the formal nor historical trajectory inherits these counts. Run423 independently established that a strict loose TPS ceiling needs only a proved positive necessary work subset and matching genuine capacity upper cap; the inspected BF16 `wo_a` subset and official white-paper options do not yet complete that pair on this SKU. Run424 specifies the missing all43 row/Graph/native binding. V3.12 keeps all finite Bound endpoints null. Formal Current remains571.681tok/s. See `evidence/20260927_loop079_identity/run425/findings.md` and `run425/bound_calibration_v3_12.json`.
""",
    "PERFORMANCE_MAP.md": """
## Loop079 Run421–425 terminal output ledger

The frozen diagnostic has 584 Scheduler-generated tokens before terminal bulk processing, so its61,440 Runtime-retained IDs are not all externally newly admitted:60,856 were admitted and584 length-clipped. The handoff-time already-generated and API-published counts remain distinct unknowns. Resource ceiling requires a matched necessary W-minus / maximum C-plus certificate, not a measured kernel peak; Run423 found the smallest candidate but no certified numeric pair. Scheduling saving and full router identity remain unproved. Next: request-correlated handoff+publication ledger, row/Graph/native certificate, and 910B3 SKU capacity binding. Evidence Run421–425/PK-042.
""",
    "PROJECT_STATE.md": """
## Loop079 Bound checkpoint (Run421–425)

Run421 clean exact60 Host diagnostic establishes terminal Scheduler prior-generated total584, Runtime incoming61,440, admitted60,856 and clipped584. Handoff-time generated/published counts remain unknown; no formal TPS or finite Bound promotion. Independent Run422/423/424 reviews and V3.12 model are saved under `evidence/20260927_loop079_identity`. Continue handoff/API ledger and full43 row identity while separately binding a minimal necessary-work subset to an actual910B3 maximum capacity certificate.
""",
    "RESULTS.md": """
## Loop079 Run421 diagnostic (not formal)

Exact60 requests,48+12,c12,all8 FULL Graph,source restored/NPUs idle. Terminal Scheduler ledger:584 previously generated tokens,61,440 Runtime bulk IDs,60,856 admitted,584 clipped. No handoff-time or API-published count and no performance-bound endpoint follow. Formal Run99 median remains571.681tok/s. Details: `evidence/20260927_loop079_identity/run425/findings.md`.
""",
    "HANDOFF.md": """
## Loop079 Run421–425 terminal output ledger

Run421 clean diagnostic:60/60 terminal Scheduler `g_before_bulk>0`, sum584; Runtime incoming61,440, admitted60,856,clipped584. Handoff generated `g_i^H` and API-published `p_i^H` are unknown; do not transfer counts to Run403/239/99. Source restored, service stopped, NPU idle; post-stop container retained212 processes/20.55GiB and was reset to one process/1.238MiB. V3.12 has no new finite endpoint. Run422 Astra review, Run423 minimal strict W/C gate, Run424 all43 row-identity design, Run425 findings/model. Next clean exact60 request-correlated handoff+OutputProcessor/API ledger; preserve independent SKU capacity and router identity tracks.
""",
}


def main():
    for name, block in ENTRIES.items():
        path = ROOT / name
        old = path.read_text()
        if block.strip().splitlines()[0] in old:
            raise RuntimeError(f"already present: {name}")
        with path.open("a") as f:
            f.write("\n" + block.strip() + "\n")
        print(name)


if __name__ == "__main__":
    main()
