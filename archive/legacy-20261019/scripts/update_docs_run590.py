from __future__ import annotations

import json
from pathlib import Path

ROOT=Path('/data/wio/Inference_Foundry')
summary=json.loads((ROOT/'evidence/20260928_loop081_bound/run590/summary.json').read_text())
assert summary['status']=='scoped_read_only_static_inventory_admitted'

notes={
'ACHIEVABLE_BOUND.md':'''## Run590 HCCS static inventory and timing boundary

After Run589's clean stop, a single read-only supported `npu-smi info -t hccs` call per rank on installed driver26.0.rc1 passed all8 exits and an offline reducer with five rejected negatives. Each rank reports seven local links with lane mode4, standard speed224 Gb/s and TX/RX bytes equal to20×reported packet counts. CLI/DSMI/DCMI and device-image hashes are frozen; the CLI directly imports DSMI, but the exact statistics dispatch and actual booted device implementation are not proven. This narrows static topology and measurement-method uncertainty, **not** physical workload traffic or a maximum cut rate. The installed device-side source has a cache refresh path with no exported sample timestamp; freshness remains unbounded for a short Graph. Do not turn standard speed, 20-byte units, counter values or the 500 ms source timer into strict `C⁺/B`. Compulsory fixed-`W₀` work/traffic and all8 legal critical path remain open. Formal Current571.681 tok/s and all strict finite Resource/Hardware, Scheduling/Execution and Product endpoints plus numeric Current→limit gap remain null. See Run590 findings/summary and PK-088.
''',
'PERFORMANCE_MAP.md':'''## Run590 HCCS inventory scope

The supported current CLI yields all8 seven-link static HCCS records with lane mode4 and standard224 Gb/s, but no remote peer/cut map, maximum serializer rate, loaded counter implementation or sample freshness. Run590 therefore informs future physical communication measurement design without assigning necessary bytes, overlap or a Resource ceiling. The largest unresolved Bound pieces remain fixed-`W₀` compulsory work/traffic plus matching exact-board cumulative capacity, and a timed all8 production dependency DAG. Current571.681 tok/s; strict numerical endpoints remain null. See PK-088.
''',
'PROJECT_STATE.md':'''## Run590 Resource measurement checkpoint

All8 supported HCCS static inventory and installed package identity PASS after Run589 stop; all8 rank commands exit0 and offline reducer rejects five negatives. No short-window traffic or maximum link rate is certified because statistics dispatch, loaded device image, counter freshness and physical cut remain open. Continue exact-board Resource certificates and Run589-labeled production DAG time/consumer acquisition. Formal Current571.681 tok/s; strict bounds and numerical gap remain unknown.
''',
'RESULTS.md':'''## Run590 HCCS read-only inventory

Eight supported current-driver CLI calls exit0: seven local links/rank, lane mode4, standard224 Gb/s; reported TX/RX bytes=20×packet counts. Package SHA and reducer with five negatives pass. This is static inventory only: no physical workload traffic, HCCL capacity upper rate or finite Bound. Formal Current571.681 tok/s. See Run590 summary/findings.
'''}
for name,body in notes.items():
    p=ROOT/name; s=p.read_text(); assert body.splitlines()[0] not in s
    p.write_text(s.rstrip()+'\n\n'+body)

entry={'id':'PK-088',
 'topic':'HCCS static link inventory is measurable on installed 26.0.rc1, but cached statistics need freshness proof before traffic attribution',
 'mechanism':'Supported npu-smi HCCS CLI reports seven local links/rank and packet-derived byte counts; installed device-side statistics source uses a cached refresh timer and has no exported sample timestamp, while exact loaded implementation and dispatch are not yet bound.',
 'environment':'Wuzhou S900K3 8×Ascend910B3, driver26.0.rc1/CANN9.1.0; Run590 one read-only static inventory per rank after Run589 stopped; fixed DSpark7 Extreme.',
 'observed':'All8 CLI exit0; seven link records/rank, lane mode4, standard224Gb/s and byte-count=20×packet-count relation. CLI/DSMI/DCMI/device images SHA-pinned, all acquisition inputs unchanged. Offline reducer exit0 and five negatives rejected.',
 'failure_or_limit':'No loaded device-side object or exact statistics subcommand proof, sample generation/timestamp, maximum staleness, remote port-peer/cut map, maximum serializer tolerance or workload traffic. The 500ms source timer is not a strict refresh bound; a short-window count delta and nominal speed cannot certify C+/B.',
 'revalidate_when':'Bind CLI/DSMI dispatch and actual booted device image to supported counter freshness semantics or obtain device-timestamped link profiler; separately acquire max-rate/topology certificate and fixed-W0 mandatory cross-cut information before a Resource time floor.',
 'extreme_relation':'Narrows exact-board static topology and rules out unsupported short-Graph counter attribution; all strict finite Resource/Scheduling/Product endpoints and numeric Current-to-limit gap remain null, Formal Current571.681tok/s.',
 'source':[{'repository':'Inference_Foundry','ref':'run590','path':'evidence/20260928_loop081_bound/run590/summary.json'},
           {'repository':'Inference_Foundry','ref':'run590','path':'evidence/20260928_loop081_bound/run590/findings.md'},
           {'repository':'Inference_Foundry','ref':'astra_capacity_path_review_589','path':'evidence/20260928_loop081_bound/astra_capacity_path_review_589/review.md'}],
 'status':'scoped_observation'}
p=ROOT/'performance_knowledge/entries.jsonl'; lines=p.read_text().splitlines()
assert entry['id'] not in {json.loads(x)['id'] for x in lines if x.strip()}
with p.open('a') as f:f.write(json.dumps(entry,ensure_ascii=False,separators=(',',':'))+'\n')
print('Run590 docs and PK appended')
