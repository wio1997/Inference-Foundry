from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,subprocess
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/moe-prefix-histogram-live3-20261019/node33');cid='c7be13e9c4f558e7da2aa395b764cccbd5026bf1dc4f5f3cba52abf4c7dc76db';d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];plan=json.loads((root/'plan.json').read_text());assert d['State']['Running'] and d['Name']=='/'+plan['container']['name'] and d['Image']==plan['image']['id'];assert not (root/'moe-prefix-histogram-probe/enable').exists()

from typing import Dict, List, Any


def summarize_prefix_paths(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    real = records
    per_rank: Dict[int, Dict[str, Any]] = {}
    global_invalid = 0

    def col(rank: int) -> Dict[str, Any]:
        if rank not in per_rank:
            per_rank[rank] = {
                "rank": rank,
                "real_frames": 0,
                "native_graph_calls": 0,
                "candidate_graph_calls": 0,
                "candidate_graph_fallbacks": 0,
                "native_dispatch_calls": 0,
                "candidate_dispatch_calls": 0,
                "candidate_dispatch_fallbacks": 0,
                "selected_graph_tokens": [],
                "selected_dispatch_N": [],
                "invalid_rows": 0,
                "both_native_and_candidate_paths_present": False,
            }
        return per_rank[rank]

    for frame in real:
        if not isinstance(frame, dict):
            global_invalid += 1
            continue
        if frame.get("request_ids") == []:
            continue
        rank = frame.get("rank")
        if not isinstance(rank, int) or isinstance(rank, bool):
            global_invalid += 1
            continue
        if rank < 0:
            global_invalid += 1
            continue
        ids = frame.get("request_ids")
        if not isinstance(ids, list) or not ids or not all(isinstance(v, str) and v for v in ids):
            col(rank)["invalid_rows"] += 1
            continue
        phase = frame.get("phase")
        if phase not in ("execute_model", "sample_tokens"):
            col(rank)["invalid_rows"] += 1
            continue
        requested = frame.get("requested")
        if not isinstance(requested, bool):
            col(rank)["invalid_rows"] += 1
            continue
        r = col(rank)
        r["real_frames"] += 1

        rows_bad = 0

        graphs = frame.get("graphs")
        if not isinstance(graphs, list):
            rows_bad += 1
        else:
            for g in graphs:
                if not isinstance(g, dict):
                    rows_bad += 1
                    continue
                gt = g.get("tokens")
                if type(gt) is not int or gt <= 0:
                    rows_bad += 1
                    continue
                gra = g.get("requested")
                gsel = g.get("selected")
                if type(gra) is not bool or type(gsel) is not bool or gra != requested:
                    rows_bad += 1
                    continue
                nc = g.get("native_capture")
                pc = g.get("prefix_capture")
                if not (isinstance(nc, dict) and isinstance(pc, dict)):
                    rows_bad += 1
                    continue
                nc_good = (type(nc.get("calls")) is int and type(nc.get("selected_calls")) is int
                           and nc.get("calls") == 75 and nc.get("selected_calls") == 0
                           and nc.get("output_list_types") == [1]
                           and all(type(v) is int for v in nc["output_list_types"]))
                pc_good = (type(pc.get("calls")) is int and type(pc.get("selected_calls")) is int
                           and pc.get("calls") == 75 and pc.get("selected_calls") == 75
                           and pc.get("output_list_types") == [0]
                           and all(type(v) is int for v in pc["output_list_types"]))
                if not gra:
                    if not gsel:
                        r["native_graph_calls"] += 1
                    else:
                        # requested false but selected true: not a native path
                        rows_bad += 1
                else:
                    if gsel:
                        if nc_good and pc_good:
                            r["candidate_graph_calls"] += 1
                            r["selected_graph_tokens"].append(gt)
                        else:
                            rows_bad += 1
                    else:
                        r["candidate_graph_fallbacks"] += 1

        dispatch = frame.get("dispatch")
        if not isinstance(dispatch, list):
            rows_bad += 1
        else:
            for d in dispatch:
                if not isinstance(d, dict):
                    rows_bad += 1
                    continue
                dN = d.get("N")
                present = type(dN) is int and dN > 0
                if not present:
                    rows_bad += 1
                    continue
                preq = d.get("prefix_requested")
                dsel = d.get("selected")
                if type(preq) is not bool or type(dsel) is not bool or preq != requested:
                    rows_bad += 1
                    continue
                olt = d.get("output_list_type")
                if not preq:
                    if dsel is False and type(olt) is int and olt == 1:
                        r["native_dispatch_calls"] += 1
                    else:
                        rows_bad += 1
                else:
                    if dsel:
                        if type(olt) is int and olt == 0:
                            r["candidate_dispatch_calls"] += 1
                            r["selected_dispatch_N"].append(dN)
                        else:
                            rows_bad += 1
                    else:
                        r["candidate_dispatch_fallbacks"] += 1

        r["invalid_rows"] += rows_bad

    for r in per_rank.values():
        nc = r["native_graph_calls"] > 0
        cc = r["candidate_graph_calls"] > 0
        nd = r["native_dispatch_calls"] > 0
        cd = r["candidate_dispatch_calls"] > 0
        fall = (r["candidate_graph_fallbacks"] + r["candidate_dispatch_fallbacks"]
                + r["invalid_rows"]) == 0
        r["both_native_and_candidate_paths_present"] = nc and cc and nd and cd and fall

    for r in per_rank.values():
        r["selected_graph_tokens"] = sorted(set(r["selected_graph_tokens"]))
        r["selected_dispatch_N"] = sorted(set(r["selected_dispatch_N"]))

    observed = sorted(per_rank.keys())
    per_rank_list = [per_rank[k] for k in observed]

    all16 = (set(observed) == set(range(16))
             and all(r["both_native_and_candidate_paths_present"] for r in per_rank_list)
             and global_invalid == 0)

    return {
        "completed": True,
        "observed_ranks": observed,
        "per_rank": per_rank_list,
        "all16_paths_verified": all16,
        "global_invalid_rows": global_invalid,
        "scope": "prefix path metadata only: graph/dispatch requested/selected flags, "
                 "native vs prefix capture call counts and output list types on real frames "
                 "(nonempty request_ids); no latency, correctness, SDMA, or quality claims, "
                 "and no per-request cost inference across mixed/frame batches.",
    }


records=[];files=[]
for p in sorted((root/'moe-prefix-histogram-probe').glob('rank-*.jsonl')):
 b=p.read_bytes();files.append({'path':p.name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
 records.extend(json.loads(line) for line in b.decode().splitlines())
diagnostic=[z for z in records if any('prefix3diag-' in rid for rid in z.get('request_ids',[]))]
ordinary=[z for z in records if not any('prefix3diag-' in rid for rid in z.get('request_ids',[]))]
assert len(diagnostic)+len(ordinary)==len(records)
summary=summarize_prefix_paths(ordinary)
summary.update(excluded_diagnostic_namespace='prefix3diag-',excluded_diagnostic_frames=len(diagnostic),included_ordinary_frames=len(ordinary))
summary.update(CID=cid,files=files,marker_present=False,raw_frames=len(records),Rope_enabled_for_both_modes=(root/'rope-cache-layout-probe/enable').exists())
assert summary['Rope_enabled_for_both_modes']
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(summary)+'\nGLM53_RESULT_JSON_END')
