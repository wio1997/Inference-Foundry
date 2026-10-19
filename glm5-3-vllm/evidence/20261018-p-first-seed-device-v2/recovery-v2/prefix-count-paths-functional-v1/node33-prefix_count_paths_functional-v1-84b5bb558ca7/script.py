from pathlib import Path
import hashlib,json,subprocess
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/prefix-count-live1-20261019/node33');cid='8f434957c821dec55a77d352a68d16c14f2558d0b4c7e11287e0e37181111fdd';d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];plan=json.loads((root/'plan.json').read_text());assert d['State']['Running'] and d['Name']=='/'+plan['container']['name'] and d['Image']==plan['image']['id'];probe=root/'moe-prefix-gmm2-count-probe';assert (probe/'target-only').exists() and not(probe/'enable').exists() and not(probe/'MTP-only').exists() and not(probe/'replica-enable').exists()
from typing import Dict, List, Any


def summarize_target_prefix_paths(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    global_invalid = 0
    per: Dict[int, Dict[str, Any]] = {}

    def col(rank: int) -> Dict[str, Any]:
        if rank not in per:
            per[rank] = {
                "rank": rank,
                "native_graph_calls": 0,
                "candidate_graph_calls": 0,
                "MTP_native_on_native_requests": 0,
                "MTP_native_on_candidate_requests": 0,
                "native_eager_target_dispatch_calls": 0,
                "invalid_rows": 0,
                "selected_graph_tokens": [],
                "MTP_N": [],
                "native_graph_tokens": [],
                "native_eager_target_N": [],
            }
        return per[rank]

    def valid_rank(rk: Any) -> bool:
        return isinstance(rk, int) and not isinstance(rk, bool) and 0 <= rk <= 15

    def is_bool(v: Any) -> bool:
        return isinstance(v, bool)

    TARGET_TOKENS = (6, 12, 18, 24)

    if not isinstance(records, list):
        records = [None]

    for frame in records:
        if not isinstance(frame, dict):
            global_invalid += 1
            continue
        rid = frame.get("request_ids")
        # empty list => dummy, ignore whole frame
        if isinstance(rid, list) and not rid:
            continue
        rank = frame.get("rank")
        if not valid_rank(rank):
            global_invalid += 1
            continue
        r = col(rank)
        if not isinstance(rid, list) or not rid or not all(isinstance(v, str) and v for v in rid):
            r["invalid_rows"] += 1
            continue
        phase = frame.get("phase")
        requested = frame.get("requested")
        if phase not in ("execute_model", "sample_tokens") or not is_bool(requested):
            r["invalid_rows"] += 1
            continue
        row_bad = 0
        graphs = frame.get("graphs")
        if not isinstance(graphs, list):
            row_bad += 1
        else:
            for g in graphs:
                if not isinstance(g, dict):
                    row_bad += 1
                    continue
                tk = g.get("tokens")
                if not (isinstance(tk, int) and not isinstance(tk, bool) and tk in TARGET_TOKENS):
                    row_bad += 1
                    continue
                g_req = g.get("requested")
                g_sel = g.get("selected")
                if not is_bool(g_req) or not is_bool(g_sel):
                    row_bad += 1
                    continue
                if g_req != requested or g_sel != requested:
                    # requested mismatch with frame or selected != requested => fallback/invalid
                    row_bad += 1
                    continue
                fam = g.get("selected_family")
                if fam != ("prefix" if requested else "native"):
                    row_bad += 1
                    continue
                nc = g.get("native_capture")
                pc = g.get("prefix_capture")
                if not (isinstance(nc, dict) and isinstance(pc, dict)):
                    row_bad += 1
                    continue
                def capture_ok(c, selected, list_type):
                    values = c.get("output_list_types")
                    return (type(c.get("calls")) is int and c["calls"] == 75
                            and type(c.get("selected_calls")) is int and c["selected_calls"] == selected
                            and type(values) is list and len(values) == 1
                            and type(values[0]) is int and values[0] == list_type)
                if not (capture_ok(nc, 0, 1) and capture_ok(pc, 75, 0)):
                    row_bad += 1
                    continue
                # graph rows only count in execute_model
                if phase != "execute_model":
                    row_bad += 1
                    continue
                if requested:
                    r["candidate_graph_calls"] += 1
                else:
                    r["native_graph_calls"] += 1
                token_key = "selected_graph_tokens" if requested else "native_graph_tokens"
                if tk not in r[token_key]:
                    r[token_key].append(tk)
        dispatch = frame.get("dispatch")
        if not isinstance(dispatch, list):
            row_bad += 1
        else:
            for d in dispatch:
                if not isinstance(d, dict):
                    row_bad += 1
                    continue
                N = d.get("N")
                if not (isinstance(N, int) and not isinstance(N, bool) and 1 <= N <= 24):
                    row_bad += 1
                    continue
                preq = d.get("prefix_requested")
                dsel = d.get("selected")
                olt = d.get("output_list_type")
                if not (is_bool(preq) and is_bool(dsel)):
                    row_bad += 1
                    continue
                if not (isinstance(olt, int) and not isinstance(olt, bool)):
                    row_bad += 1
                    continue
                if preq is not False or dsel is not False or olt != 1:
                    row_bad += 1
                    continue
                shape_key = "MTP_N" if phase == "sample_tokens" else "native_eager_target_N"
                if N not in r[shape_key]:
                    r[shape_key].append(N)
                if phase == "execute_model":
                    r["native_eager_target_dispatch_calls"] += 1
                    # original eager target fallback outside captured target scope;
                    # do not claim prefix here
                elif phase == "sample_tokens":
                    pass  # decided below
                # split MTP native by requested for sample_tokens proof
                # (dispatch in execute_model is eager target fallback, not MTP)
                if phase == "sample_tokens":
                    if requested:
                        r["MTP_native_on_candidate_requests"] += 1
                    else:
                        r["MTP_native_on_native_requests"] += 1
        r["invalid_rows"] += row_bad

    per_list = [col(k) for k in sorted(per.keys())]
    for r in per_list:
        r["selected_graph_tokens"].sort()
        r["MTP_N"].sort()
        r["native_graph_tokens"].sort()
        r["native_eager_target_N"].sort()

    all16 = (set(per.keys()) == set(range(16))
             and all((r["native_graph_calls"] > 0
                      and r["candidate_graph_calls"] > 0
                      and r["MTP_native_on_native_requests"] > 0
                      and r["MTP_native_on_candidate_requests"] > 0
                      and r["invalid_rows"] == 0)
                     for r in per_list)
             and global_invalid == 0)

    return {
        "completed": True,
        "observed_ranks": sorted(per.keys()),
        "all16_paths_verified": all16,
        "global_invalid_rows": global_invalid,
        "per_rank": per_list,
        "scope": ("path metadata only for target-prefix experiment: target graph requested -> "
                  "selected_family prefix with native/prefix capture 75/75 and [1]/[0] lists, "
                  "requested False -> native graph; MTP keeps original native count dispatch in "
                  "sample_tokens for both native and candidate requests; dispatch in "
                  "execute_model is original eager target fallback outside captured target "
                  "scope and prefix is not claimed there. Counts report shape/path coverage "
                  "only — no correctness, SDMA, latency, or quality claim."),
    }

records=[];files=[]
for p in sorted(probe.glob('rank-*.jsonl')):
 b=p.read_bytes();files.append({'path':p.name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()});records.extend(json.loads(s) for s in b.decode().splitlines())
restores=[];restore_files=[]
for p in sorted(probe.glob('gmm2-count-rank-*.jsonl')):
 b=p.read_bytes();restore_files.append({'path':p.name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()});restores.extend(json.loads(s) for s in b.decode().splitlines())
assert sorted(set(z['rank'] for z in restores))==list(range(16))
for z in restores:
 assert type(z['rank']) is int and z['N'] in [6,12,18,24] and all(type(z[k]) is int and z[k]==75 for k in ['prefix','fused','restored']) and z['native_routing_prefix_type']==0 and z['native_GMM2_histogram_type']==1 and z['GMM1_prefix_dtype']=='torch.int64' and z['recovered_GMM2_count_dtype']=='torch.int64' and z['recovery_at']=='GMM2' and z['recovery_utility']=='cumsum_group_list(0,1)'
assert all({z['N'] for z in restores if z['rank']==rank}=={6,12,18,24} for rank in range(16))
summary=summarize_target_prefix_paths(records);summary.update(actual_prefix_and_native_GMM2_capture_rows=restores,restore_files=restore_files,all16_native_GMM2_restore_proved=True);
summary.update(CID=cid,files=files,raw_frames=len(records),marker_present=False,target_only_selector=True,MTP_native_in_both_modes=True,Rope_enabled_for_both_modes=(root/'rope-cache-layout-probe/enable').exists())
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(summary)+'\nGLM53_RESULT_JSON_END')
