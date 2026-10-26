#!/usr/bin/env python3
"""Run401 fail-closed offline analysis. Missing timing/control evidence => INCONCLUSIVE.
A positive margin is instrumented local ordering, never a universal happens-before proof.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


class CorrectnessFailure(ValueError):
    pass


def need(ok, message):
    if not ok:
        raise ValueError(message)


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def record(row, timing):
    need(row['schema'] == 2, 'schema')
    if row['generation_error'] is not None:
        raise CorrectnessFailure('generation/storage correctness failure: '+str(row['generation_error']))
    need(not row['parking'] and row['reason'] == 'normal_proposer', 'not normal unparked branch')
    need(row['commit_pending'] is True, 'commit pending flag')
    need(row['draft_graph'] is False, 'Draft graph mode')
    need(row['cycles'] > 65 and len(row['accepted_counts']) == row['cycles'], 'count history coverage')
    need(len(row['remaining']) == len(row['initial_output_counts']) == 12, 'batch shape')
    need(all(a+b == 1024 and b+66*8 < 1024 for a,b in
             zip(row['remaining'], row['initial_output_counts'])), 'ineligible output/parking contract')
    for c in (64,65):
        need(len(row['accepted_counts'][c]) == 12 and
             all(type(n) is int and 1 <= n <= 8 for n in row['accepted_counts'][c]), 'inactive/invalid selected counts')
    markers = row['device_records']
    labels = ['R_BEGIN_64','R_DONE_64','W_PRE_65','W_POST_65']
    need(set(markers) == set(labels), 'event coverage')
    need(len({markers[x]['event'] for x in labels} | {row['production_event']}) == 5, 'event reuse')
    need(len({markers[x]['device'] for x in labels}) == 1, 'cross-device comparison')
    need(markers['R_BEGIN_64']['stream'] == markers['R_DONE_64']['stream'] == row['copy_stream'], 'copy stream identity')
    need(markers['W_PRE_65']['stream'] == markers['W_POST_65']['stream'], 'overwrite stream identity')
    for label in labels:
        m = markers[label]
        need(m['entry'] == (64 if label.startswith('R_') else 65) and m['generation'] == 64, 'generation pairing')
        need(m['storage'] == row['source'], 'source/overwrite storage identity')
    need(row['source']['shape'] == [12] and row['source']['dtype'] == 'torch.int32', 'count source contract')
    need(row['destination']['device'] == 'cpu', 'destination CPU identity')
    host = row['host']
    ordered = ['commit_entry','sync_pre','sync_post','count_add_pre','count_add_post',
               'seq_add_pre','seq_add_post','mirrors_done','launch65']
    ns = []
    for key in ordered:
        value = host[key]
        need(type(value['ns']) is int and value['entry'] == 65, 'missing/wrong Host site '+key)
        need(value['generation'] == (65 if key == 'launch65' else 64), 'Host generation '+key)
        ns.append(value['ns'])
    need(ns == sorted(ns), 'Host sync/consumption/relaunch ordering')
    need(host['launch64']['ns'] < ns[0], 'launch64/commit65 host ordering')
    need(host['count_add_pre']['ptr'] == row['destination']['ptr'], 'first count consumer destination alias')
    need(host['seq_add_pre']['ptr'] in [x['ptr'] for x in row['seq_mirrors']], 'sequence mirror alias')
    need(host['progress64']['entry'] == 64 and host['progress65']['entry'] == 65, 'progress entry freeze')
    clone = row.get('first_clone')
    if clone is not None:
        need(host['clone_pre']['ns'] is not None and host['clone_post']['ns'] is not None, 'clone timestamps missing')
        need(ns[-2] <= host['clone_pre']['ns'] <= host['clone_post']['ns'], 'clone Host ordering')
        need(clone['source_pointer'] == host['clone_pre']['ptr'] and clone['output_pointer'] == host['clone_post']['ptr'], 'clone pointer identity')
        need(clone['lineage'] in ('direct_mirror', 'tracked_clone'), 'clone lineage')
        if clone['lineage'] == 'direct_mirror':
            need(clone['source_pointer'] in [x['ptr'] for x in row['seq_mirrors']], 'clone source mirror alias')
    downstream = row['downstream'] is not None and host['downstream_pre']['ns'] is not None
    if downstream:
        need(row['downstream']['site'].endswith(('.add_', '.max_item')), 'clone/reference is not a numeric consumer')
        need(row['downstream']['pointer'] == host['downstream_pre']['ptr'], 'numeric consumer pointer identity')
        if row['downstream']['lineage'] == 'direct_mirror':
            need(row['downstream']['pointer'] in [x['ptr'] for x in row['seq_mirrors']], 'numeric source mirror alias')
        if row['downstream']['lineage'] == 'tracked_clone':
            need(clone is not None and row['downstream']['pointer'] == clone['output_pointer'], 'numeric consumer clone lineage')
            need(host['clone_post']['ns'] <= host['downstream_pre']['ns'], 'numeric consumer before clone completed')
        need(ns[-2] <= host['downstream_pre']['ns'] <= host['downstream_post']['ns'], 'downstream Host ordering')
        need(row['downstream']['lineage'] in ('direct_mirror','tracked_clone'), 'downstream lineage')
    anchor = row['before_anchor_ms']
    need(all(finite(anchor[k]) and anchor[k] >= 0 for k in labels), 'invalid terminal anchor intervals')
    margin_anchor = anchor['R_DONE_64']-anchor['W_PRE_65']
    margin_direct = row['direct']['margin'].get('ms')
    uncertainty = timing['direct_uncertainty_ms'] if margin_direct is not None else timing['anchor_uncertainty_ms']
    margin = margin_direct if margin_direct is not None else margin_anchor
    need(finite(margin), 'invalid direct margin')
    if margin_direct is not None:
        need(abs(margin_direct-margin_anchor) <= timing['direct_uncertainty_ms']+timing['anchor_uncertainty_ms'],
             'direct/anchor disagreement exceeds uncertainty')
    read_ms = anchor['R_BEGIN_64']-anchor['R_DONE_64']
    write_ms = anchor['W_PRE_65']-anchor['W_POST_65']
    need(read_ms >= -timing['anchor_uncertainty_ms'] and write_ms >= -timing['anchor_uncertainty_ms'], 'negative local event envelopes')
    threshold = uncertainty+timing['local_marker_allowance_ms']
    return dict(rank=row['rank'],cohort=row['cohort'],margin_ms=margin,
        margin_anchor_ms=margin_anchor,margin_direct_ms=margin_direct,threshold_ms=threshold,
        observed_ordering=margin > uncertainty, original_schedule_slack_supported=margin > threshold,
        read_envelope_ms=read_ms,write_envelope_ms=write_ms,
        sync_host_ns=host['sync_post']['ns']-host['sync_pre']['ns'], downstream_observed=downstream)


def timing_gates(manifest):
    ref = manifest['timing_report']
    path = Path(ref['path'])
    need(hashlib.sha256(path.read_bytes()).hexdigest() == ref['sha256'], 'timing report hash mismatch')
    report = json.loads(path.read_text())
    script = Path(__file__).with_name('loop078_count_markers_preflight.py')
    need(report['producer_script_sha256'] == hashlib.sha256(script.read_bytes()).hexdigest(), 'timing producer script mismatch')
    need(report['passed'] is True, 'isolated timing preflight did not pass')
    need(sorted(report['devices']) == list(range(8)), 'timing preflight must cover all eight visible devices')
    t = dict(report['timing'])
    t['stack'] = (report['torch_version'], report['torch_npu_version'])
    for key in ('same_device_cross_stream_validated','d2h_completion_host_visibility_validated',
                'lazy_event_preflight_passed'):
        need(t[key] is True, 'timing preflight missing: '+key)
    for key in ('direct_uncertainty_ms','anchor_uncertainty_ms','local_marker_allowance_ms'):
        need(finite(t[key]) and t[key] > 0, 'positive measured uncertainty/allowance required: '+key)
    evidence = manifest['evidence_files']
    need(bool(evidence), 'no pinned external evidence')
    for entry in evidence:
        need(hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() == entry['sha256'], 'external evidence hash mismatch')
    return t


def control_gates(manifest):
    need(not manifest.get("intervening_container_restart"), "A1 container reset separates environment from A0/B")
    for name in ('A0','B','A1'):
        r = manifest['controls'][name]
        for k,v in dict(http_posts=60,max_concurrency=12,outputs_per_request=1024,cohorts=5,
                        errors=0,full_target=True,post_handoff_oracle_calls=0,host_mirror_exact=True,
                        source_restored=True,lifecycle_closed=True).items():
            need(r[k] == v, f'{name}: frozen contract/lifecycle {k}')
        need(finite(r['runtime_wall_s']) and r['runtime_wall_s'] > 0, 'control runtime wall')
        need(bool(r['request_ledger_sha256']), 'missing request ledger')
    controls=manifest['controls']
    need(len({r['run_id'] for r in controls.values()}) == 3, 'control run IDs not distinct')
    need(all(isinstance(r["acceptance_sha256"], str) and len(r["acceptance_sha256"]) == 64 for r in controls.values()), "per-cycle acceptance trajectory unavailable")
    need(len({controls[x]['acceptance_sha256'] for x in ('A0','B','A1')}) == 1, 'acceptance trajectory differs; overhead uninterpretable')
    a0,a1,b=[controls[x]['runtime_wall_s'] for x in ('A0','A1','B')]
    need(min(a0,a1) <= b <= max(a0,a1), 'instrumented wall outside A/A envelope')
    return True


def analyze(rows, manifest):
    timing = timing_gates(manifest)
    need(all((x['torch_version'], x['torch_npu_version']) == timing['stack'] for x in rows), 'preflight/runtime stack mismatch')
    try:
        controls_ok = control_gates(manifest)
        control_reason = None
    except (ValueError, KeyError, TypeError) as exc:
        controls_ok = False
        control_reason = str(exc)
    need(len(rows)==40, 'need five cohorts x eight ranks')
    pairs={(x['cohort'],x['rank']) for x in rows}
    need(pairs=={(c,r) for c in range(1,6) for r in range(8)}, 'cohort/rank coverage')
    need(all(x['run_id']==manifest['controls']['B']['run_id'] for x in rows), 'mixed instrumented run identity')
    need(all(x['phase']==('warmup' if x['cohort']<=4 else 'diagnostic') for x in rows), 'phase labeling')
    for c in range(1,6):
        cohort=[x for x in rows if x['cohort']==c]
        need(all(x['accepted_counts']==cohort[0]['accepted_counts'] for x in cohort), 'rank acceptance parity')
    results=[record(x,timing) for x in rows]
    accepted=all(x['observed_ordering'] and x['downstream_observed'] for x in results) and manifest['controls']['B'].get('host_mirror_exact') is True
    demonstrated_failure = manifest['controls']['B'].get('host_mirror_exact') is False
    return dict(status='REJECTED' if demonstrated_failure else ('ACCEPTED' if accepted else 'INCONCLUSIVE'), records=results,
        original_schedule_extrapolation_supported=controls_ok and all(x['original_schedule_slack_supported'] for x in results),
        control_gate_passed=controls_ok, control_gate_reason=control_reason,
        scope='Observed same-device local generation64 read-before-generation65 overwrite and linked Host consumers only.',
        universal_happens_before_proven=False, algorithm_floor=None, hardware_floor=None, product_ceiling=None)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('capture_dir',type=Path);ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    try:
        rows=[json.loads(p.read_text()) for p in sorted(a.capture_dir.glob('rank*_cohort*.json'))]
        result=analyze(rows,json.loads(a.manifest.read_text()))
    except CorrectnessFailure as exc:
        result=dict(status='REJECTED', reason=str(exc),algorithm_floor=None,hardware_floor=None,product_ceiling=None)
    except (ValueError,KeyError,TypeError,OSError) as exc:
        result=dict(status='INCONCLUSIVE',reason=str(exc),algorithm_floor=None,hardware_floor=None,product_ceiling=None)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],reason=result.get('reason'))))


if __name__=='__main__':main()
