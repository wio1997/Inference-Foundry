"""Finite ordinary PD validation; quality review precedes separate formal phase."""
from pathlib import Path
import argparse, json, subprocess, sys, time

r = Path(__file__).resolve().parent; repo = r.parents[3]
control = '/tmp/glm53-active-20261019.sock'
root = '/data/wio/glm5-3-vllm-execution/20261018-state-tail/prefix-count-live1-20261019/node33'
cid = '8f434957c821dec55a77d352a68d16c14f2558d0b4c7e11287e0e37181111fdd'
pcid = 'a3db778639b1f5d16f49ccff123da323d264bc4eb9aeab982b602f0f921b943a'

def collect(script, label, out, args=(), timeout=1000, node=33):
    dest = r/out
    existing = list(dest.glob('*/metrics.json')) if dest.exists() else []
    if existing:
        assert len(existing) == 1
        x = json.loads(existing[0].read_text()); assert not x.get('failure'); return x
    assert not dest.exists(), 'Inspect partial receipt before a justified new label'
    q = subprocess.run([sys.executable, str(repo/'glm5-3-vllm/tools/collect_node.py'), '--node', str(node), '--script', str(script), '--label', label, '--output', str(dest), '--control', control, '--timeout', str(timeout)]+['--script-arg='+v for v in args], cwd=repo, capture_output=True, text=True, timeout=timeout+180)
    (r/(out+'-collector.txt')).write_text(q.stdout+q.stderr)
    assert q.returncode == 0, q.stdout+q.stderr
    return json.loads(next(dest.glob('*/metrics.json')).read_text())

def run(phase, extra=()):
    x = collect(repo/'glm5-3-vllm/runtime-patches/p-first-seed-v2/validate_sfa_replay_order.py', 'prefix_count_'+phase.replace('-', '_')+'_v1', 'prefix-count-'+phase+'-v1', ('--execute', '--phase', phase, '--probe-kind', 'moe_prefix_gmm2_count', '--run-dir', root, '--expected-D-CID', cid, '--cases', root+'/prefill-padding-cases-v1.json', '--run-tag', 'v1')+extra)
    assert x['completed'] and x['numeric_all_pass'], x.get('failure')
    print(json.dumps(dict(phase=phase, rows=len(x['rows']), numeric_all=True)), flush=True)
    return x

def paths(tag):
    x = collect(r/'read_prefix_count_paths_v1.py', 'prefix_count_paths_'+tag, 'prefix-count-paths-'+tag, timeout=90)
    assert x['observed_ranks'] == list(range(16)) and x['all16_paths_verified'] and x['all16_native_GMM2_restore_proved']
    assert x['MTP_native_in_both_modes'] and x['Rope_enabled_for_both_modes'] and not x['marker_present']
    print(json.dumps(dict(phase='all16_paths_'+tag, verified=True)), flush=True)

def preflight(tag):
    for node in [32, 33]:
        x = collect(r/'preflight_prefix_count_v1.py', 'prefix_count_preflight_'+tag+'_'+str(node), 'prefix-count-preflight-'+tag+'-'+str(node), ('--disable-D-trace',) if node == 33 else (), node=node, timeout=90)
        assert x['remaining_s'] > 900
        print(json.dumps(dict(phase='preflight_'+tag, node=node, remaining_s=x['remaining_s'])), flush=True)

def main():
    p = argparse.ArgumentParser(); p.add_argument('--phase', choices=['correctness', 'formal', 'cancel'], required=True); a = p.parse_args()
    if a.phase == 'correctness':
        start = time.monotonic()
        while not (r/'prefix-count-startup-outcome.json').exists():
            assert time.monotonic()-start < 1300; time.sleep(10)
        assert json.loads((r/'prefix-count-startup-outcome.json').read_text())['both_ready']
        run('functional'); paths('functional-v1')
        proxy = collect(repo/'glm5-3-vllm/runtime-patches/p-first-seed-v2/start_proxy.py', 'prefix_count_official_proxy_v1', 'prefix-count-official-proxy-v1', ('--run-dir', root.replace('node33', 'node32'), '--expected-P-CID', pcid, '--minimum-lifetime-seconds', '900'), timeout=100, node=32)
        assert proxy['ready']; print(json.dumps(dict(phase='official_proxy', CID=proxy['CID'])), flush=True)
        run('c4-functional'); paths('c4-v1'); preflight('quality-v1')
        quality = run('quality'); assert len(quality['sequence_comparisons']) == 26
        same = sum(v['all_same'] for v in quality['sequence_comparisons'].values())
        unequal = [k for k, v in quality['sequence_comparisons'].items() if not v['all_same']]
        run('cached')
        outcome = dict(completed=True, functional16_numeric=True, C4_functional8_numeric=True, quality52_numeric=True, cached16_numeric=True, all16_paths=True, actual_prefix_GMM1_and_count_GMM2=True, MTP_native=True, quality_full64_equal_cases=same, unequal_full64_cases=unequal, next='Primary review before formal. New differing sequences require finite native/replica/candidate/native controls; never assume harmless rounding.')
        (r/'prefix-count-correctness-outcome-v1.json').write_text(json.dumps(outcome, indent=2)+'\n')
        print(json.dumps(outcome), flush=True)
    elif a.phase == 'formal':
        review = json.loads((r/'prefix-count-quality-reviewed-v1.json').read_text()); assert review['proceed_finite_formal']
        assert json.loads((r/'prefix-count-correctness-outcome-v1.json').read_text())['completed']
        preflight('formal-v1'); c1 = run('c1', ('--retain-native-counters',)); c4 = run('c4', ('--rotate-cases', '--c4-cycles', '3', '--retain-native-counters'))
        assert len(c4['blocks']) == 12 and all(z['native_counter_boundary']['delta']['33']['vllm:generation_tokens_total'] == 3072 for z in c4['blocks'])
        outcome = dict(completed=True, C1=c1['summary'], C4=c4['summary'], native_marker_restored=True, P_CID=pcid, D_CID=cid, mode_owned_native_counters_each_block=True, scope='Observer-off finite C1/ABBAx3 C4, native SDMA PD, same RoPE and MTP in both arms. All slow blocks retained. Not general quality or maximum SLO capacity.')
        (r/'prefix-count-formal-outcome-v1.json').write_text(json.dumps(outcome, indent=2)+'\n'); print(json.dumps(outcome), flush=True)
    else:
        assert json.loads((r/'prefix-count-formal-outcome-v1.json').read_text())['completed']
        x = collect(r/'validate_prefix_count_cancel_v1.py', 'prefix_count_cancel_v1', 'prefix-count-cancel-v1', timeout=600)
        assert x['completed'] and x['client_close_requests'] == 10 and x['full_reuse_and_peer_responses'] == 16
        print(json.dumps(dict(phase='cancel', client_close=x['client_close_requests'], full_reuse_and_peer=x['full_reuse_and_peer_responses'], explicit_backend_abort=x['backend_abort_log_request_count'])), flush=True)

if __name__ == '__main__': main()
