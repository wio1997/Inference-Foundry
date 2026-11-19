"""Render a separate wide initial gate from the pinned 32K service gate."""

import argparse
from hashlib import sha256
from pathlib import Path


SOURCE_SHA256 = 'add746993173c6e0abe0e89433868bebf933f3c04bc82b985eec245e5b4d03b0'


def render(source):
    assert sha256(source).hexdigest() == SOURCE_SHA256
    text = source.decode()
    pairs = (
        ('Bind the verified G1317 P/D pair and initial full-width warmup to AISBench.',
         'Bind a verified wide P/D pair and its initial warmup to AISBench.'),
        ('import urllib.request\n', 'import urllib.request\n\nfrom wide_initial_gate_context_g1434 import PLAN_BY_WIDTH, binding, load_plan\n'),
        ("    run = root/'sg_pd_g1317'\n",
         "    epoch, _ = PLAN_BY_WIDTH[args.length]\n    run = root/f'sg_pd_g{epoch}'\n"
         "    plan, plan_sha = load_plan(run/'wide_workload_plan.json', args.length)\n"),
        ("    assert p['guard_FD_verified'] and p['guard_PID']==2227468\n",
         "    assert p['guard_FD_verified']\n"),
        ("    assert p['handle']['epoch'] == d['handle']['epoch'] == 1317\n"
         "    assert d['KV_waiting_timeout_actual_by_rank'] == {str(rank): 14400 for rank in range(8)}\n",
         "    assert p['handle']['epoch'] == d['handle']['epoch'] == epoch\n"),
        ("    assert router['configured_request_deadline_seconds'] == 14400\n",
         "    cpu_path=args.cpu_lease.resolve(strict=True)\n"
         "    assert cpu_path.is_relative_to(client)\n"
         "    cpu, cpu_sha=load(cpu_path)\n"
         "    wide=binding(now,plan,plan_sha,args.run_id,p,d,pd,cpu)\n"
         "    cpu_info=json.loads(subprocess.check_output(['docker','inspect',cpu['container_ID']],timeout=10))[0]\n"
         "    assert cpu_info['State']['Running'] and not cpu_info['HostConfig']['Devices']\n"
         "    assert cpu_info['Config']['Labels']['codex.controller_pid']==str(cpu['CPU_controller_PID'])\n"
         "    assert cpu_info['Name']==f'/yxy-ds41-aisbench-g1424-{args.length // 1024}k'\n"),
        ("    assert p['guard_PID'] == d['handle']['guard_PID'] == router['guard_PID'] == guard['pid'] == 2227468\n", ""),
        ("    assert state['generation']==1317", "    assert state['generation']==epoch"),
        ("    for role, receipt, cid, chunk in", "    disaggregation_queues={\n"
         "      'prefill':('num_prefill_bootstrap_queue_reqs','num_prefill_inflight_queue_reqs'),\n"
         "      'decode':('num_decode_prealloc_queue_reqs','num_decode_transfer_queue_reqs')}\n"
         "    for role, receipt, cid, chunk in"),
        ("        for name in ('num_running_reqs','num_queue_reqs'):",
         "        for name in ('num_running_reqs','num_queue_reqs',*disaggregation_queues[role]):"),
        ('            assert counts and sum(counts)==0,', '            assert counts and all(value==0 for value in counts),'),
        ("        counter=select_counter(values,metric,labels)\n", ''),
        ("            'metrics_url':base+'/metrics','prefix_hit_counter':metric,'prefix_hit_counter_labels':labels,\n"
         "            'device_hit_count_at_gate':counter,'verified_args':opts}\n",
         "            'metrics_url':base+'/metrics','verified_args':opts}\n"
         "        if role=='prefill':\n"
         "            roles[role].update(prefix_hit_counter=metric,prefix_hit_counter_labels=labels,\n"
         "                               device_hit_count_at_gate=select_counter(values,metric,labels))\n"),
        ("'profile_epoch':1317", "'profile_epoch':epoch"),
        ("        'D_prefix_counter_scope':'Observed decode prefill-effective device series. P transfer is recorded separately.'}",
         "        'D_prefix_counter_scope':'Decode cache hits require independent evidence.',\n"
         "        'warmup_wall_budget':{**wide,'CPU_client_lease_sha256':cpu_sha}}"),
        ("G1324_SG_AISBENCH_GATE_JSON ", "G1434_SG_WIDE_INITIAL_GATE_JSON "),
        ("    cli.add_argument('--run-id',required=True)\n",
         "    cli.add_argument('--run-id',required=True)\n    cli.add_argument('--cpu-lease',type=Path,required=True)\n"),
        ("choices=(32768,65536,131072,262144)", "choices=(65536,131072,262144)"),
    )
    for old, new in pairs:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    return text.encode()


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('source', type=Path)
    cli.add_argument('output', type=Path)
    args = cli.parse_args()
    rendered = render(args.source.read_bytes())
    compile(rendered, str(args.output), 'exec')
    with args.output.open('xb') as stream:
        stream.write(rendered)
