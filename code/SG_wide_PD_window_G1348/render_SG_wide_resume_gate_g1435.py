"""Render the wide W4/W5 and first-formal gates from the pinned 32K gate."""

import argparse
from hashlib import sha256
from pathlib import Path


SOURCE_SHA256 = '8a430c60a4490c8f5391610f531666ca28d512db819aab0dc6565bc83251347f'


def render(source):
    assert sha256(source).hexdigest() == SOURCE_SHA256
    text = source.decode()
    pairs = (
        ('import urllib.request\n',
         'import urllib.request\n\nfrom wide_initial_gate_context_g1434 import PLAN_BY_WIDTH, binding, load_plan\n'
         'from wide_resume_context_g1435 import bound_prefill_comparison, bound_prefill_snapshot, remaining_seconds\n'),
        ("    run = root/'sg_pd_g1317'\n",
         "    epoch, _ = PLAN_BY_WIDTH[args.length]\n    run = root/f'sg_pd_g{epoch}'\n"
         "    plan, plan_sha = load_plan(run/'wide_workload_plan.json', args.length)\n"),
        ("    assert p['handle']['epoch']==d['handle']['epoch']==1317\n"
         "    assert d['KV_waiting_timeout_actual_by_rank']=={str(rank):14400 for rank in range(8)}\n",
         "    assert p['handle']['epoch']==d['handle']['epoch']==epoch\n"),
        ("    assert d['handle']['guard_PID'] == router['guard_PID'] == pd['fresh_live_guard']['pid'] == 2227468\n"
         "    assert router['router_PID']==2308473 and router['configured_request_deadline_seconds']==14400\n",
         "    prior=client/'runs'/'sglang'/args.run_id\n"
         "    warm,warm_sha=load(prior/'state.json')\n"
         "    previous,previous_sha=load(prior/'run_manifest.json')\n"
         "    required=remaining_seconds(args.stage,args.next_warmup,warm,previous,plan,now)\n"
         "    cpu_path=args.cpu_lease.resolve(strict=True)\n"
         "    assert cpu_path.is_relative_to(client)\n"
         "    cpu,cpu_sha=load(cpu_path)\n"
         "    wide=binding(now,plan,plan_sha,args.run_id,p,d,pd,cpu,required_seconds=required)\n"
         "    old=previous['gate']['warmup_wall_budget']\n"
         "    assert all(old[key]==value for key,value in wide.items())\n"
         "    assert previous['client_container_id']==cpu['container_ID']\n"
         "    cpu_info=json.loads(subprocess.check_output(['docker','inspect',cpu['container_ID']],timeout=10))[0]\n"
         "    assert cpu_info['State']['Running'] and not cpu_info['HostConfig']['Devices']\n"
         "    assert cpu_info['Config']['Labels']['codex.controller_pid']==str(cpu['CPU_controller_PID'])\n"
         "    assert cpu_info['Name']==f'/yxy-ds41-aisbench-g1424-{args.length // 1024}k'\n"),
        ("state['generation']==1317", "state['generation']==epoch"),
        ("        counter=select_counter(values,metric,labels)\n", ''),
        ("            'metrics_url':base+'/metrics','prefix_hit_counter':metric,'prefix_hit_counter_labels':labels,\n"
         "            'device_hit_count_at_gate':counter,'verified_args':opts}\n",
         "            'metrics_url':base+'/metrics','verified_args':opts}\n"
         "        if role=='prefill':\n"
         "            roles[role].update(prefix_hit_counter=metric,prefix_hit_counter_labels=labels,\n"
         "                               device_hit_count_at_gate=select_counter(values,metric,labels))\n"),
        ("    prior=client/'runs'/'sglang'/args.run_id\n"
         "    warm,warm_sha=load(prior/'state.json')\n"
         "    previous,previous_sha=load(prior/'run_manifest.json')\n"
         "    assert previous['run_id']",
         "    assert previous['run_id']"),
        ("previous['gate']['profile_epoch']==1317", "previous['gate']['profile_epoch']==epoch"),
        ("        if completed>=3:\n"
         "            snapshot,p_baseline_sha=load(args.prefill_warmup3_snapshot)\n"
         "            sample=snapshot['summary']\n"
         "            assert sample['epoch']==1317 and sample['P_controller_PID']==router['P_controller_PID']\n"
         "            assert sample['P_container_ID']==roles['prefill']['container_id']\n"
         "            assert sample['boundary_label']==f'after_warmup3_{args.length}' and sample['guard_FD_verified']\n"
         "            assert sample['ready']==200 and all(v and all(x==0 for x in v) for v in sample['idle'].values())\n"
         "            assert datetime.datetime.fromisoformat(sample['at'])>=datetime.datetime.fromisoformat(previous['benchmark_started_at'])\n"
         "        assert args.prefill_compile is None\n",
         "        _,p_baseline_sha=bound_prefill_snapshot(args.prefill_warmup3_snapshot,3,plan,p,previous)\n"
         "        assert args.prefill_warmup5_snapshot is None\n"),
        ("        assert args.next_warmup is None and args.prefill_warmup3_snapshot is None\n",
         "        assert args.next_warmup is None and args.prefill_warmup5_snapshot is not None\n"),
        ("        prefill_compile,compare_sha=load(args.prefill_compile)\n",
         "        prefill_compile=bound_prefill_comparison(args.prefill_warmup3_snapshot,args.prefill_warmup5_snapshot,plan,p,previous)\n"),
        ("        prefill_compile['comparison_SHA256']=compare_sha\n", ''),
        ("'profile_epoch':1317", "'profile_epoch':epoch"),
        ("        'D_prefix_counter_scope':'Observed decode prefill-effective device series. P transfer is recorded separately.'}\n"
         "    if args.next_warmup in (2,3,4):\n",
         "        'D_prefix_counter_scope':'Decode cache hits require independent evidence.'}\n"
         "    if args.stage=='formal':\n"
         "        workload=plan['workload']\n"
         "        out['formal_wall_budget']={'phase_seconds':{'cold':workload['cold_phase_seconds'],\n"
         "            'hot':workload['replay_phase_seconds']},'resource_deadlines_UTC':wide['resource_deadlines_UTC'],\n"
         "            'CPU_client_container_id':cpu['container_ID'],'wide_preparation_sha256':plan_sha,\n"
         "            'CPU_client_lease_sha256':cpu_sha}\n"
         "    else:\n"
         "        out['warmup_wall_budget']={**wide,'CPU_client_lease_sha256':cpu_sha}\n"
         "    if args.next_warmup==4:\n"),
        ("G1331_SG_AISBENCH_GATE_JSON ", "G1435_SG_WIDE_RESUME_GATE_JSON "),
        ("    cli.add_argument('--prefill-compile',type=Path)\n",
         "    cli.add_argument('--prefill-warmup5-snapshot',type=Path)\n"
         "    cli.add_argument('--cpu-lease',type=Path,required=True)\n"),
        ("    cli.add_argument('--prefill-warmup3-snapshot',type=Path)\n",
         "    cli.add_argument('--prefill-warmup3-snapshot',type=Path,required=True)\n"),
        ("choices=(2,3,4,5)", "choices=(4,5)"),
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
