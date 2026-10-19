from pathlib import Path
import ast,hashlib,json
repo=Path('/Users/wio/work/Inference-Foundry-execution-plan');r=repo/'glm5-3-vllm/evidence/20261018-p-first-seed-device-v2/recovery-v2';source=repo/'glm5-3-vllm/runtime-patches/p-first-seed-v2/validate_sfa_replay_order.py';s=source.read_text();assert hashlib.sha256(source.read_bytes()).hexdigest()=='3cd48a945cbd1d5dc770e4098179c926918fd9c46b0744855e5c14b7656d1cc0';sp=r/'prefix-count-fixed-arrival-schedule-v1.json';schedule=json.loads(sp.read_text());digest=hashlib.sha256(sp.read_bytes()).hexdigest()
s=s.replace("D_ROOT = Path(", 'FROZEN_SCHEDULE = '+repr(schedule)+'\nFROZEN_SCHEDULE_SHA256 = '+repr(digest)+'\n\nD_ROOT = Path(',1)
start=s.index('    def run_c4(cases):');end=s.index('\n    try:\n        idle()',start)
new='''    def run_c4(cases):
        assert args.phase == 'c4' and args.c4_cycles == 1
        assert args.probe_kind == 'moe_prefix_gmm2_count' and args.run_tag == 'fixed-arrival-v1'
        by_id = {case['case_id']: case for case in cases}
        schedule = FROZEN_SCHEDULE['rows']
        assert len(schedule) == 32 and all(a['offset_ns'] <= b['offset_ns'] for a, b in zip(schedule, schedule[1:]))
        for item in schedule:
            case = by_id[item['case_id']]
            assert case['prompt_sha256'] == item['prompt_sha256'] and case['output_tokens'] == item['output_tokens']
        report['fixed_arrival_control'] = dict(schedule_sha256=FROZEN_SCHEDULE_SHA256, source=FROZEN_SCHEDULE,
            condition='Same external arrivals in all4 blocks;32 client workers permit legitimate queueing instead of completion-coupled resubmission. Original service max_num_seqs4 unchanged.',
            dispatch_gate='p95 delay<=10ms and max<=25ms in both modes; otherwise timing invalid. No unchanged benchmark repeat.')
        for mode in ['native', 'adaptive']:
            select(mode)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda case: one(case, mode, 'warmup'), cases))
            save()
        for index, mode in enumerate(['native', 'adaptive', 'adaptive', 'native']):
            select(mode)
            counter_before = report['idle_checks'][-1].get('native_counters') if args.retain_native_counters else None
            start_ns = time.monotonic_ns() + 200_000_000
            block = dict(index=index, mode=mode, started_ns=start_ns)
            report['blocks'].append(block)
            def scheduled(item):
                target = start_ns + item['offset_ns']
                left = (target-time.monotonic_ns())/1e9
                if left > 0: time.sleep(left)
                row = one(by_id[item['case_id']], mode, 'formal', index)
                row.update(planned_dispatch_ns=target, dispatch_lag_ms=(row['sent_ns']-target)/1e6,
                           source_arrival_offset_ns=item['offset_ns'])
                return row
            with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
                actual = list(pool.map(scheduled, schedule))
            block.update(finished_ns=max(z['finished_ns'] for z in actual), requests=32,
                         output_tokens=sum(z['usage']['completion_tokens'] for z in actual))
            block['seconds'] = (block['finished_ns']-start_ns)/1e9
            block['output_tokens_per_s'] = block['output_tokens']/block['seconds']
            assert block['output_tokens'] == 3072
            if counter_before is not None: block['native_counter_boundary'] = counter_boundary(counter_before, 3072)
            save()
        lags = {}
        for mode in ['native', 'adaptive']:
            values = sorted(z['dispatch_lag_ms'] for z in report['rows'] if z['phase'] == 'formal' and z['mode'] == mode)
            assert len(values) == 64 and min(values) >= 0
            i = (len(values)-1)*.95
            lags[mode] = dict(p95=values[int(i)]+(values[min(int(i)+1,len(values)-1)]-values[int(i)])*(i%1), max=max(values))
        report['fixed_arrival_control'].update(dispatch_lag_ms=lags,
            timing_valid=all(v['p95']<=10 and v['max']<=25 for v in lags.values()))
'''
s=s[:start]+new+s[end:];ast.parse(s);out=r/'validate_prefix_count_fixed_arrival_v1.py';assert not out.exists();out.write_text(s)
contract={'hypothesis':'Original closed-loop C4 tail regression may include completion-coupled arrival/admission feedback; identical external arrivals distinguish it from a persistent within-service tail effect. No causal/kernel claim from counterfactual alone.','source_schedule_sha256':digest,'validator_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'original_formal_verdict':'MIXED_DEFAULT_OFF_PREFIX_COUNT_NO_JOINT_GAIN; preserved unchanged','change':'Only externally timed arrivals and32 load-generator workers, same native PD model+image+operators+precision+SDMA+RoPE+MTP+helper and server parameters','schedule':'First original native block32 requests, each of4 cases8; offsets frozen17.52641221s','design':'One ABBA,32 requests/block=128 formal plus8 warm; no parameter sweep or extra run. Exact native counters outside timers.','required':'Fresh own resource/current exact CID/idle preflight, observer/profiler/export/related transfers off, all numeric/full output budgets; dispatch lag p95<=10ms and max<=25ms each arm','interpretation':'All pooled and per-case p95 TTFT/completion/meanTPOT plus fixed token-delivery percentiles. If previous regression vanishes/reduces, supports arrival-feedback contribution in this finite trace. If persists, retain within-service hypotheses and choose a new targeted diagnostic; invalid timing -> no effect conclusion. This control does not overturn old gate or adopt the candidate.'};(r/'prefix-count-fixed-arrival-contract-v1.json').write_text(json.dumps(contract,indent=2)+'\n');print(json.dumps({'validator_sha256':contract['validator_sha256'],'schedule_sha256':digest,'formal_requests':128,'warmup_requests':8}))
