from pathlib import Path
from decimal import Decimal
from collections import Counter
from datetime import datetime, timezone
import json, hashlib, statistics

R = Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
OUT = Path(__file__).parent / 'profile-first-20261031'
SIDE = R / 'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'


def stat(vals):
    return dict(n=len(vals), min=min(vals), median=statistics.median(vals), max=max(vals)) if vals else None


def tasks_of(scope):
    tasks = []
    for group in scope['operator_groups']:
        for a in group['tasks']:
            c = a['csv']
            start = int(Decimal(a['device_start_us']) * 1000)
            duration = int(Decimal(a['trace_duration_us']) * 1000)
            tasks.append(dict(type=c['Type'], name=c['Name'], task_id=c['Task ID'], flow=a['flow_id'],
                              stream=c['Stream ID'], start_ns=start, end_ns=start + duration,
                              duration_ns=duration, host_start_us=a['host_start_us'], host_pid=a['host_pid'],
                              host_tid=a['host_tid'], input_shapes=c['Input Shapes'], output_shapes=c['Output Shapes'],
                              input_dtypes=c['Input Data Types'], output_dtypes=c['Output Data Types']))
    return sorted(tasks, key=lambda t: (t['start_ns'], t['end_ns']))


def main():
    assert hashlib.sha256(SIDE.read_bytes()).hexdigest() == '0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
    cfg = json.loads((R / 'evidence/model-config.json').read_text())
    cfg = cfg.get('text_config', cfg)
    # This label is a frozen-configuration potential hook, not observed runtime attribution.
    engram_layers = cfg.get('engram_layer_ids', cfg.get('engram_layers', [1, 14]))
    scopes = []
    bykey = {}
    for line in SIDE.open():
        data = json.loads(line)
        if data.get('kind') != 'rank':
            continue
        for s in data['data']['layers']:
            tasks = tasks_of(s)
            pre = [t for t in tasks if t['type'] == 'HcPre']
            post = [t for t in tasks if t['type'] == 'HcPost']
            row = dict(ep=s['ep'], layer=s['layer'], scope_id=s['layer_scope_id'], digest=s['scheduler_digest'],
                       rank=s['rank'], pre_count=len(pre), post_count=len(post), pre=pre, post=post)
            if len(pre) == len(post) == 2:
                a, b = post[0], pre[1]
                row['intralayer_attn_post_to_ffn_pre'] = dict(
                    post=a, pre=b, gap_ns=b['start_ns'] - a['end_ns'], same_stream=a['stream'] == b['stream'],
                    intervening_sidecar=[t for t in tasks if t['start_ns'] >= a['end_ns'] and t['end_ns'] <= b['start_ns']],
                    tensor_identity_proved=False)
            key = (s['ep'], s['scheduler_digest'], s['layer'])
            assert key not in bykey, ('duplicate_scope_key', key)
            bykey[key] = row
            scopes.append(row)
    assert len(scopes) == 1280
    edges = []
    for (ep, digest, layer), a in bykey.items():
        b = bykey.get((ep, digest, layer + 1))
        if b is None:
            continue
        edge = dict(ep=ep, digest=digest, from_layer=layer, to_layer=layer + 1,
                    from_scope=a['scope_id'], to_scope=b['scope_id'], rank=a['rank'],
                    configured_potential_engram_before_to_layer=(layer + 1 in engram_layers),
                    post_count=a['post_count'], next_pre_count=b['pre_count'])
        if a['post_count'] == b['pre_count'] == 2:
            post, pre = a['post'][-1], b['pre'][0]
            edge.update(post=post, pre=pre, gap_ns=pre['start_ns'] - post['end_ns'], same_stream=post['stream'] == pre['stream'])
        edges.append(edge)
    complete = [x for x in scopes if 'intralayer_attn_post_to_ffn_pre' in x]
    matched_edges = [e for e in edges if 'gap_ns' in e]
    intra = [s['intralayer_attn_post_to_ffn_pre'] for s in complete]
    targets = []
    for s in scopes:
        if s['ep'] == 3 and s['layer'] in (0, 1, 13, 14, 36, 37, 38, 39):
            targets.append(dict(kind='scope_hc', **s))
    summary = dict(generated_utc=datetime.now(timezone.utc).isoformat(), retained_scopes=len(scopes),
                   pre_count_hist=dict(Counter(s['pre_count'] for s in scopes)), post_count_hist=dict(Counter(s['post_count'] for s in scopes)),
                   intralayer=dict(complete_scopes=len(intra), gap_ns=stat([x['gap_ns'] for x in intra]),
                                   same_stream=sum(x['same_stream'] for x in intra),
                                   intervening_counts=dict(Counter(len(x['intervening_sidecar']) for x in intra)),
                                   intervening_types=dict(Counter(t['type'] for x in intra for t in x['intervening_sidecar']))),
                   crosslayer=dict(scope_pairs=len(edges), matched_pairs=len(matched_edges),
                                   gap_ns=stat([x['gap_ns'] for x in matched_edges]),
                                   same_stream=sum(x['same_stream'] for x in matched_edges),
                                   configured_engram_layers=engram_layers,
                                   groups={str(flag):dict(n=len(g), gap_ns=stat([x['gap_ns'] for x in g]))
                                           for flag in (True, False)
                                           for g in [[e for e in matched_edges if e['configured_potential_engram_before_to_layer'] == flag]]}),
                   first_target_pre_shapes=dict(Counter(t['input_shapes'] for s in scopes if s['layer'] == 0 for t in s['pre'][:1])),
                   source_path=str(SIDE), source_sha256=hashlib.sha256(SIDE.read_bytes()).hexdigest(),
                   new_target_source_read=False, new_device_work=False, overall_complete=False,
                   limits=['All retained scopes queried; sidecar association may miss intervening/outside-layer tasks and communication.',
                           'Scope key pairing is not tensor identity or a request/iteration identity proof.',
                           'Order/shape/stream/gap do not prove legal fusion or removable time.',
                           'Configured potential Engram hook labels do not prove actual runtime branch.',
                           'Crosslayer intervals on this rank do not yield crossrank critical path or request latency.'])
    (OUT / 'hc-boundaries-query.json').write_text(json.dumps(dict(summary=summary, scope_results=scopes, crosslayer_edges=edges), ensure_ascii=False, indent=2) + '\n')
    (OUT / 'hc-boundaries-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'hc-boundaries-db-targets.json').write_text(json.dumps(dict(targets=targets, crosslayer_edges=[e for e in edges if e['ep'] == 3 and e['from_layer'] in (0, 13, 36, 37, 38)]), ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
