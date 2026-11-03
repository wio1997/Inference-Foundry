from pathlib import Path
import json

n = Path(__file__).parent
d = json.loads((n / 'profile-first-20261031/hc-boundaries-db-targets.json').read_text())
config = []
for edge in d['crosslayer_edges']:
    config.append(dict(label=f"CROSS_{edge['digest'][:8]}_L{edge['from_layer']}_L{edge['to_layer']}",
                       interval_ns=[edge['post']['end_ns'], edge['pre']['start_ns']],
                       anchors=[edge['post'], edge['pre']]))
for s in d['targets']:
    if s['layer'] != 0:
        continue
    a, b = s['post'][0], s['pre'][1]
    config.append(dict(label=f"INTRA_{s['digest'][:8]}_L0", interval_ns=[a['end_ns'], b['start_ns']], anchors=[a, b]))
    pre = s['pre'][0]
    config.append(dict(label=f"LEADING_{s['digest'][:8]}_L0_5ms", interval_ns=[pre['start_ns']-5000000, pre['start_ns']], anchors=[pre]))
for s in d['targets']:
    if s['layer'] != 39:
        continue
    a = s['post'][-1]
    config.append(dict(label=f"TAIL_{s['digest'][:8]}_L39_3ms", interval_ns=[a['end_ns'], a['end_ns']+3000000], anchors=[a]))
assert len(config) == 16
script = n / 'query_archived_hc_boundaries166_v2.py'
script.write_text('CONFIG = ' + repr(config) + '\n' + (n / 'query_archived_hc166_template.py').read_text())
print(json.dumps(dict(script=str(script), intervals=len(config), labels=[c['label'] for c in config])))
