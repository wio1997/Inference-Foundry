from pathlib import Path
import json

n = Path(__file__).parent
d = json.loads((n / 'profile-first-20261031/hc-boundaries-db-targets.json').read_text())
config = []
for s in d['targets']:
    if s['layer'] == 39:
        a = s['post'][-1]
        config.append(dict(label=f"TAIL_{s['digest'][:8]}_L39_20ms", interval_ns=[a['end_ns'],a['end_ns']+20000000], anchors=[a]))
assert len(config) == 2
template = (n / 'query_archived_hc166_template.py').read_text()
assert template.count('ep3-hc-boundary-task-stack-evidence-v2.json') == 1
template = template.replace('ep3-hc-boundary-task-stack-evidence-v2.json', 'ep3-hc-tail20ms-stack-evidence.json')
script = n / 'query_archived_hc_tail166.py'
script.write_text('CONFIG = ' + repr(config) + '\n' + template)
print(json.dumps(dict(script=str(script), labels=[x['label'] for x in config])))
