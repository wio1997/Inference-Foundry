from pathlib import Path
import json

n=Path(__file__).parent
a=json.loads((n/'profile-first-20261031/moe-db-targets-v7.json').read_text())['targets']
assert 0<len(a)<=8
p=n/'query_dspark_moe_v7_166_v2.py'
p.write_text('ANCHORS = '+repr(a)+'\n'+(n/'query_dspark_moe_v7_template.py').read_text())
print(json.dumps(dict(script=str(p),moe_anchors=len(a),scope='One archivedEP3 DB, full saved DSpark submission connections and8selectedMC2 anchors; no source/NPU work')))
