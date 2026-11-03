"""Preserve and losslessly deduplicate the derived 22k graph-boundary records."""
from pathlib import Path
from collections import defaultdict
import datetime,gzip,hashlib,json,os
T=Path(__file__).resolve().parents[1]
p=T/'candidates/cross-module-G1.json';gz=T/'candidates/cross-module-G1-original.json.gz';assert p.exists() and not gz.exists()
g=json.loads((T/'graphs/G1.json').read_text()); edges=g['edges']
out=defaultdict(list);inc=defaultdict(list)
for i,e in enumerate(edges):
 if e.get('dependency_kind')!='NEXT_INVOCATION_RAW':out[e['producer']].append(i);inc[e['consumer']].append(i)
original_sha=hashlib.sha256()
with p.open('rb') as a,gzip.open(gz,'wb',compresslevel=6) as b:
 while True:
  chunk=a.read(1024*1024)
  if not chunk:break
  original_sha.update(chunk);b.write(chunk)
check=hashlib.sha256()
with gzip.open(gz,'rb') as f:
 for chunk in iter(lambda:f.read(1024*1024),b''):check.update(chunk)
assert check.hexdigest()==original_sha.hexdigest()
data=json.loads(p.read_text());assert data['count']==len(data['items'])==22802
for row in data['items']:
 S=set(row['nodes']);relevant=sorted({i for n in S for i in out[n]+inc[n]})
 internal=[i for i in relevant if edges[i]['producer'] in S and edges[i]['consumer'] in S]
 inputs=[i for i in relevant if edges[i]['consumer'] in S and edges[i]['producer'] not in S]
 outputs=[i for i in relevant if edges[i]['producer'] in S and edges[i]['consumer'] not in S]
 for key,idx in [('internal_edges',internal),('external_inputs',inputs),('external_outputs',outputs)]:
  assert [edges[i] for i in idx]==row[key],(row['id'],key)
  del row[key];row[key+'_G1_edge_indices']=idx
data.update(storage='Lossless compact edge references; indices are zero-based indices in graphs/G1.json edges array.',
 original_uncompressed_sha256=original_sha.hexdigest(),original_archive='candidates/cross-module-G1-original.json.gz',
 original_archive_sha256=hashlib.sha256(gz.read_bytes()).hexdigest(),
 G1_sha256=hashlib.sha256((T/'graphs/G1.json').read_bytes()).hexdigest(),
 semantic_warning='Indices recover the same partial graph; they do not prove all consumers, aliases, branches or fusion validity.')
tmp=p.with_suffix('.json.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,separators=(',',':'))+'\n');json.loads(tmp.read_text());os.replace(tmp,p)
doc={'status':'ALL_BOUNDARY_LISTS_RECONSTRUCTED_AND_VERIFIED','date':datetime.datetime.now(datetime.timezone.utc).isoformat(),
     'items':data['count'],'original_uncompressed_sha256':original_sha.hexdigest(),
     'original_archive_sha256':data['original_archive_sha256'],'original_archive_bytes':gz.stat().st_size,
     'compact_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'compact_bytes':p.stat().st_size,
     'source_graph_sha256':data['G1_sha256'],'no_user_or_raw_trace_data_removed':True}
(T/'evidence/crossmodule-compaction-audit.json').write_text(json.dumps(doc,indent=2)+'\n');print(json.dumps(doc,indent=2))
