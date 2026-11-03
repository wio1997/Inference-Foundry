from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72')
def pinned(p,sha):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
inv=pinned(P/'analysis/profile-capabilities-index-v12-v2.json','79ba556ef3855b9066392e438371a0a9d3b38ea7511269ce7ff2a6814529a095')
origin=pinned(P/'analysis/prepare-field-origins-v12.json','c23b75f3fbd3ae3d3d7152f6882524e899bade53001ea2f1befa09ce943f091f')
index=pinned(P/'analysis/native-execution-identity-v14/index.json','a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4')
OUT=P/'analysis/native-execution-identity-v14/graph-sidecar-envelope-candidates-v14.json';assert not OUT.exists()
pairs=defaultdict(dict)
for f in inv['metadata']:
 for e in f['entries']:
  if e['event']in ['execute_enter','execute_exit']:
   key=(e['pid'],e['tid'],e['label']);assert e['event']not in pairs[key];pairs[key][e['event']]=e
complete={k:v for k,v in pairs.items()if set(v)=={'execute_enter','execute_exit'}}
bythread=defaultdict(list)
for (pid,tid,label),p in complete.items():bythread[(pid<<32)|tid].append(p)
validation=[]
for b in origin['bindings']:
 m=b['marker'];e=b['event'];p=complete.get((e['pid'],e['tid'],e['label']))
 validation.append(dict(database_index=b['database_index'],marker_API_rowid=m['api_rowid'],label=e['label'],paired=p is not None,epoch_envelope_contains_marker=p is not None and p['execute_enter']['epoch_ns']<=m['startNs']<=m['endNs']<=p['execute_exit']['epoch_ns'],epoch_minus_monotonic_enter_ns=e['epoch_ns']-e['monotonic_ns']))
rows=[]
for ref in index['databases']:
 d=pinned(Path(ref['path']),ref['sha256'])
 for gi,g in enumerate(d['graph_API_control_bindings']):
  a=g['CANN_API'];cs=[p for p in bythread[a['globalTid']]if p['execute_enter']['epoch_ns']<=a['startNs']<=a['endNs']<=p['execute_exit']['epoch_ns']]
  named={b['event']['label']for b in g['same_thread_markers']}
  rows.append(dict(database_index=d['database_index'],graph_ordinal=gi,CANN_API=a,existing_marker_labels=sorted(named),sidecar_envelope_candidates=cs,unique_candidate=len(cs)==1,agrees_with_existing_marker=bool(named)and len(cs)==1 and cs[0]['execute_enter']['label']in named))
info=inv['profiler_infos'][0];rawinfo=json.loads(Path(info['path']).read_text())
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),paired_sidecar_calls=len(complete),metadata_pair_total=len(pairs),validation_markers=len(validation),markers_inside_same_call_epoch_envelope=sum(v['epoch_envelope_contains_marker']for v in validation),graph_records=len(rows),candidate_count_hist=dict(Counter(len(r['sidecar_envelope_candidates'])for r in rows)),existing_marker_agreement=sum(r['agrees_with_existing_marker']for r in rows),previously_unbound_with_unique_candidate=sum(not r['existing_marker_labels']and r['unique_candidate']for r in rows),no_source_DB_or_device_work=True,clock_semantics_closed=False,overall_complete=False,
 limits=['These are same-host epoch envelope candidates, not promoted phase bindings until clock/export conversion is reviewed. No cross-rank timestamp join or nearest-time fallback.','Existing exact named-marker bindings remain stronger and are retained independently.'])
payload=(json.dumps(dict(summary=summary,marker_epoch_validation=validation,graph_candidates=rows,first_profiler_info=dict(path=info['path'],raw=rawinfo)),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
print(json.dumps(dict(summary=summary,full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest(),first_profiler_info=rawinfo,first_previously_unbound=[{k:v for k,v in r.items()if k!='sidecar_envelope_candidates'}for r in rows if not r['existing_marker_labels']][:1]),ensure_ascii=False))
