"""Evidence-specific controller audit and safe checkpoint sync."""
from pathlib import Path
from decimal import Decimal
from fractions import Fraction
import datetime,gzip,hashlib,json,subprocess,sys
T=Path(__file__).resolve().parents[1];now=datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(n):return json.loads((T/n).read_text())
def write(n,v):(T/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

records=read('records.json');ids={s['id'] for s in records['sources']}
for h in read('evidence/dspark-runner-source-scan.json')['hits']:
 sid='SRC_STAGE_'+hashlib.sha256(h['path'].encode()).hexdigest()[:12]
 if sid not in ids:
  records['sources'].append({'id':sid,'url_or_path':h['path'],'snapshot_path':h['snapshot_path'],'sha256':h['sha256'],
   'registered_at':now,'version_commit':'Image sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b; stage source snapshot',
   'scope':'Matches baseline for five files in dspark-source-provenance.json; other files are stage-image snapshots only.'})
write('records.json',records)
for s in records['sources']:
 if s.get('snapshot_path') and s.get('sha256'):
  p=T/s['snapshot_path'];assert p.is_file() and sha(p)==s['sha256'],s['id']

g1=read('graphs/G1.json');compact=read('candidates/cross-module-G1.json');compaudit=read('evidence/crossmodule-compaction-audit.json')
assert compact['count']==len(compact['items'])==compaudit['items']==22802
assert compact['G1_sha256']==sha(T/'graphs/G1.json')==compaudit['source_graph_sha256']
assert sha(T/'candidates/cross-module-G1-original.json.gz')==compaudit['original_archive_sha256']
h=hashlib.sha256()
with gzip.open(T/'candidates/cross-module-G1-original.json.gz','rb') as f:
 for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
assert h.hexdigest()==compaudit['original_uncompressed_sha256']
for j in [0,len(compact['items'])//2,len(compact['items'])-1]:
 x=compact['items'][j];nodes=set(x['nodes'])
 for kind in ['internal_edges','external_inputs','external_outputs']:
  for i in x[kind+'_G1_edge_indices']:
   e=g1['edges'][i];a=e['producer'] in nodes;b=e['consumer'] in nodes
   assert (a and b) if kind=='internal_edges' else ((not a and b) if kind=='external_inputs' else (a and not b))
assert read('fusion-scan.json')['status']=='PAUSED_WITH_GAPS'
assert not read('fusion-scan.json')['whole_model_exhaustion_proven']
for fid,a,b in [('F03','ffn_post','next_pre'),('F23','merge','returned_moe_output'),('F29','scatter','persistent_swa_state'),('F27','main_norm','context_buffer')]:
 c=next(x for x in records['fusion_candidates'] if x['id']==fid);assert sha(T/c['candidate_graph_path'])==c['candidate_graph_sha256']
 g=read(c['candidate_graph_path']);edges=g['internal_edges']+g['external_inputs']+g['external_outputs']
 assert any(e['producer']==a and e['consumer']==b for e in edges),fid
for fid in ['F01','F04','F13']:
 c=next(x for x in records['fusion_candidates'] if x['id']==fid);assert c['implemented'] and not c['new_fusion_opportunity']

a=read('module-analysis.json');numeric=[x for x in a['performance_comparisons'] if x.get('bound_status')=='CONDITIONAL_CEILING'];assert len(numeric)==7
for c in numeric:
 assert 'INT8 Cube' in c['implementation_domain']
 calc=read(c['calculation_path']);assert Fraction(calc['layers'][0]['time_lower_from_known_terms']['exact_fraction'])==Fraction(c['time_lower_L']['exact_fraction'])
 assert Fraction(c['throughput_upper_U']['exact_fraction'])==Fraction(c['effective_work_Q'])/Fraction(c['time_lower_L']['exact_fraction'])
 assert c['scope_match_status']=='CONDITIONAL_DOMAIN_MEMBERSHIP_NOT_FULLY_ESTABLISHED_FOR_OBSERVED_RUN'
 if c['object_id']=='TARGET_SELECTED_INT8_C1':
  assert 'without extra cross-request/position reuse' in c['implementation_domain']
  assert c['wide_sharing_domain_status']=='INCOMPLETE_NO_NUMERIC_UPPER_BOUND'
wide=next(x for x in a['performance_comparisons'] if x['object_id']=='TARGET_FULL_SEMANTICS_SHARING_MIXED_RESOURCE_DOMAIN')
assert wide['throughput_upper_U'] is None and wide['bound_status']=='INCOMPLETE'

strict=read('eager-draft-mainstream-attribution-v2/draft_call_metrics.json')
sen=read('eager-draft-mainstream-attribution-v5/draft_call_metrics.json')
key=lambda x:(x['rank'],x['start_ns'],x['end_ns'],x['main_stream_id'])
assert len(strict)==162 and len(sen)==288 and len({key(x) for x in strict})==162 and len({key(x) for x in sen})==288
assert {key(x) for x in strict}=={key(x) for x in sen if x['attention_validation']=='ONE_SPARSE_FLASH_MLA'}
assert sum(x['attention_validation']=='EMPTY_LOCAL_QUERY_SOURCE_CONSISTENT_NOT_DIRECTLY_OBSERVED' for x in sen)==126
diagnosis=read('evidence/draft-module-diagnosis.json')
assert diagnosis['strict']['rank_count']==15 and diagnosis['strict']['excluded_calls']==126
assert diagnosis['conditional_sensitivity']['empty_query_status'].startswith('HYPOTHESIS')
assert diagnosis['strict']['draft_pooled_3block']['ratio'] < diagnosis['strict']['target_layer0_matching_15ranks']['ratio']
assert all(x.get('global_full_framework_minimum_known') is False for x in a['rankings'])
assert len([x for x in a['modules'] if x['module_id'].startswith('target.layers.')])==40
assert len([x for x in a['modules'] if x['module_id']=='draft.decoder_block.pooled'])==1
assert (T/'profile-summary.json').stat().st_size<32768

zipfile=Path('/Users/wio/work/operator-ceiling-agent-kit-v2.zip')
zip_hash=sha(zipfile) if zipfile.is_file() else None
state=read('STATE.json');state.update(current_phase='P6',last_update=now,overall_work_paused=False,
 checkpoint_status='REVIEWED_PARTIAL_WITH_CORRECTED_SCOPES_AND_DRAFT_SENSITIVITY',
 next_read_only_action='Investigate remaining whole-framework attribution, especially exact empty-query metadata and eager prefill/Engram; do not infer global minimum from pooled draft samples.')
write('STATE.json',state)
review={'date':now,'status':'REVIEWED_PARTIAL_CORRECTED_SCOPE','reviewer':'GPT primary controller',
 'checks':['All registered source hashes match', 'Archived original and all 22802 compact graph boundary references verified by compaction audit; representative edge indices rechecked',
           'F03,F23,F29,F27 source corrections exist and candidate hashes match',
           'Seven exact numeric ceilings explicitly restrict INT8 resource pool and C8 additionally excludes cross-position reuse',
           'Wide sharing/mixed-resource implementation domain has no unsupported numeric ceiling',
           'Strict162 draft brackets are exactly the SparseFlashMla subset of288 conditional brackets',
           'Pooled draft and target comparisons use identical15-rank subset for strict sample',
           'All reported rankings state that full-framework minimum remains unknown'],
 'unresolved':['Complete primitive graph/consumer and alias coverage','Prefill/Engram/root/head attribution',
               'Strict draft attribution for126 local-query-unobserved calls and individual draft layer IDs',
               'Cross-rank MC2 operation IDs, precise clock error, causal wait share and end-to-end exposure',
               'Hardware physical capacity upper bound and general-domain necessary distinct work'],
 'reviewed_zip_sha256':zip_hash,'expected_frozen_zip_sha256':'9a11fd3494533b930cf68028bcc5ce13bb3388646c7d33f14e39b760f4c1da21',
 'zip_matches_frozen':zip_hash=='9a11fd3494533b930cf68028bcc5ce13bb3388646c7d33f14e39b760f4c1da21',
 'formal_proof_claimed':False,'target_implementation_modified':False,'new_device_work':False}
write('evidence/REVIEW-PARTIAL-AFTER-DRAFT.json',review)
r=subprocess.run([sys.executable,str(T/'tools/audit_and_sync_checkpoint.py'),'--sync-only'],capture_output=True,text=True,timeout=300)
assert r.returncode==0,r.stderr
print(json.dumps({'review':review['status'],'zip_matches_frozen':review['zip_matches_frozen'],
 'strict_draft_proxy':diagnosis['strict']['draft_pooled_3block']['ratio'],
 'wide_domain_numeric_ceiling':'UNKNOWN','sync_receipt':json.loads(r.stdout)},indent=2))
