#!/usr/bin/env python3
"""CPU-only saved-artifact audit. Writes only Run517."""
import collections,copy,hashlib,importlib.util,json,pathlib,re,shutil,sys,tempfile
sys.dont_write_bytecode=True
ROOT=pathlib.Path('/data/wio/Inference_Foundry');OUT=ROOT/'evidence/20260927_loop079_identity/run517';FORMAL=ROOT/'evidence/20260924_loop036_metadata/run99'
inputs={};checks=[]
def read(p):
 p=pathlib.Path(p);b=p.read_bytes();inputs[str(p)]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)};return b
def ok(name):checks.append({'name':name,'pass':True})
original=json.loads(read(OUT/'original_inventory.json'))
assert len(original['inputs'])==132
for p,h in original['inputs'].items():assert hashlib.sha256(read(ROOT/p)).hexdigest()==h
ok('all132 original saved input hashes match')
spec=importlib.util.spec_from_file_location('original_inventory',OUT/'original_inventory_script.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);old.ROOT=ROOT;old.FORMAL=FORMAL
assert (json.dumps(old.build(),ensure_ascii=False,indent=2)+'\n').encode()==read(OUT/'original_inventory.json');ok('original inventory rebuild byte identical')
groups={'formal_clients':[],'runtime':[],'other_json':[]};nested={g:collections.defaultdict(set) for g in groups}
def walk(x,paths,path=''):
 if isinstance(x,dict):
  for k,v in x.items():paths[path+'.'+k].add(type(v).__name__);walk(v,paths,path+'.'+k)
 elif isinstance(x,list):
  for v in x:walk(v,paths,path+'[]')
for p in sorted(FORMAL.rglob('*')):
 if not p.is_file():continue
 b=read(p)
 if p.suffix=='.json':
  group='runtime' if p.parent.name=='runtime' else 'formal_clients' if p.name.startswith('bench48_') else 'other_json'
  groups[group].append(str(p));walk(json.loads(b),nested[group])
clients=[json.loads((FORMAL/f'bench48_{i}.json').read_text()) for i in (1,2,3)]
for c in clients:
 assert len(c['requests'])==48 and sorted(q['i'] for q in c['requests'])==list(range(48))
 assert sum(q['output_tokens'] for q in c['requests'])==49152 and all(q['error'] is None and q['output_tokens']==1024 for q in c['requests'])
 assert set().union(*(set(q) for q in c['requests']))==set(original['client_record_fields'])
runtime=[json.loads(p.read_text()) for p in sorted((FORMAL/'runtime').glob('*.json'))]
assert set().union(*(set(q) for q in runtime))==set(original['runtime_row_fields'])
assert all(q['block_table_audit']==[] for q in runtime)
assert set().union(*(set(a) for q in runtime for a in q['dspark_slot_refresh_audit']))=={'cycle','gid','changed','zero_blocks'}
ok('recursive union adds only aggregate and slot-refresh fields, no witness chain')
reqs={q['cohort']:q['req_ids'] for q in runtime if q['rank']==0}
assert len(set(x for a in reqs.values() for x in a))==192
assert all(q['req_ids']==reqs[q['cohort']] for q in runtime)
ok('Runtime has192 own request IDs, same within each all8 cohort; client has no identity key')
launcher=(FORMAL/'launcher.log').read_text();log=pathlib.Path(re.search(r'SERVE_LOG\s*:\s*(\S+)',launcher)[1]);text=read(log).decode(errors='replace')
terms=['request_id','req_id','chatcmpl','target_argmax','sampled_token_ids','raw_token','pre_append','wo_a','projection']
term_counts={term:len(re.findall(re.escape(term),text,re.I)) for term in terms}
assert not any(term_counts.values())
posts=[l for l in text.splitlines() if 'POST /v1/chat/completions' in l]
assert len(posts)==192 and all('200 OK' in l for l in posts)
ok('linked historical server log has192 HTTP200 posts but no missing witness identifiers')
for rel in ['scripts/bench.py','scripts/run_loop036_static_e2e.sh','scripts/loop044_run99_e2e_accounting.py','evidence/20260927_loop079_identity/run495/resource_first_position_design.md','evidence/20260927_loop079_identity/run497/findings.md','evidence/20260927_loop079_identity/run497/first_position_cpu.json','evidence/20260927_loop079_identity/run421/analysis.json','evidence/20260927_loop079_identity/run427/analysis.json','evidence/20260927_loop079_identity/run509/astra_resource_certificate_review.md']:
 read(ROOT/rel)
# Demonstrate original scanner weakness without touching originals.
negative=[]
with tempfile.TemporaryDirectory(prefix='SYNTHETIC_ONLY_',dir=OUT) as tmp:
 tmp=pathlib.Path(tmp);shutil.copytree(FORMAL,tmp/'formal');old.FORMAL=tmp/'formal'
 f=old.FORMAL/'bench48_1.json';before=f.read_bytes();a=json.loads(before);a['requests'][0]['request_id']='SYNTHETIC_ONE_ROW';f.write_text(json.dumps(a))
 x=old.build();assert x['witness_fields_present']['client_request_id'] is False
 negative.append({'case':'one client row contains request_id','old_scanner_misses':True,'injected_value':'SYNTHETIC_ONE_ROW'});f.write_bytes(before)
 f=old.FORMAL/'runtime/rank0_cohort1.json';before=f.read_bytes();a=json.loads(before);a['witness']={'target_generation':'SYNTHETIC_GENERATION','first_target_token':123,'pre_append_G':5};f.write_text(json.dumps(a))
 x=old.build();assert not any(x['witness_fields_present'].values())
 negative.append({'case':'one Runtime row has nested generation/token/G witness','old_scanner_misses':True,'injected_fields':a['witness']});f.write_bytes(before)
 for i in (1,2,3):
  f=old.FORMAL/f'bench48_{i}.json';a=json.loads(f.read_text())
  for q in a['requests']:q['request_id']='SYNTHETIC_ALL_ROWS'
  f.write_text(json.dumps(a))
 try:old.build()
 except ValueError:negative.append({'case':'every client row has request_id','old_scanner_rejects':True})
 else:raise AssertionError('old scanner positive-name control did not reject')
old.FORMAL=FORMAL
ok('old sparse/nested scanner false-negatives reproduced and all-row positive control rejects')
for p,h in inputs.items():assert hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h['sha256']
result={'verdict':'PASS_FROZEN_ARTIFACT_CONCLUSION_WITH_SCANNER_LIMITATION','checks':checks,'input_count':len(inputs),'scanned_run99_files':sum(1 for p in FORMAL.rglob('*') if p.is_file()),'json_groups':{g:len(v) for g,v in groups.items()},'recursive_path_union':{g:{p:sorted(t) for p,t in sorted(v.items())} for g,v in nested.items()},'server_log':{'path':str(log),'post_count':len(posts),'candidate_term_counts':term_counts},'original_scanner_negative_controls':negative,'formal_W_minus_certified':False,'scope':'inspected saved files only; not universal impossibility'}
(OUT/'saved_artifact_audit.json').write_text(json.dumps(result,indent=2)+'\n');(OUT/'input_sha256_manifest.json').write_text(json.dumps(inputs,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('recursive_path_union','server_log')},indent=2))
