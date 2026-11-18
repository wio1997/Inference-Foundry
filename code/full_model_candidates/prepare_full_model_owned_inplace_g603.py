from pathlib import Path
import ast,base64,copy,datetime,hashlib,json,re,shutil
obs=json.loads(Path('G601_Engram_owned_inplace_actual_2203_parsed.json').read_text())
assert not obs['actual_live'] and obs['state']['phase']=='stopped_released' and obs['state']['failure'] is None and obs['state']['cleanup']['all8_process_lists_clear']
ranks=list(obs['ranks'].values());assert {r['rank'] for r in ranks}==set(range(8))
assert all(r['phase']=='passed' and len(r['cases'])==14 and all(c['literal_BF16_int16_bits_equal'] and c['inputs_and_weights_unchanged'] and not c['row_partition_changed'] for c in r['cases']) for r in ranks)
for r in ranks:
 for c in r['cases']:
  if c['rows'] in [8191,8193] and c['kv_dtype']=='torch.bfloat16':
   p=c['peak_delta_bytes'];assert p['candidate']<2.3*1024**3 and p['reference']-p['candidate']>=.8*c['rows']*4*5120*4
proof={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_observer':'G601_Engram_owned_inplace_actual_2203_parsed.json','raw_directory':'G601_Engram_owned_inplace_evidence','cases':112,'ranks':8,'all112_BF16_int16_bit_patterns_equal':True,'all112_inputs_and_weights_unchanged':True,'all8_released':True,'candidate_sha256':hashlib.sha256(Path('sg_engram_owned_inplace_g600/candidate_gate.py').read_bytes()).hexdigest(),'scope':'Selected eager Engram tensors only; not full-model/PD/quality/performance acceptance'}
Path('G602_parent_owned_inplace_acceptance.json').write_text(json.dumps(proof,indent=2)+'\n')
src=Path('sg_pd_g587');dst=Path('sg_pd_g603');assert not dst.exists();shutil.copytree(src,dst,ignore=shutil.ignore_patterns('__pycache__'))
for p in dst.rglob('*'):
 if p.is_file() and p.name!='candidate_d_unchanged.json':p.write_text(p.read_text().replace('sg_pd_g587','sg_pd_g603').replace('-g587','-g603').replace('generation=587','generation=603'))
p=dst/'adapter.py';s=p.read_text();old='    dot=(original*(q_weight.float()*k_weight.float())*key).sum(-1)*rstd*dim**-0.5';new='    original.mul_(q_weight.float()*k_weight.float())\n    original.mul_(key)\n    _record_owned_inplace_large_path(x,kv,original)\n    dot=original.sum(-1)*rstd*dim**-0.5';assert s.count(old)==1;s=s.replace(old,new)
helper='''
_OWNED_INPLACE_LARGE_PATH_RECORDED = False

def _record_owned_inplace_large_path(x,kv,original):
    """Host metadata only, after two mutations of the fresh matmul output."""
    global _OWNED_INPLACE_LARGE_PATH_RECORDED
    if _OWNED_INPLACE_LARGE_PATH_RECORDED or x.shape[0] < 8191:
        return
    import os, datetime
    directory=Path(__file__).resolve().parent/'gate_owned_inplace_hits'
    directory.mkdir(exist_ok=True)
    evidence={'pid':os.getpid(),'device_index':x.device.index,'rows':x.shape[0],
              'x_shape':list(x.shape),'x_dtype':str(x.dtype),'x_stride':list(x.stride()),
              'kv_shape':list(kv.shape),'kv_dtype':str(kv.dtype),'kv_stride':list(kv.stride()),
              'original_shape':list(original.shape),'original_dtype':str(original.dtype),
              'original_stride':list(original.stride()),'owned_buffer_products':2,
              'row_partition_changed':False,'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'adapter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope':'Executed model path only; not output correctness/request completion'}
    (directory/f'worker{os.getpid()}.json').write_text(json.dumps(evidence,indent=2)+'\\n')
    _OWNED_INPLACE_LARGE_PATH_RECORDED=True

'''
s=s.replace('\ndef install(model_root):',helper+'\ndef install(model_root):');p.write_text(s)
# Numeric AST equals the exact component candidate after removing host receipts.
def numeric_fn(text):
 fn=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='rotated_gate')
 fn.body=[n for n in fn.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id.startswith('_record_'))]
 return ast.dump(fn,include_attributes=False)
assert numeric_fn(s)==numeric_fn(Path('sg_engram_owned_inplace_g600/candidate_gate.py').read_text())
assert (dst/'candidate_d_unchanged.json').read_bytes()==(src/'candidate_d_unchanged.json').read_bytes()
for name in ['candidate_p.json']:
 p=dst/name;c=json.loads(p.read_text());prior=json.loads((src/name).read_text());assert c['argv']==prior['argv'] and c['environment']==prior['environment'];c['scope']='G603 P-only owned original FP32 in-place dot products, G600 strict112-case gate passed. Preserve native128MiB score budget/deadref/.88/chunk8192/EP8/DSPARK5 and D501. Full model/function/quality/perf UNVERIFIED.';p.write_text(json.dumps(c,indent=2)+'\n')
(dst/'owned_inplace_prerequisite.json').write_text(json.dumps(proof,indent=2)+'\n')
(dst/'candidate_status.json').write_text(json.dumps({'status':'PREPARED_AFTER_ACTUAL112_ENGRAM_GATE','delta':'Fresh owned original FP32 buffer reused for two products after rstd; exact component numeric AST, host-only dtype/stride path receipt','P587_failure':'Engram dot642MiB OOM/all8released; prior score128MiB path8ranks observed','full_model_recovery':False,'E2E_gain':False,'D':'Retain actual D501; fresh long/PD/cache gates required'},indent=2)+'\n')
for p in dst.rglob('*.py'):ast.parse(p.read_text())
pins={str(p.relative_to(dst)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dst.rglob('*') if p.is_file() and p.name not in ['launch_pd.py','deployment_source_sha256.json']};(dst/'deployment_source_sha256.json').write_text(json.dumps(pins,indent=2)+'\n')
files={str(p.relative_to(dst)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'b64':base64.b64encode(p.read_bytes()).decode()} for p in dst.rglob('*') if p.is_file()}
s=Path('start_full_model_score_budget_g587.sh').read_text();s='\n'.join('files='+repr(files) if l.startswith('files=') else l for l in s.splitlines())+'\n'
s=re.sub(r"(?<![A-Za-z_])r=root/'sg_pd_g587'","r=root/'sg_pd_g603'",s).replace("prior=root/'sg_pd_g542'","prior=root/'sg_pd_g587'").replace("s['generation']==542","s['generation']==587")
s=s.replace("assert 'indexer.scores(q[tok_c]' in (prior/'server.log').read_text() and '1.00 GiB' in (prior/'server.log').read_text()","assert 'original*(q_weight.float()*k_weight.float())*key' in (prior/'server.log').read_text() and '642.00 MiB' in (prior/'server.log').read_text()")
s=s.replace("probe=root/'sg_native_FP4_score_gate_g581'","probe=root/'sg_engram_owned_inplace_g600'").replace("len(a['cases'])==8 and all(c['score_FP32_int32_bit_patterns_equal'] and c['native_unsorted_topk_indices_equal'] and c['inputs_unchanged'] for c in a['cases'])","len(a['cases'])==14 and all(c['literal_BF16_int16_bits_equal'] and c['inputs_and_weights_unchanged'] for c in a['cases'])")
s=s.replace("r.mkdir()","with urllib.request.urlopen('http://<ROLE_HOST>:19094/ready',timeout=5) as reply:assert reply.status==200\nr.mkdir()")
s=s.replace('G587_NATIVE_SCORE_P_START_JSON','G603_OWNED_INPLACE_P_START_JSON').replace('P-only native NPU score128MiB budget afteractualFP4-domain strict64gate; priorEngramfix/.88/chunk8192/EP8/DSPARK5 retained','P-only ownedFP32 Engram products after strict112cases; priornative128MiBscore/.88/chunk8192/EP8/DSPARK5 retained')
assert "prior=root/'sg_pd_g587'" in s and "r=root/'sg_pd_g603'" in s;Path('start_full_model_owned_inplace_g603.sh').write_text(s)
s=Path('functional_PD_P587_D501_g588.py').read_text().replace('sg_pd_g587','sg_pd_g603').replace('P587','P603').replace('g588','g604').replace("s['generation']==587","s['generation']==603")
needle="    state('functional_cases_passed_requires_PD_and_effective_graph_evidence_review')"
extra="    hits=list((RUN/'gate_owned_inplace_hits').glob('*.json')); records=[json.loads(p.read_text()) for p in hits]; assert {x['device_index'] for x in records}==set(range(8)); assert all(x['owned_buffer_products']==2 and not x['row_partition_changed'] for x in records); save('engram_owned_inplace_path_hits.json',records)\n"
assert needle in s;s=s.replace(needle,extra+needle);ast.parse(s);Path('functional_PD_P603_D501_g604.py').write_text(s)
launcher=Path('start_functional_PD_P587_D501_g588.sh')
if not launcher.exists():
 matches=[p for p in Path('.').glob('*588*.sh') if 'digest=' in p.read_text() and 'functional_actual_handle' in p.read_text()];assert len(matches)==1,matches;launcher=matches[0]
raw=s.encode();s=launcher.read_text();s='\n'.join("raw=base64.b64decode("+repr(base64.b64encode(raw).decode())+")" if l.startswith('raw=base64.b64decode(') else "digest="+repr(hashlib.sha256(raw).hexdigest()) if l.startswith('digest=') else l for l in s.splitlines())+'\n';s=s.replace('sg_pd_g587','sg_pd_g603').replace('P587','P603').replace('g588','g604').replace("s['generation']==587","s['generation']==603").replace('G526_FUNCTIONAL_START_JSON','G604_FUNCTIONAL_START_JSON');Path('start_functional_PD_g604.sh').write_text(s)
print(json.dumps({'component_cases':112,'exact_component_candidate_AST':True,'D_unchanged':True,'CLI_env_unchanged':True,'P_candidate':'sg_pd_g603','functional_launcher_source':str(launcher)}))
