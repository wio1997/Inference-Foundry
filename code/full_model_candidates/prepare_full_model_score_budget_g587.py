from pathlib import Path
import ast,base64,hashlib,json,shutil,re
src=Path('sg_pd_g542');dst=Path('sg_pd_g587');assert not dst.exists();shutil.copytree(src,dst,ignore=shutil.ignore_patterns('__pycache__'))
for p in dst.rglob('*'):
 if p.is_file() and p.name!='candidate_d_unchanged.json':p.write_text(p.read_text().replace('sg_pd_g542','sg_pd_g587').replace('-g542','-g587').replace('generation=542','generation=587'))
assert (dst/'adapter.py').read_bytes()==(src/'adapter.py').read_bytes()
proof=json.loads(Path('G586_parent_actual_FP4_score_gate.json').read_text());assert proof['all64_score_FP32_int32_bits_equal'] and proof['all64_native_unsorted_topk_indices_equal'] and proof['all8_released']
(dst/'actual_FP4_score_gate_prerequisite.json').write_text(json.dumps(proof,indent=2)+'\n');shutil.copyfile('indexer_score_budget_g587.py',dst/'indexer_score_budget_g587.py')
p=dst/'integration.py';s=p.read_text();needle="    receipt={'pid':os.getpid(),'candidate_backend':'DeepseekV41NPUBackend',";assert s.count(needle)==1;s=s.replace(needle,"    from indexer_score_budget_g587 import install as install_score_budget\n    install_score_budget(root)\n"+needle);p.write_text(s)
for name in ['candidate_p.json']:
 p=dst/name;c=json.loads(p.read_text());prior=json.loads((src/name).read_text());assert c['argv']==prior['argv'] and c['environment']==prior['environment'];c['scope']='G587 P-only native NPU score rowchunk budget128MiB using unchanged code object and isolatedglobalnamespace; fixed native FP4 q/k domain gate64 strictcases passed. G542 Engram dead-reference fix/.88/chunk8192/EP8/DSPARK5/BF16KV and D501 .80 remain. GenericBF16strictfailure preserved; fullmodel/PD/long/cache/quality/performance unverified.';p.write_text(json.dumps(c,indent=2)+'\n')
(dst/'candidate_status.json').write_text(json.dumps({'status':'PREPARED_AFTER_ACTUAL_FP4_DOMAIN_STRICT_GATE','delta':'Native_score_chunks same code object; NPU-only clonedglobalsbudget128MiB, CPUglobaloriginal1GiB unchanged; host-onlyactualchunkpathreceipt','previousP542':'Math and eightEngram-pathhits passed;8191failednative1GiBscoretemporary; rawfailurepreserved/all8released','full_model_recovery':False,'E2E_gain':False,'D':'Keep existingD501 .80/fullgraphs; requirefreshP-Dstate/function/long/cache gates'},indent=2)+'\n')
assert (dst/'candidate_d_unchanged.json').read_bytes()==(src/'candidate_d_unchanged.json').read_bytes()
pins={str(p.relative_to(dst)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dst.rglob('*') if p.is_file() and p.name not in ['launch_pd.py','deployment_source_sha256.json']};(dst/'deployment_source_sha256.json').write_text(json.dumps(pins,indent=2)+'\n')
for p in dst.rglob('*.py'):ast.parse(p.read_text())
files={str(p.relative_to(dst)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'b64':base64.b64encode(p.read_bytes()).decode()} for p in dst.rglob('*') if p.is_file()}
s=Path('start_prefill_lifetime_g542.sh').read_text();ls=s.splitlines();ls=[('files='+repr(files)) if l.startswith('files=') else l for l in ls];s='\n'.join(ls)+'\n';s=re.sub(r"(?<![A-Za-z_])r=root/'sg_pd_g542'","r=root/'sg_pd_g587'",s);s=s.replace("prior=root/'sg_pd_g523'","prior=root/'sg_pd_g542'").replace("s['generation']==523","s['generation']==542")
s=s.replace("assert 'rotated_gate' in (prior/'server.log').read_text() and '642.00 MiB' in (prior/'server.log').read_text()","assert 'indexer.scores(q[tok_c]' in (prior/'server.log').read_text() and '1.00 GiB' in (prior/'server.log').read_text()")
s=s.replace("probe=root/'sg_engram_gate_lifetime_g537'","probe=root/'sg_native_FP4_score_gate_g581'")
s=s.replace("all(c['bitwise_equal'] and 'int16' in c['numeric_tolerance'] for c in a['cases'])","all(c['score_FP32_int32_bit_patterns_equal'] and c['native_unsorted_topk_indices_equal'] and c['inputs_unchanged'] for c in a['cases'])")
s=s.replace('G542_LIFETIME_P_START_JSON','G587_NATIVE_SCORE_P_START_JSON').replace('P-only validated dead-reference lifetime repair at unchanged .88/chunk8192/EP8/DSPARK5','P-only native NPU score128MiB budget afteractualFP4-domain strict64gate; priorEngramfix/.88/chunk8192/EP8/DSPARK5 retained')
assert "prior=root/'sg_pd_g542'" in s and "r=root/'sg_pd_g587'" in s and "probe=root/'sg_native_FP4_score_gate_g581'" in s;Path('start_full_model_score_budget_g587.sh').write_text(s)
print('PreparednativeNPU-onlysame-code scorechunkbudget withstrictFP4gates/priorCPUbudgetpreserved; CLI/env/Dbyteunchanged')
