from pathlib import Path
import ast,base64,hashlib,json,shutil,re
src=Path('sg_pd_g523');dst=Path('sg_pd_g542');assert not dst.exists();shutil.copytree(src,dst,ignore=shutil.ignore_patterns('__pycache__'))
h=lambda b:hashlib.sha256(b).hexdigest()
proof=json.loads(Path('G540_parent_strict_engram_acceptance.json').read_text());assert proof['all8ranks_64cases_BF16_integer_bit_patterns_equal'] and proof['all8_released']
for p in dst.rglob('*'):
 if p.is_file() and p.name!='candidate_d_unchanged.json':
  s=p.read_text().replace('sg_pd_g523','sg_pd_g542').replace('-g523','-g542').replace('generation=523','generation=542');p.write_text(s)
p=dst/'adapter.py';s=p.read_text()
assert h(src.joinpath('adapter.py').read_bytes())==json.loads(Path('sg_engram_gate_lifetime_g537/mechanism.json').read_text())['original_adapter_sha256']
needle='    gate=torch.sigmoid(torch.where(negative,-magnitude,magnitude))\n';assert s.count(needle)==1
s=s.replace(needle,needle+'    del key, original\n    _record_lifetime_large_path(x)\n')
receipt='''

_LIFETIME_LARGE_PATH_RECORDED = False

def _record_lifetime_large_path(x):
    """One host-only path receipt per actual worker; no tensor operations."""
    global _LIFETIME_LARGE_PATH_RECORDED
    if _LIFETIME_LARGE_PATH_RECORDED or x.shape[0] < 8191:
        return
    import os, datetime
    directory=Path(__file__).resolve().parent/'gate_lifetime_hits'
    directory.mkdir(exist_ok=True)
    evidence={'pid':os.getpid(),'device_index':x.device.index,'rows':x.shape[0],
              'shape':list(x.shape),'dtype':str(x.dtype),
              'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'adapter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'path':'actual rotated_gate after del key,original and before final unchanged output expression',
              'scope':'Executed model path only; not output correctness or request completion'}
    (directory/f'worker{os.getpid()}.json').write_text(json.dumps(evidence,indent=2)+'\\n')
    _LIFETIME_LARGE_PATH_RECORDED=True
'''
s=s.replace('\n\ndef install(model_root):',receipt+'\n\ndef install(model_root):');p.write_text(s);ast.parse(s)
fn=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='rotated_gate')
fn.body=[n for n in fn.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='_record_lifetime_large_path')]
ref=next(n for n in ast.parse(Path('sg_engram_gate_lifetime_g537/candidate_gate.py').read_text()).body if isinstance(n,ast.FunctionDef))
assert ast.dump(fn,include_attributes=False)==ast.dump(ref,include_attributes=False),'Only host receipt outside tested candidate arithmetic'
c=json.loads((dst/'candidate_p.json').read_text());old=json.loads((src/'candidate_p.json').read_text());assert c['argv']==old['argv'] and c['environment']==old['environment']
c['scope']='G542 P-only .88 unchanged from G523; validated arithmetic-identical Engram del key,original lifetime repair, plus one host-only large-row path receipt per worker. D501 .80 unchanged. Full PD long-shape/cache/function gates pending; no E2E performance claim.';(dst/'candidate_p.json').write_text(json.dumps(c,indent=2)+'\n')
assert (dst/'candidate_d_unchanged.json').read_bytes()==(src/'candidate_d_unchanged.json').read_bytes()
proof['actual_candidate_function_AST_verified_against_strict_probe']=True;proof['actual_adapter_sha256']=h(p.read_bytes());(dst/'engram_lifetime_prerequisite.json').write_text(json.dumps(proof,indent=2)+'\n')
(dst/'candidate_status.json').write_text(json.dumps({'status':'PREPARED_AFTER_64_STRICT_BITWISE_CASES','D':'Existing D501 .80 retained, native new-peer and actual request gates required','delta':'del key,original only in arithmetic AST; host-only firstlarge path receipt','benchmark_started':False,'full_model_recovery_accepted':False},indent=2)+'\n')
pins={str(p.relative_to(dst)):h(p.read_bytes()) for p in dst.rglob('*') if p.is_file() and p.name not in ['launch_pd.py','deployment_source_sha256.json']}
(dst/'deployment_source_sha256.json').write_text(json.dumps(pins,indent=2)+'\n')
p=dst/'launch_pd.py';s=p.read_text();needle="    candidate=json.loads((ROOT/'candidate.json').read_text())";assert s.count(needle)==1;s=s.replace(needle,"    for name,expected in json.loads((ROOT/'deployment_source_sha256.json').read_text()).items():\n        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected, name\n"+needle);p.write_text(s);ast.parse(s)
files={str(p.relative_to(dst)):{'sha256':h(p.read_bytes()),'b64':base64.b64encode(p.read_bytes()).decode()} for p in dst.rglob('*') if p.is_file()}
body='''from pathlib import Path
import subprocess,json,socket,base64,hashlib,time,datetime,shutil
assert socket.gethostname()=='<ROLE_HOSTNAME>'
mounts={p:shutil.disk_usage(p).free for p in ['/data','/data1']};assert max(mounts,key=mounts.get)=='/data'
root=Path('/PRIVATE_TASK_ROOT');prior=root/'sg_pd_g523';r=root/'sg_pd_g542';assert prior!=r and not r.exists()
s=json.loads((prior/'state.json').read_text());assert s['generation']==523 and s['phase']=='failed_released' and s['cleanup']['all8_process_lists_clear'];p=Path('/proc',str(s['pid']),'cmdline');assert not p.exists() or str(prior/'node_controller.py').encode() not in p.read_bytes()
assert 'rotated_gate' in (prior/'server.log').read_text() and '642.00 MiB' in (prior/'server.log').read_text()
probe=root/'sg_engram_gate_lifetime_g537';s=json.loads((probe/'state.json').read_text());assert s['phase']=='stopped_released' and s['failure'] is None and s['cleanup']['all8_process_lists_clear']
for rank in range(8):
 a=json.loads((probe/'evidence'/f'rank{rank}.json').read_text());assert a['phase']=='passed' and a['rank']==rank and len(a['cases'])==8 and all(c['bitwise_equal'] and 'int16' in c['numeric_tolerance'] for c in a['cases'])
import urllib.request
with urllib.request.urlopen('http://<ROLE_HOST>:19094/server_info',timeout=5) as reply:peer=json.load(reply)
args=peer.get('server_args',peer);assert args['disaggregation_mode']=='decode' and args['tp_size']==args['ep_size']==8 and args['speculative_algorithm']=='DSPARK' and args['disaggregation_decode_enable_radix_cache'] is True
assert args['mem_fraction_static']==0.80 and args['cuda_graph_config']['decode']['backend']=='full' and peer['max_total_num_tokens']>=263296
r.mkdir()
for name,e in files.items():
 raw=base64.b64decode(e['b64']);assert hashlib.sha256(raw).hexdigest()==e['sha256'];p=r/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
(r/'candidate.json').write_bytes((r/'candidate_p.json').read_bytes());(r/'storage_preflight.json').write_text(json.dumps(mounts,indent=2))
with (r/'controller.log').open('x') as f:p=subprocess.Popen(['python3','-u',str(r/'node_controller.py')],cwd=r,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(2);h={'pid':p.pid,'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'bound_s':72000,'source_sha256':hashlib.sha256((r/'node_controller.py').read_bytes()).hexdigest(),'scope':'P-only validated dead-reference lifetime repair at unchanged .88/chunk8192/EP8/DSPARK5; D501 .80 retained. Full model/function/length/cache gates pending.'};(r/'actual_launch_handle.json').write_text(json.dumps(h,indent=2));print('G542_LIFETIME_P_START_JSON '+json.dumps({'handle':h,'actual_live':p.poll() is None,'state':json.loads((r/'state.json').read_text()) if (r/'state.json').exists() else None}),flush=True)
'''
Path('start_prefill_lifetime_g542.sh').write_text("python3 - <<'REMOTE_CODE'\nfiles="+repr(files)+'\n'+body+'REMOTE_CODE\n')
print('Prepared pinned P-only Engram liveness candidate; unchanged CLI/env/D501; host pathreceipt isolated from tested arithmetic AST')
