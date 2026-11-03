"""Review the evidence package, then sync only this task's analysis directory.

No host install, service mutation, model code change, NPU operation, or inference.
"""
from pathlib import Path
from fractions import Fraction
import ast, datetime, hashlib, io, json, shlex, shutil, subprocess, sys, tarfile

T = Path(__file__).resolve().parents[1]
Q = Path('/Users/wio/work/173-sglang-matched-172-benchmark')
SSH = '/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
SOCKET = '/Users/wio/work/169-sglang-deepseek-v41/ssh169.sock'
C = 'wio-vllm-v41-repro169-stage-20261028'
BASE = 'wio-vllm-v41-repro169-20261028'
REMOTE = '/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41'
def load(p):
    return json.loads((T / p).read_text())
def save(p, d):
    (T / p).write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
now = datetime.datetime.now(datetime.timezone.utc).isoformat()

def copy_receipts():
    for p in sorted(Q.glob('operator-ceiling169-v2-*.verified.json')):
        for suffix in ['.verified.json', '.exit-code', '.task.txt', '.local.sh', '.output.txt', '.evidence.txt']:
            src = p.with_name(p.name.removesuffix('.verified.json') + suffix)
            if src.exists():
                shutil.copyfile(src, T / 'zcode' / src.name)
    receipts = []
    for p in sorted((T / 'zcode').glob('*.verified.json')):
        d = json.loads(p.read_text()); assert d['model_verified']
        assert all('deepseek' in s.lower() for s in d['observed_models'])
        e = p.with_name(p.name.removesuffix('.verified.json') + '.exit-code')
        receipts.append({'stage': p.name.removesuffix('.verified.json'), 'session_id': d['session_id'],
                         'model': d['observed_models'], 'exit_code': int(e.read_text()) if e.exists() else None, 'sha256': sha(p)})
    save('evidence/ZCODE-EXECUTION-INDEX.json', {'stages': receipts, 'controller': 'GPT', 'mechanical_executor': 'Local Mac ZCode DeepSeek',
         'failure_retained': ['v2-007 equal-count guard rejected400vs440; v2-008 analyzed shifts without forced equal counts', 'v2-013 audit used outer multimodal config; preserved failure and original script; v2-014 correctly read text_config and passed']})

copy_receipts()
if '--sync-only' not in sys.argv:
    records = load('records.json')
    initial_ids = {s['id'] for s in records['sources']}
    snapshot_ids = {s.get('snapshot_path') for s in records['sources']}
    for p in sorted((T / 'sources').rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts:
            continue
        rel = str(p.relative_to(T))
        if rel not in snapshot_ids:
            sid = 'SRC_' + hashlib.sha256(rel.encode()).hexdigest()[:12]
            records['sources'].append({'id': sid, 'url_or_path': '/' + str(p.relative_to(T / 'sources')),
                'version_commit': load('contract.json')['graph_scope']['source_version'], 'retrieved_at': None,
                'registered_at': now, 'snapshot_path': rel, 'sha256': sha(p), 'chapter_symbol': None, 'supported_claim_ids': []})
    aliases = {
      'CAPTURED_TARGET_MAPPING': 'captured-layer-attribution/source-mapping-validation.json',
      'COUNTER_FIELD_AUDIT': 'counter-field-audit/summary.json',
      'OFFICIAL_NOMINAL_PEAK': 'sources/primary/msprof-analyze/chip_peak_flops.py',
      'QR_CONSUMER_REVIEW': 'evidence/QR-consumers-and-fusion-review.md',
      'RETAINED_16RANK_KERNEL_EXPORTS': 'evidence/TRACE-EXPORT-SHA256.json',
      'profile-device-capacity-info': 'evidence/profile-device-capacity-info.json',
      'chip_peak_flops': 'sources/primary/msprof-analyze/chip_peak_flops.py',
      'existing kernel_details': 'evidence/TRACE-EXPORT-SHA256.json'}
    for sid, rel in aliases.items():
        if sid not in initial_ids:
            records['sources'].append({'id': sid, 'url_or_path': rel, 'snapshot_path': rel,
                'sha256': sha(T / rel), 'registered_at': now, 'retrieved_at': None,
                'version_commit': '6b8f431d4724043f44a39eef000fd5c6600784d8' if sid in ['OFFICIAL_NOMINAL_PEAK', 'chip_peak_flops'] else None,
                'chapter_symbol': 'Conditional nominal INT8 FLOPS formula' if 'PEAK' in sid or sid == 'chip_peak_flops' else None,
                'supported_claim_ids': []})
    for s in records['sources']:
        if s.get('snapshot_path') == 'sources/primary/msprof-analyze/chip_peak_flops.py':
            s['version_commit'] = '6b8f431d4724043f44a39eef000fd5c6600784d8'
            s['url_or_path'] = 'https://github.com/Ascend/msprof-analyze/blob/6b8f431d4724043f44a39eef000fd5c6600784d8/msprof_analyze/cluster_analyse/cluster_kernels_analysis/operator_mfu/chip_peak_flops.py'
    # A source filename hash is integrity evidence, never a proof of the claim.
    for s in records['sources']:
        if s.get('snapshot_path') and s.get('sha256'):
            assert sha(T / s['snapshot_path']) == s['sha256'], s['id']
    save('records.json', records)
    contract = load('contract.json')
    assert sha(T / contract['graph_scope']['original_graph_path']) == contract['graph_scope']['original_graph_sha256']
    f = load('fusion-scan.json')
    assert f['status'] == 'PAUSED_WITH_GAPS' and not f['whole_model_exhaustion_proven']
    assert f['stop_reason']['overall_task_paused'] is False
    for c in records['fusion_candidates']:
        if c.get('candidate_graph_path') and c.get('candidate_graph_sha256'):
            assert sha(T / c['candidate_graph_path']) == c['candidate_graph_sha256']
    a = load('module-analysis.json')
    assert len(a['modules']) == 40 and a['rankings'][0]['lowest_module_ids'] == ['target.layers.0.forward']
    for m in a['modules']:
        x = m['metrics'][0]
        assert abs(float(Fraction(x['numerator']) / Fraction(x['denominator'])) - x['value']) < 1e-12
        assert m['call_count'] == 48
    cap = load('evidence/conditional-nominal-capacity.json')
    P = Fraction(24 * 1800 * 10**6 * 16 * 32 * 16)  # MAC not OPS
    assert P == Fraction(cap['capacity_mac_per_second'])
    config = load('evidence/model-config.json')
    # Config can retain the original model_config wrapper.
    cfg = config.get('text_config', config.get('model_config', config))
    if 'hidden_size' not in cfg:
        cfg = config.get('config', cfg)
    assert cfg['hidden_size'] == 5120 and cfg['num_hidden_layers'] == 40
    F_per_position = 40 * (1280 * 64 * 512 + 7 * 3 * 5120 * 2304)
    F_framework = F_per_position * (262144 - 235904 + 8 * 1023)
    numeric = [r for r in a['performance_comparisons'] if r.get('bound_status') == 'CONDITIONAL_CEILING']
    assert len(numeric) == 7
    for c in numeric:
        result = load(c['calculation_path'])
        inp = load(c['bound_input_path'])
        assert Fraction(result['layers'][0]['time_lower_from_known_terms']['exact_fraction']) == Fraction(c['time_lower_L']['exact_fraction'])
        expected_work = F_framework if c['object_id'] == 'TARGET_SELECTED_INT8_C1' else int(c['object_id'].split('_M')[1]) * 1280 * 64 * 512
        assert c['necessary_MAC'] == expected_work
        rate = 16 * P if c['object_id'] == 'TARGET_SELECTED_INT8_C1' else P
        assert Fraction(c['time_lower_L']['exact_fraction']) == Fraction(expected_work, rate)
        assert Fraction(c['throughput_upper_U']['exact_fraction']) == Fraction(c['effective_work_Q']) / Fraction(c['time_lower_L']['exact_fraction'])
        assert len(result['layers'][-1]['unresolved_term_ids']) == 3
        assert result['layers'][-1]['proof_status'] == 'NOT_CHECKED_BY_CALCULATOR'
    # Same-core-cycle ratios are kept as descriptive report proxies.
    audit = load('counter-field-audit/summary.json')
    assert audit['status'].startswith('PASS_')
    assert all(not r.get('aic_positive_activity_missing_cycles', 0) and not r.get('aiv_positive_activity_missing_cycles', 0) for r in audit['ranks'])
    jsfiles, pyfiles = 0, 0
    for p in T.rglob('*.json'):
        json.loads(p.read_text()); jsfiles += 1
    for p in (T / 'tools').glob('*.py'):
        ast.parse(p.read_text()); pyfiles += 1
    assert (T / 'profile-summary.json').stat().st_size < 32768
    fresh = subprocess.run([SSH, shlex.join(['docker', 'inspect', '--format', '{{json .}}', BASE])], capture_output=True, text=True, timeout=90)
    assert fresh.returncode == 0, fresh.stderr
    di = json.loads(fresh.stdout)
    evidence = {'checked_at': now, 'name': BASE, 'id': di['Id'], 'image': di['Image'],
        'running': di['State']['Running'], 'started_at': di['State']['StartedAt'],
        'scope': 'Read-only docker inspect; no device command or service mutation'}
    assert evidence['id'] == 'b00f4d2018e3d495d876b6e396ba9579e78b47a97111b99c27d6678e31039092'
    assert evidence['running'] and evidence['started_at'] == '2026-10-28T02:04:13.1426007Z'
    assert evidence['image'] == 'sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
    save('evidence/BASELINE-CONTINUITY-CHECK.json', evidence)
    save('evidence/REVIEW-CHECKPOINT.json', {'date': now, 'reviewer': 'GPT primary controller; reviewed script executed by ZCode DeepSeek',
      'status': 'PASS_ARITHMETIC_AND_PACKAGE_INTEGRITY_WITH_EXPLICIT_SCOPE_GAPS', 'json_files_parsed': jsfiles,
      'analysis_python_files_parsed': pyfiles, 'numeric_calculations_checked': len(numeric),
      'checks': ['G0and source hashes unchanged', 'Candidate boundary graph hashes match', 'All40module ratios reconstruct from saved numerators/denominators',
                 'Config-derived dense work and nominal capacity independently recounted', 'Exact lower/upper arithmetic agrees', 'Unknown resource terms remain unresolved',
                 'Baseline original container still running with original starttime'],
      'unproven': ['Physical hardware ceiling', 'Global primitive fusion exhaustion', 'Full-framework module minimum', 'MC2causal wait share and removable time', 'Host replay submission mapping'],
      'not_formal_verification': True})
    s = load('STATE.json')
    s.update(current_phase='P7', checkpoint_status='REVIEWED_PARTIAL_DIAGNOSIS_READY_TO_DELIVER', last_update=now,
             overall_work_paused=False, next_read_only_action='Extend only unresolved evidence if new source/runtime data can resolve it; current diagnosis and bounds remain explicitly partial. Goal tool resume is controlled outside available agent APIs.')
    save('STATE.json', s)

# Refresh the known SSH control connection using a read-only container query.
r = subprocess.run([SSH, shlex.join(['docker', 'inspect', '--format={{.State.Running}}', C])], capture_output=True, text=True, timeout=90)
assert r.returncode == 0 and r.stdout.strip() == 'true', r.stderr
live_names = {'operator-ceiling169-v2-20261029-all-operators-allranks.zcode.json',
              'operator-ceiling169-v2-20261029-all-operators-allranks.output.txt'}
files = [p for p in sorted(T.rglob('*')) if p.is_file() and '__pycache__' not in p.parts
         and p.name != 'CHECKPOINT-MANIFEST.json'
         and not ('--exclude-live-output' in sys.argv and (p.name.endswith('.part') or p.name in live_names))]
manifest = {'created_at': now, 'scope': 'This task analysis directory only; retained raw traces are referenced, not copied.',
            'excluded_live_output': '--exclude-live-output' in sys.argv,
            'files': [{'path': str(p.relative_to(T)), 'size': p.stat().st_size, 'sha256': sha(p)} for p in files]}
save('CHECKPOINT-MANIFEST.json', manifest)
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode='w:gz') as tar:
    for p in files + [T / 'CHECKPOINT-MANIFEST.json']:
        tar.add(p, arcname=str(p.relative_to(T)), recursive=False)
remote_code = '''from pathlib import Path
import hashlib,io,json,sys,tarfile
dest=Path(DEST)
assert str(dest).startswith('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/')
data=sys.stdin.buffer.read()
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
 for m in tar.getmembers():
  p=Path(m.name);assert not p.is_absolute() and '..' not in p.parts and m.isfile()
  out=dest/p
  if out.exists() and (p.parts[0]=='sources' or str(p) in ['graphs/G0.json','graphs/G1.json']):
   assert hashlib.sha256(out.read_bytes()).digest()==hashlib.sha256(tar.extractfile(m).read()).digest(),str(p)
 tar.extractall(dest,filter='data')
manifest=json.loads((dest/'CHECKPOINT-MANIFEST.json').read_text())
for e in manifest['files']:
 p=dest/e['path'];assert p.stat().st_size==e['size'] and hashlib.sha256(p.read_bytes()).hexdigest()==e['sha256'],e['path']
print(json.dumps({'status':'SYNCED_AND_HASH_VERIFIED','remote_task_path':str(dest),'files_verified':len(manifest['files']),
 'manifest_sha256':hashlib.sha256((dest/'CHECKPOINT-MANIFEST.json').read_bytes()).hexdigest(),'raw_traces_modified':False,'target_code_modified':False}))
'''.replace('DEST', repr(REMOTE))
args = ['/usr/bin/ssh', '-F', '/dev/null', '-S', SOCKET, '-o', 'BatchMode=yes', '-p', '2224', '-l',
        'xinyufeng@root@172.16.10.169', 'bastion.aiops.baai.ac.cn', shlex.join(['docker', 'exec', '-i', C, 'python3', '-c', remote_code])]
r = subprocess.run(args, input=buf.getvalue(), capture_output=True, timeout=180)
assert r.returncode == 0, r.stderr.decode()
receipt = json.loads(r.stdout)
receipt['verified_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
receipt['local_task_path'] = str(T)
# Outside the manifested task directory: no self-referential hash cycle.
receipt_path = T.parents[2] / 'CHECKPOINT-SYNC-RECEIPT.json'
receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
