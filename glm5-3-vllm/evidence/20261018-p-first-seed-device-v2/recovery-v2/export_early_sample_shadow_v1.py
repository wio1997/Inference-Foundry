"""Preserve both stopped all16 sources; export only each rank0 on CPU."""
from pathlib import Path
import hashlib, json, shutil, socket, subprocess, tarfile, time

assert socket.gethostname() == 'B5-5U-Atlas900RCKA3-33'
root = Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/early-copy-shadow-live1-20261019/node33')
gate = json.loads((root/'early-sample-shadow-c4-supply-profile-v1/result.json').read_text())
assert gate['completed'] and gate['shadow_restored_off'] and gate['observer_restored_off']
assert len(gate['profiles']) == 1 and all(z['profile_stopped'] and z['ranks'] == list(range(16)) for z in gate['profiles'])
assert all(z['numeric_pass'] and z['budget_pass'] for z in gate['rows'])
out = Path(__file__).resolve().parent/'artifacts'; out.mkdir(exist_ok=False)
shutil.copyfile(root/'early-sample-shadow-c4-supply-profile-v1/result.json', out/'profile-result.json')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): h.update(chunk)
    return h.hexdigest()

results = []
for record in gate['profiles']:
    mode = record['mode']; assert mode in ['adaptive']
    dirs = [Path(p) for p in record['dirs']]
    assert all(p.parent == root/'profile' and not p.is_symlink() for p in dirs)
    files = sorted(p for directory in dirs for p in directory.rglob('*') if p.is_file())
    assert all(not p.is_symlink() for p in files) and sum(p.stat().st_size for p in files) < 4*1024**3
    expected = {z['path']: z['bytes'] for z in record['files']}
    assert {str(p.relative_to(root)): p.stat().st_size for p in files} == expected
    snapshot = root/('early-sample-shadow-c4-supply-'+mode+'-rank0-raw-v1'); snapshot.mkdir(exist_ok=False)
    dest = out/mode; dest.mkdir()
    manifest = []
    for p in files:
        digest = sha(p); rel = p.relative_to(root)
        manifest.append(dict(path=str(rel), bytes=p.stat().st_size, sha256=digest))
        if '_rank0_' in p.relative_to(root/'profile').parts[0]:
            q = snapshot/p.relative_to(root/'profile'); q.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, q); assert sha(q) == digest
    (dest/'raw-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    shutil.copyfile(root/('early-sample-shadow-c4-supply-profile-v1/'+mode+'-metadata.json'), dest/'metadata.json')
    with tarfile.open(dest/'all16-raw.tar.gz', 'w:gz', compresslevel=3) as tar:
        for p, entry in zip(files, manifest):
            tar.add(p, arcname=entry['path'], recursive=False); assert sha(p) == entry['sha256']
    image = 'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
    argv = ['docker', 'run', '--rm', '--pull', 'never', '--network', 'none', '--name', 'glm53-early-sample-shadow-c4-supply-'+mode+'-rank0-export-v1',
            '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--cpus', '16', '--memory', '64g', '--pids-limit', '4096',
            '--tmpfs', '/tmp:rw,size=2g', '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'OMP_NUM_THREADS=1', '--device', '/dev/davinci_manager',
            '-v', str(snapshot)+':/profiles', '-v', '/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro', '--entrypoint', '/usr/bin/timeout', image,
            '--signal=TERM', '--kill-after=15', '300', 'python3', '-c', "from torch_npu.profiler.profiler import analyse; analyse('/profiles',max_process_number=1,export_type='text')"]
    started = time.monotonic()
    with (dest/'export-stdout.txt').open('xb') as a, (dest/'export-stderr.txt').open('xb') as b:
        proc = subprocess.run(argv, stdout=a, stderr=b, timeout=330)
    assert proc.returncode == 0, (mode, proc.returncode)
    directory = next(snapshot.glob('*_rank0_*_ascend_pt')); trace = next(directory.rglob('trace_view.json'))
    rank_out = dest/'rank0'; rank_out.mkdir(); shutil.copyfile(trace, rank_out/'trace_view.json')
    for name in ['kernel_details.csv', 'api_statistic.csv', 'communication_matrix.json']:
        for p in directory.rglob(name): shutil.copyfile(p, rank_out/name)
    results.append(dict(mode=mode, completed=True, export_rc=0, export_elapsed_s=time.monotonic()-started, argv=argv,
                        raw_files=len(manifest), raw_bytes=sum(z['bytes'] for z in manifest), raw_ranks=record['ranks'],
                        raw_archive_bytes=(dest/'all16-raw.tar.gz').stat().st_size, raw_archive_sha256=sha(dest/'all16-raw.tar.gz'),
                        trace_bytes=trace.stat().st_size, trace_sha256=sha(trace), compute_devices_used=[], exported_ranks=[0]))
    (out/'progress.json').write_text(json.dumps(results, indent=2)+'\n')
result = dict(completed=True, modes=results, scope='All16 raw per mode preserved with per-file SHA and archive before CPU-only rank0 export in pinned090 image. No inference requests, NPU compute mounts or profiler left active. Rank0 analysis does not imply all32rank/PD-byte coverage or unprofiled service causality.')
(out/'export-result.json').write_text(json.dumps(result, indent=2)+'\n')
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(result)+'\nGLM53_RESULT_JSON_END')
