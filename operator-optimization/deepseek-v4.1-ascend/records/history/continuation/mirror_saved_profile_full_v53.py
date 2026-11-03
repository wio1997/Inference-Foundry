"""Cache the indexed original saved files; no capture, device or model operation."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shlex
import shutil
import subprocess
import time

from finalize_origin_coverage_v9 import N, O, read, write
from mirror_saved_profile_core_v53 import ORIGINAL, CACHE, ssh_args, run_readonly_python


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    index_path = O / 'original-profile-full-cache-index-v53.json'
    caps = read(CACHE / 'analysis/profile-capabilities-index-v12-v2.json')
    native = read(CACHE / 'analysis/native-execution-identity-v14/index.json')
    pmu = read(CACHE / 'analysis/native-PMU-complete-v18/index.json')
    refs = []
    # Complete derived native/PMU records first, then raw databases with full callchains.
    for family, rows in [('native', native['databases']), ('PMU', pmu['databases']),
                         ('config', caps['profiler_infos']), ('host-metadata', caps['metadata']),
                         ('raw-database', caps['databases'])]:
        for row in rows:
            p = Path(row['path']); rel = p.relative_to(ORIGINAL)
            assert '..' not in rel.parts
            refs.append(dict(family=family, original_path=str(p), relative=str(rel), sha256=row['sha256']))
    assert len(refs) == 144 and len({r['relative'] for r in refs}) == len(refs)
    code = '''import os,socket,json
from pathlib import Path
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105' and os.geteuid()==0
out=[]
for name in %r:
 p=Path(name);s=p.stat();assert p.is_file()
 out.append(dict(path=name,bytes=s.st_size,mtime_ns=s.st_mtime_ns))
print(json.dumps(out))
''' % [r['original_path'] for r in refs]
    probe = run_readonly_python(code, timeout=45)
    assert probe.returncode == 0, probe.stderr.decode(errors='replace')[-1000:]
    stats = json.loads(probe.stdout)
    assert len(stats) == len(refs)
    total = sum(x['bytes'] for x in stats)
    assert total < 8 * 1024**3 and max(x['bytes'] for x in stats) < 1024**3
    assert shutil.disk_usage(CACHE).free > total + 20 * 1024**3
    for row, st in zip(refs, stats):
        assert row['original_path'] == st['path']
        row.update(bytes=st['bytes'], mtime_ns=st['mtime_ns'])
    result = dict(generated_utc=datetime.now(timezone.utc).isoformat(),
                  original_root=str(ORIGINAL), cache_root=str(CACHE),
                  indexed_files=len(refs), indexed_bytes=total, files=[],
                  all_indexed_files_copied_and_hash_matched=False,
                  all_original_profile_files_copied=False,
                  no_new_capture_model_device_service_health_process_work=True,
                  overall_complete=False)
    write(O / 'original-profile-cache-plan-v53.json', dict(**result, planned_files=refs))
    print(json.dumps(dict(plan_files=len(refs), plan_bytes=total)), flush=True)
    start = time.monotonic()
    for i, row in enumerate(refs):
        dest = CACHE / row['relative']; dest.parent.mkdir(parents=True, exist_ok=True)
        reused = dest.exists()
        if reused:
            assert dest.stat().st_size == row['bytes'] and digest(dest) == row['sha256']
        else:
            code = '''import os,socket,sys
from pathlib import Path
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105' and os.geteuid()==0
p=Path(%r);before=p.stat();assert(before.st_size,before.st_mtime_ns)==(%d,%d)
with p.open('rb')as f:
 while True:
  block=f.read(1024*1024)
  if not block:break
  sys.stdout.buffer.write(block)
after=p.stat();assert(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
''' % (row['original_path'], row['bytes'], row['mtime_ns'])
            temp = dest.with_suffix(dest.suffix + '.partial')
            with temp.open('wb') as output:
                p = subprocess.run(ssh_args('python3 -I -S -c ' + shlex.quote(code)),
                                   stdout=output, stderr=subprocess.PIPE, timeout=300)
            if p.returncode != 0:
                result['error'] = dict(relative=row['relative'], returncode=p.returncode,
                                       stderr=p.stderr.decode(errors='replace')[-1000:])
                write(index_path, result)
                raise RuntimeError(result['error'])
            assert temp.stat().st_size == row['bytes'] and digest(temp) == row['sha256'], row['relative']
            temp.replace(dest)
        result['files'].append(dict(**row, local_path=str(dest), hash_matched=True, cache_reused=reused))
        result['completed_files'] = i + 1
        result['completed_bytes'] = sum(x['bytes'] for x in result['files'])
        result['updated_utc'] = datetime.now(timezone.utc).isoformat()
        write(index_path, result)
        if i % 4 == 0 or i + 1 == len(refs):
            print(json.dumps(dict(completed=i+1, total=len(refs), bytes=result['completed_bytes'],
                                  elapsed_s=round(time.monotonic()-start, 1), family=row['family'])), flush=True)
    result['all_indexed_files_copied_and_hash_matched'] = True
    result['raw_database_files'] = sum(x['family'] == 'raw-database' for x in refs)
    write(index_path, result)


if __name__ == '__main__':
    main()
