#!/usr/bin/env python3
"""Prepare a fresh two-round AISBench run against the existing fixed service."""
import argparse
from pathlib import Path
import re
import socket

ROOT = Path('/data/tiankuan/wio/910c-deepinfer-172')
VERSION = 'allopt-mc2-cap4096-v1-20261028'
PACKAGE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if socket.gethostname() != 'bm-jn-zs-zone1-910C-64G-10-172':
        raise SystemExit('This package is pinned to server 172.')
    if not re.fullmatch(r'retest-[a-z0-9-]{1,64}', args.run_id):
        parser.error('--run-id must start with retest- and contain lowercase letters, digits or hyphens')
    run = args.run_id
    required = [ROOT / 'tools/aisbench-venv-7bcb6440da11/bin/python',
                ROOT / 'tools/aisbench-7bcb6440da11/plugins/prefix_cache',
                ROOT / 'serving' / VERSION / 'ready-rank0.json']
    for file in required:
        if not file.exists():
            raise FileNotFoundError(str(file))
    destination = ROOT / 'benchmarks' / run
    targets = [destination, ROOT / 'runs' / (run + '-benchmark-status.json'),
               ROOT / 'tools' / (run + '-benchmark.py'), ROOT / 'tools' / (run + '-job.py')]
    if any(file.exists() for file in targets):
        raise FileExistsError('Run ID already exists: ' + run)
    driver = (PACKAGE / 'recorded/benchmark.py').read_text()
    driver = driver.replace("version='" + VERSION + "'", "version='" + run + "'", 1)
    driver = driver.replace("control=p/'serving'/version", "control=p/'serving'/" + repr(VERSION), 1)
    begin = driver.index("    prior=json.loads((p/'benchmarks/allopt-engram-uva-v2-20261028/progress.json')")
    end = driver.index("    save('COMPLETE')", begin)
    driver = driver[:begin] + driver[end:]
    job = (PACKAGE / 'recorded/benchmark-job.py').read_text().replace(VERSION, run)
    configs = {phase: (PACKAGE / 'recorded' / ('reference-' + phase + '.py')).read_text().replace(VERSION, run)
               for phase in ('measure1', 'measure2')}
    for name, text in [('driver', driver), ('job', job), *configs.items()]:
        compile(text, name, 'exec')
    destination.mkdir()
    for phase, text in configs.items():
        (destination / ('reference-' + phase + '.py')).write_text(text)
    (ROOT / 'tools' / (run + '-benchmark.py')).write_text(driver)
    (ROOT / 'tools' / (run + '-job.py')).write_text(job)
    print('Prepared:', destination)
    print('Run: python3', ROOT / 'tools' / (run + '-job.py'))


if __name__ == '__main__':
    main()
