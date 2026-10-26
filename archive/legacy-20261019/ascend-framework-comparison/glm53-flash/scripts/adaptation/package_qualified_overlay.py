# SPDX-License-Identifier: Apache-2.0
"""Package a whole accepted bounded model unit's exact source, never raw data.

Local file preparation only. Resource actions and Git publication remain under
the project's frozen-review/ZCode execution contract.
"""
import argparse
import hashlib
import json
import pathlib
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package(run, output):
    acceptance = json.loads((run / 'acceptance.json').read_text())
    if acceptance.get('passed') is not True:
        raise ValueError('A whole accepted model unit is required')
    review = json.loads((run / 'review.json').read_text())
    manifest = json.loads((run / 'combined-manifest.json').read_text())
    # A primitive or a supplementary audit cannot satisfy the packaging gate.
    if len(manifest['files']) < 39 or not (run / 'receipt.json').is_file():
        raise ValueError('Expected the full native model PD source manifest')
    analysis = json.loads((run / 'analysis.json').read_text())
    if not (analysis.get('passed') is True and
            analysis.get('actual_SDK_destroy_complete_records') == 32 and
            analysis.get('native_PD_managers_workers_ZMQ_closed') == 32):
        raise ValueError('Full model manager/SDK shutdown has not qualified')
    for field, name in (('review_sha256', 'review.json'),
                        ('analysis_sha256', 'analysis.json'),
                        ('receipt_sha256', 'receipt.json'),
                        ('independent_validator_sha256', 'independent_acceptance.py')):
        if acceptance.get(field) != sha(run / name):
            raise ValueError('Acceptance provenance mismatch: ' + name)
    for name, expected in review['files_sha256'].items():
        if sha(run / name) != expected:
            raise ValueError('Frozen file changed: ' + name)
    if output.exists():
        raise FileExistsError(output)
    # Validate all paths and bytes before creating a partial package.
    for row in manifest['files']:
        path = pathlib.PurePosixPath(row['target'])
        if path.is_absolute() or '..' in path.parts or not str(path).startswith('sglang/'):
            raise ValueError('Invalid source target')
        if sha(run / 'overlay' / path) != row['sha256']:
            raise ValueError('Overlay changed: ' + str(path))
    output.mkdir(parents=True)
    for row in manifest['files']:
        dest = output / 'overlay' / row['target']
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run / 'overlay' / row['target'], dest)
    for name in ('combined-manifest.json', 'argv46.json', 'argv47.json',
                 'numerical-gates-v1.json', 'native-source-pins.json'):
        shutil.copyfile(run / name, output / name)
    lifecycle = bool(analysis.get('actual_late_write_cancel_ACK_MIN_safe_release_and_natural_reuse'))
    reorder = bool(analysis.get('actual_completion_reversal_groups'))
    qualified = analysis.get('scope', acceptance.get('scope'))
    if not isinstance(qualified, str) or not qualified:
        raise ValueError('Accepted exact scope is required')
    unqualified = ['generic production service', 'arbitrary peer-crash failure shutdown',
                   '80000-input/600-output', 'graph', 'MTP D5', 'cache/L3', 'vision',
                   'formal performance']
    if not lifecycle:
        unqualified.insert(0, 'cancellation/drain/recycle in this unit')
    if not reorder:
        unqualified.insert(0, 'forced completion reorder in this unit')
    provenance = {
        'project_id': 'glm53-flash', 'accepted_run': run.name,
        'review_sha256': sha(run / 'review.json'),
        'acceptance_sha256': sha(run / 'acceptance.json'),
        'analysis_sha256': sha(run / 'analysis.json'),
        'receipt_sha256': sha(run / 'receipt.json'),
        'image': 'sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263',
        'qualification': qualified,
        'accepted_verdict': analysis['verdict'],
        'bounded_cancel_drain_recycle_in_this_unit': lifecycle,
        'forced_completion_reorder_in_this_unit': reorder,
        'unqualified': unqualified,
        'formal_baseline': None,
        'source_targets': len(manifest['files']),
        'excluded': ['weights', 'raw logs', 'raw requests', 'business text', 'credentials'],
    }
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (output / 'README.md').write_text(
        '# GLM-5.3-Flash bounded SGLang B103 adaptation\n\n'
        'These are the exact readonly source bindings of the accepted unit recorded in '
        '`provenance.json`. Apply only to the pinned image and verify every source SHA. '
        'The argv files describe the qualified bounded profile. Source files retain '
        'their upstream license headers.\n\n'
        'New deployments require a fresh unique run directory, frozen identity/configuration '
        'review and one execution through local ZCode DeepSeek under EXECUTION_CONTRACT.md. '
        'Historical launchers and locks are never replayed. Native 15-second shutdown '
        'watchdog and numerical gates remain unchanged.\n\n'
        'The exact accepted scope and remaining unqualified behavior are listed in '
        '`provenance.json`; scope is never inherited from the number of source files.\n')
    return provenance


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=pathlib.Path, required=True)
    p.add_argument('--output', type=pathlib.Path, required=True)
    a = p.parse_args()
    print(json.dumps(package(a.run, a.output)))
