# SPDX-License-Identifier: Apache-2.0
"""Source-only cancellation cleanup fence; callbacks own all remote effects.

Decode's host must independently require this permit once destinations have
been exposed. A controller-only ordering rule cannot protect a host's finally.
This helper is not mounted or runtime qualified.
"""
import hashlib
import json
import time


def writer_stop_proof(status, *, run_id, image, container):
    """Accept actual saved post-stop observations for this exact writer only."""
    if not isinstance(status, dict):
        raise ValueError('Missing writer host status')
    if (status.get('project_id'), status.get('run_id'), status.get('node'),
        status.get('image'), status.get('container')) != (
            'glm53-flash', run_id, 46, image, container):
        raise ValueError('Writer ownership mismatch')
    if not status.get('cleanup_identity_verified'):
        raise ValueError('Writer cleanup identity was not verified')
    stopped = status.get('post_inspect', {})
    idle = status.get('post_npu', {})
    state = stopped.get('stdout', '').strip().split()
    if (stopped.get('rc') != 0 or len(state) != 2 or state[0] != 'false'
            or not state[1].lstrip('-').isdigit()):
        raise ValueError('Writer is not proven stopped')
    if idle.get('rc') != 0 or idle.get('stdout', '').count('No running processes found') != 8:
        raise ValueError('Writer NPUs are not proven idle')
    # Preserve a nonzero exit in the proof; stopped is a cleanup condition,
    # whereas successful native shutdown remains an independent acceptance gate.
    fields = {'project_id': 'glm53-flash', 'run_id': run_id, 'writer_node': 46,
              'writer_container': container, 'image': image,
              'post_inspect': stopped, 'post_npu': idle,
              'cleanup_identity_verified': True}
    fields['sha256'] = hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()
    return fields


def verify_decode_permit(permit, *, run_id, image, writer_container):
    """Decode host's independent cleanup guard after destination exposure."""
    expected = writer_stop_proof(
        {'project_id': permit.get('project_id'), 'run_id': permit.get('run_id'),
         'node': permit.get('writer_node'), 'container': permit.get('writer_container'),
         'image': permit.get('image'),
         'cleanup_identity_verified': permit.get('cleanup_identity_verified'),
         'post_inspect': permit.get('post_inspect'), 'post_npu': permit.get('post_npu')},
        run_id=run_id, image=image, container=writer_container)
    if permit != expected:
        raise ValueError('Wrong or corrupted writer-stop permit')
    return True


def cleanup(signal, read_writer_status, *, run_id, image, container,
            bound=90., monotonic=time.monotonic, sleep=time.sleep):
    """Return failures explicitly; never permit Decode destruction on timeout."""
    result = {'project_id': 'glm53-flash', 'writer_stop_proven': False,
              'decode_stop_permitted': False, 'errors': []}
    try:
        signal(46, 'stop.json', {'project_id': 'glm53-flash', 'run_id': run_id,
                                'reason': 'writer_first_cleanup', 'error': None})
    except Exception as exc:
        result['errors'].append('writer stop signal: ' + repr(exc))
    deadline = monotonic() + bound
    while monotonic() < deadline:
        try:
            proof = writer_stop_proof(read_writer_status(), run_id=run_id,
                                      image=image, container=container)
            result['writer_stop_proven'] = True
            result['writer_stop_proof'] = proof
            break
        except Exception as exc:
            result['last_observation_error'] = repr(exc)
        sleep(.5)
    if not result['writer_stop_proven']:
        result['errors'].append('Writer stop/idle unproven; keep Decode destinations owned')
        return result
    try:
        # Permit must be published before stop: Decode's host checks it even
        # when its own collection/startup/timeout path enters finally first.
        signal(47, 'writer-stop-permit.json', result['writer_stop_proof'])
        result['decode_stop_permitted'] = True
        signal(47, 'stop.json', {'project_id': 'glm53-flash', 'run_id': run_id,
                                'writer_stop_proof_sha256': proof['sha256'], 'error': None})
    except Exception as exc:
        result['errors'].append('Decode permit/stop signal: ' + repr(exc))
    return result
