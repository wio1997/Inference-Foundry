"""Observe the exact bounded quality controller before releasing its dependency."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import select


def read(path):
    return json.loads(path.read_bytes())


def wait_quality(control, source, expected_pid, cutoff, enabled=lambda: True):
    started = read(control / 'actual_launch_handle.json')
    assert started['controller_PID'] == expected_pid
    deadline = datetime.fromisoformat(cutoff)
    try:
        fd = os.pidfd_open(expected_pid)
    except ProcessLookupError:
        pass
    else:
        with os.fdopen(fd, 'rb') as event:
            poller = select.poll()
            poller.register(event, select.POLLIN)
            if not poller.poll(0):
                command = Path('/proc', str(expected_pid), 'cmdline')
                try:
                    argv = command.read_bytes().split(b'\0')
                except FileNotFoundError:
                    argv = []
                if not poller.poll(0):
                    assert str(source).encode() in argv and b'--hold' in argv
                    assert b'--gate-sha256' in argv
                while not poller.poll(0):
                    remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
                    if remaining <= 0 or not enabled():
                        raise TimeoutError('Quality controller terminal evidence is still pending')
                    poller.poll(min(60000, max(1, int(remaining * 1000))))
    worker = read(control / 'worker_handle.json')
    assert worker['controller_PID'] == expected_pid
    outcome = read(control / 'terminal.json')
    assert datetime.fromisoformat(outcome['at']) >= datetime.fromisoformat(started['at'])
    assert type(outcome['exit_code']) is int
    return {'controller_PID': expected_pid, 'worker_PID': worker['docker_exec_PID'],
            'worker_exit_code': outcome['exit_code'], 'terminal_at': outcome['at']}
