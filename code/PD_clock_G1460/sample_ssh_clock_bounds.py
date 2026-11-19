"""Bracket P and D host realtime offsets using existing authenticated SSH transports."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import select
import shlex
import subprocess
import time


REMOTE = """
import sys
import time
for raw in sys.stdin.buffer:
    received = time.time_ns()
    first = time.monotonic_ns()
    seq = int(raw)
    sent = time.time_ns()
    last = time.monotonic_ns()
    print(seq, received, sent, last-first, flush=True)
"""


def exchange(proc, sequence):
    mono_start = time.monotonic_ns()
    start = time.time_ns()
    proc.stdin.write(f'{sequence}\n'.encode())
    pending = bytearray()
    deadline = time.monotonic() + 5
    while not pending.endswith(b'\n'):
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([proc.stdout], [], [], remaining)[0]:
            raise TimeoutError('Clock response timed out')
        packet = proc.stdout.read(128)
        if not packet or len(pending) + len(packet) > 128:
            raise ValueError('Missing or malformed timestamp response')
        pending.extend(packet)
    stop = time.time_ns()
    mono_stop = time.monotonic_ns()
    observed, remote_in, remote_out, remote_mono_elapsed = map(int, pending.split())
    if observed != sequence:
        raise ValueError('Clock response sequence mismatch')
    if abs((stop - start) - (mono_stop - mono_start)) >= 1_000_000:
        raise ValueError('Collector wall clock moved during a sample')
    if abs((remote_out - remote_in) - remote_mono_elapsed) >= 1_000_000:
        raise ValueError('Endpoint wall clock moved during a response')
    if not (start < stop and remote_in <= remote_out):
        raise ValueError('Clock sample timestamps moved backwards')
    low, high = remote_out - stop, remote_in - start
    if low > high:
        raise ValueError('Timestamps imply a negative transport interval')
    return {'sequence': sequence, 'collector_send_realtime_ns': start, 'collector_receive_realtime_ns': stop,
            'remote_receive_realtime_ns': remote_in, 'remote_send_realtime_ns': remote_out,
            'offset_lower_ns': low, 'offset_upper_ns': high, 'round_trip_ns': mono_stop - mono_start}


def collect(argv, sample_count):
    remote = shlex.join(['python3', '-u', '-B', '-c', REMOTE])
    proc = subprocess.Popen([*argv, remote], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, bufsize=0)
    try:
        exchange(proc, 0)
        samples = [exchange(proc, seq) for seq in range(1, sample_count + 1)]
        proc.stdin.close()
        if proc.wait(timeout=5):
            raise RuntimeError('Timestamp transport failed to close cleanly')
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
        proc.stdout.close()
        if not proc.stdin.closed:
            proc.stdin.close()
    low = max(row['offset_lower_ns'] for row in samples)
    high = min(row['offset_upper_ns'] for row in samples)
    if low > high:
        raise ValueError('Clock offset was inconsistent across the sample bounds')
    return {'remote_minus_collector_ns': [low, high], 'samples': samples}


def paired(transports, sample_count):
    with ThreadPoolExecutor(max_workers=2) as pool:
        runs = {role: pool.submit(collect, transports[role], sample_count) for role in ('prefill', 'decode')}
        out = {role: job.result() for role, job in runs.items()}
    p_low, p_high = out['prefill']['remote_minus_collector_ns']
    d_low, d_high = out['decode']['remote_minus_collector_ns']
    return {'prefill_minus_decode_realtime_ns': [p_low - d_high, p_high - d_low], 'roles': out,
            'scope': 'Short-window host CLOCK_REALTIME only, conditional on stable offsets during the exchanges. '
                     'Repeat near any captured window before comparing host timestamps; hardware trace clocks '
                     'require independent calibration.'}


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--transports', type=Path, required=True,
                     help='JSON object of prefill and decode SSH argv vectors, without the remote command')
    cli.add_argument('--samples', type=int, default=8, choices=range(2, 21))
    args = cli.parse_args()
    ssh_commands = json.loads(args.transports.read_bytes())
    report = paired(ssh_commands, args.samples)
    report['source_SHA256'] = sha256(Path(__file__).read_bytes()).hexdigest()
    print(json.dumps(report, indent=2, sort_keys=True))
