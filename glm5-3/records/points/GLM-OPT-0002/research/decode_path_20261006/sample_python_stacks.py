"""Bounded nonblocking source-stack reader; never imports Torch or sends model work."""
from pathlib import Path
import hashlib
import json
import os
import selectors
import signal
import subprocess
import sys
import time


def protocol_line(timeout):
    sel = selectors.DefaultSelector()
    sel.register(sys.stdin, selectors.EVENT_READ)
    try:
        assert sel.select(timeout), 'protocol deadline'
        return sys.stdin.readline().strip()
    finally:
        sel.close()


def main(root):
    plan = json.loads((root / "observer_plan.json").read_text())
    binary = plan["binary"]
    assert hashlib.sha256(Path(binary["path"]).read_bytes()).hexdigest() == binary["sha256"]
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    for row in plan["workers"]:
        stat = Path("/proc/%d/stat" % row["pid"]).read_text().rsplit(")", 1)[1].split()
        assert boot == row["boot_id"] and stat[19] == row["start_ticks"] and stat[0] != "Z"
        assert row["rank_name"] in Path("/proc/%d/cmdline" % row["pid"]).read_bytes().decode()
    children = []
    records = []
    ready_ns = None
    try:
        print(json.dumps(dict(ready=True, CPU_reader_only=True)), flush=True)
        assert protocol_line(150) == "go"
        started_ns = time.monotonic_ns()
        for row in plan["workers"]:
            output = root / (row["rank_name"] + ".speedscope.json")
            assert not output.exists()
            argv = [binary["path"], "record", "--nonblocking", "--full-filenames",
                    "--threads", "--rate", "100", "--duration", "20",
                    "--format", "speedscope", "--pid", str(row["pid"]), "--output", str(output)]
            err = (root / (row["rank_name"] + ".stderr.log")).open("xb")
            child = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=err, text=True)
            err.close()
            children.append(child)
            sel = selectors.DefaultSelector()
            sel.register(child.stdout, selectors.EVENT_READ)
            try:
                assert sel.select(12), "reader readiness timeout"
                line = child.stdout.readline()
            finally:
                sel.close()
            assert "Sampling process 100 times a second" in line and child.poll() is None, line
            records.append(dict(worker=row, reader_pid=child.pid, argv=argv,
                                startup_line=line, output=str(output), stderr=str(err.name),
                                reader_ready_ns=time.monotonic_ns()))
        ready_ns = time.monotonic_ns()
        print(json.dumps(dict(go=True, started_ns=started_ns, ready_ns=ready_ns,
                              setup_ns=ready_ns-started_ns, readers=records)), flush=True)
        # A hard20-second CLI duration bounds sampling even if the parent exits.
        assert protocol_line(40) == "done"
        stopped_ns = time.monotonic_ns()
        for child, record in zip(children, records):
            if child.poll() is None:
                stat = Path('/proc/%d/stat' % child.pid).read_text().rsplit(')', 1)[1].split()
                record.update(reader_start_ticks=stat[19], reader_user_ticks=int(stat[11]),
                              reader_system_ticks=int(stat[12]), clock_ticks_per_second=os.sysconf('SC_CLK_TCK'))
            if child.poll() is None:
                child.send_signal(signal.SIGINT)  # signal only the owned reader
        for child, record in zip(children, records):
            tail, _ = child.communicate(timeout=8)
            record.update(exit=child.returncode, stdout_tail=tail)
            assert child.returncode == 0, record
            output = Path(record["output"])
            raw = output.read_bytes()
            z = json.loads(raw)
            record.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                          profiles=len(z["profiles"]), frames=len(z["shared"]["frames"]))
        result = dict(done=True, started_ns=started_ns, ready_ns=ready_ns,
                      stopped_ns=stopped_ns, readers=records, nonblocking=True,
                      locals_collected=False, NPU_profiler_active=False)
        (root / "observer_final.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        cleanup = []
        for child in children:
            try:
                if child.poll() is None:
                    child.send_signal(signal.SIGINT)
                    try:
                        child.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        child.terminate()
                        try:
                            child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait(timeout=5)
                cleanup.append(dict(reader_pid=child.pid, exit=child.returncode))
            except Exception as error:
                cleanup.append(dict(reader_pid=child.pid, error=repr(error)))
        (root / 'observer_cleanup.json').write_text(json.dumps(cleanup, indent=2) + '\n')


if __name__ == "__main__":
    main(Path(sys.argv[1]))
