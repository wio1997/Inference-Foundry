"""Exercise generated CAPTURE fragment with fake subprocesses; no device I/O."""
import ast
import json
from pathlib import Path
import subprocess
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent
tree=ast.parse((HERE/'engram_adapter_supervise.py').read_text())
CAPTURE=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='CAPTURE' for x in n.targets))


class FakeProc:
    def __init__(self):self.alive=True;self.terminated=False;self.killed=False
    def poll(self):return None if self.alive else 0
    def terminate(self):self.terminated=True;self.alive=False
    def kill(self):self.killed=True;self.alive=False
    def wait(self,timeout):return 0


class CaptureTests(unittest.TestCase):
    def run_capture(self,sets,lock_failure=False):
        run='a'*32;diag='d'*64;proc=FakeProc();started=[];fence_calls=[]
        def check(args,**kwargs):
            if args[:2]==['docker','inspect']:
                return json.dumps([{'Id':diag,'Config':{'Labels':{'wio.binding_run_id':run}},'State':{'Running':True}}])
            if args[:2]==['docker','top']:return 'PID\n123\n'
            if args[:3]==['docker','exec','stage']:return 'snapshot'
            raise AssertionError(args)
        def parse(_):return sets.pop(0)
        def fence(_):
            fence_calls.append(1)
            if lock_failure and len(fence_calls)>=3:raise RuntimeError('fence lost')
        env={'ROOT':'/data/tiankuan/wio/main/task','STAGE':'stage','DIAG_ID':diag,'RUN_ID':run,
             'APP':'/owned','acquire_owner_fence':lambda _:object(),'ensure_owner_fence':fence,'parse_npu_processes':parse}
        def launch(*args,**kwargs):started.append(args);return proc
        with mock.patch.object(subprocess,'check_output',side_effect=check),mock.patch.object(subprocess,'Popen',side_effect=launch),mock.patch('time.sleep',return_value=None):
            with self.assertRaises((AssertionError,RuntimeError)):
                exec(compile(CAPTURE,'fake_capture','exec'),env)
        return proc,started
    def test_other_pid_before_capture_never_starts_client(self):
        proc,started=self.run_capture([{123,999}])
        self.assertEqual(started,[]);self.assertFalse(proc.terminated)
    def test_other_pid_during_capture_terminates_only_owned_child(self):
        proc,started=self.run_capture([{123},{123,999}])
        self.assertEqual(len(started),1);self.assertTrue(proc.terminated);self.assertFalse(proc.killed)
        self.assertIn('WIO_ENGRAM_RUN_ID='+'a'*32,started[0][0])
    def test_lost_fence_during_capture_terminates_child(self):
        proc,started=self.run_capture([{123}],lock_failure=True)
        self.assertEqual(len(started),1);self.assertTrue(proc.terminated)


if __name__=='__main__':unittest.main()
