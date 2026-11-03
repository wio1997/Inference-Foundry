"""CPU-only heartbeat, signal-identity and unreleased child gate checks."""
import ast
import os
from pathlib import Path
import signal
import subprocess
import sys
import json
import time
import unittest
from unittest import mock
import engram_liveness as live

HERE=Path(__file__).resolve().parent


def heartbeat(phase='supervising',start=0,updated=3500):
    return {'run_id':'a'*32,'owner_pid':123,'owner_birth':[1,2,501],'child_pid':456,
            'child_birth':[2,3,501],'phase':phase,'phase_start':start,'updated':updated,
            'deadline':start+live.PHASE_LIMITS[phase]}


class LivenessTests(unittest.TestCase):
    def test_supervising_40min_load_not_timed_out(self):
        self.assertEqual(live.check_heartbeat(heartbeat(updated=2399),'a'*32,2400),'FRESH')
    def test_starting_has_1000s_absolute_deadline(self):
        self.assertEqual(live.check_heartbeat(heartbeat('starting',updated=999),'a'*32,999),'FRESH')
        self.assertEqual(live.check_heartbeat(heartbeat('starting',updated=1001),'a'*32,1001),'DEADLINE')
    def test_sigstop_progress_stale_and_absolute_timeout(self):
        self.assertEqual(live.check_heartbeat(heartbeat(updated=2300),'a'*32,2500),'STALE')
        self.assertEqual(live.check_heartbeat(heartbeat(updated=6001),'a'*32,6001),'DEADLINE')
    def test_invalid_identity_or_raised_deadline_refused(self):
        h=heartbeat();h['deadline']+=1
        self.assertEqual(live.check_heartbeat(h,'a'*32,3501),'DEADLINE')
        self.assertEqual(live.check_heartbeat(heartbeat(),'b'*32,3501),'INVALID')
    def test_group_signal_only_owned_same_identity_and_parent(self):
        p={'birth':[1,2,501],'ppid':123}
        with mock.patch.object(live,'process_info',return_value=p),mock.patch.object(os,'getpgid',return_value=456),mock.patch.object(os,'getsid',return_value=456),mock.patch.object(os,'killpg') as kill:
            self.assertFalse(live.signal_owned_group(456,[9,2,501],signal.SIGTERM,123))
            self.assertFalse(live.signal_owned_group(456,[1,2,501],signal.SIGTERM,999))
            kill.assert_not_called()
            self.assertTrue(live.signal_owned_group(456,[1,2,501],signal.SIGTERM,123))
            kill.assert_called_once_with(456,signal.SIGTERM)
    def test_stopped_group_killed_without_resume(self):
        p={'birth':[1,2,501],'ppid':123}
        with mock.patch.object(live,'process_info',return_value=p),mock.patch.object(os,'getpgid',return_value=456),mock.patch.object(os,'getsid',return_value=456),mock.patch.object(live,'group_absent',side_effect=[False,True]),mock.patch.object(os,'killpg') as sent,mock.patch.object(time,'sleep'):
            self.assertEqual(live.stop_owned_group(456,[1,2,501],123),'EXIT_CONFIRMED')
            sent.assert_called_once_with(456,signal.SIGKILL)
    def test_leader_disappears_but_group_exists_unknown_no_kill(self):
        with mock.patch.object(live,'process_info',side_effect=RuntimeError('leader gone')),mock.patch.object(live,'group_absent',return_value=False),mock.patch.object(os,'killpg') as sent:
            self.assertEqual(live.stop_owned_group(456,[1,2,501],123),'UNKNOWN')
            sent.assert_not_called()
    def test_dead_owner_requires_pid_and_group_absent(self):
        with mock.patch.object(os,'kill',side_effect=ProcessLookupError()),mock.patch.object(live,'group_absent',return_value=True):
            self.assertTrue(live.owner_absent(123))
        with mock.patch.object(os,'kill',side_effect=ProcessLookupError()),mock.patch.object(live,'group_absent',return_value=False):
            self.assertFalse(live.owner_absent(123))
        with mock.patch.object(os,'kill',side_effect=PermissionError()):
            self.assertFalse(live.owner_absent(123))
    def test_dead_owner_snapshot_still_bound(self):
        old=heartbeat(updated=100);current=heartbeat(updated=101)
        self.assertTrue(live.snapshot_bound(current,old,'a'*32,123,300))
        current['owner_birth']=[9,2,501]
        self.assertFalse(live.snapshot_bound(current,old,'a'*32,123,300))
    def test_failed_takeover_preserves_supervision_and_owner_resume(self):
        source=(HERE/'watchdog_heartbeat.template').read_text()
        self.assertIn("if live.owner_absent(owner):",source)
        self.assertIn("signal.SIGCONT)",source)
        self.assertIn("OWNER_RESUMED_SUPERVISION_CONTINUES",source)
        self.assertIn("OWNER_DEAD_CHILD_UNKNOWN",source)
        controller=(HERE/'orchestrator_heartbeat.template').read_text()
        self.assertIn("if live.group_absent(p.pid):",controller)
    def test_stopped_state_requires_matching_identity_and_group(self):
        info={'birth':[1,2,501],'status':4,'ppid':1}
        with mock.patch.object(live,'process_info',return_value=info),mock.patch.object(os,'getpgid',return_value=123),mock.patch.object(os,'getsid',return_value=123):
            self.assertTrue(live.wait_owner_stopped(123,[1,2,501]))
            self.assertFalse(live.wait_owner_stopped(123,[9,2,501]))
        source=(HERE/'watchdog_heartbeat.template').read_text()
        self.assertLess(source.index('live.wait_owner_stopped'),source.index('frozen_state,current=read_progress()'))
    def test_real_stop_state_on_local_child(self):
        if sys.platform!='darwin':self.skipTest('Mac libproc only')
        p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
        try:
            birth=live.identity(p.pid)
            self.assertTrue(live.signal_owned_group(p.pid,birth,signal.SIGSTOP))
            self.assertTrue(live.wait_owner_stopped(p.pid,birth))
        finally:
            p.kill();p.wait(timeout=5)
    def test_completed_leader_with_live_group_blocks_next_phase(self):
        tree=ast.parse((HERE/'engram_adapter_orchestrate.py').read_text())
        fns=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('phase','run_phase')]
        child=mock.Mock();child.pid=456;child.poll.return_value=0;child.returncode=0
        fake_live=mock.Mock();fake_live.identity.return_value=[2,3,501];fake_live.group_absent.return_value=False;fake_live.PHASE_LIMITS=live.PHASE_LIMITS
        cur={'child_pid':None,'child_birth':None}
        fake_os=mock.Mock();fake_os.getpgid.return_value=456;fake_os.getsid.return_value=456
        env={'current':cur,'live':fake_live,'beat':mock.Mock(),'time':time,'subprocess':mock.Mock(Popen=mock.Mock(return_value=child),PIPE=object()),'sys':sys,'CHILD_GATE':'gate','here':HERE,'os':fake_os,'env':{},'owner':123,'watchdog':mock.Mock()}
        exec(compile(ast.Module(body=fns,type_ignores=[]),'mock-controller','exec'),env)
        with self.assertRaisesRegex(RuntimeError,'exit UNKNOWN'):
            env['run_phase']('starting','unused')
        self.assertEqual(cur['child_pid'],456);self.assertEqual(cur['child_birth'],[2,3,501])
        with self.assertRaisesRegex(RuntimeError,'phase transition'):
            env['phase']('supervising')
        self.assertEqual(cur['child_pid'],456)
        env['subprocess'].Popen.assert_called_once()
    def test_missing_snapshot_cannot_use_old_record_for_recovery(self):
        self.assertFalse(live.snapshot_bound({},heartbeat(),'a'*32,123,3501))
        self.assertIn("save('TAKEOVER_SNAPSHOT_UNKNOWN',reason=state);continue",(HERE/'watchdog_heartbeat.template').read_text())
    def test_latest_stale_child_not_previous_child(self):
        old=heartbeat(updated=100);old['child_pid']=None;old['child_birth']=None
        current=heartbeat(updated=101)
        def proc(pid):return {'birth':[1,2,501],'ppid':1} if pid==123 else {'birth':[2,3,501],'ppid':123}
        with mock.patch.object(live,'process_info',side_effect=proc),mock.patch.object(os,'getpgid',side_effect=lambda p:p),mock.patch.object(os,'getsid',side_effect=lambda p:p):
            self.assertTrue(live.validate_takeover_snapshot(current,old,'a'*32,123,300))
            current['child_birth']=[9,3,501]
            self.assertFalse(live.validate_takeover_snapshot(current,old,'a'*32,123,300))
        source=(HERE/'watchdog_heartbeat.template').read_text()
        self.assertIn("stop_owned_group(current['child_pid'],current['child_birth']",source)
        self.assertNotIn("stop_owned_group(last.get('child_pid')",source)
    def test_watchdog_read_rejects_foreign_owner(self):
        tree=ast.parse((HERE/'engram_adapter_watchdog.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='read_progress')
        fake=mock.Mock();h=heartbeat(start=time.monotonic()-2,updated=time.monotonic()-1);h['owner_pid']=999
        fake.read_text.return_value=json.dumps(h)
        env={'heartbeat':fake,'json':json,'run_id':'a'*32,'time':time,'live':live,'owner':123,'os':os}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'mock-watchdog-read','exec'),env)
        state,_=env['read_progress']()
        self.assertEqual(state,'OWNER_MISMATCH')
    def test_child_gate_eof_never_executes_script(self):
        tree=ast.parse((HERE/'engram_adapter_orchestrate.py').read_text())
        gate=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CHILD_GATE' for t in n.targets))
        r=subprocess.run([sys.executable,'-c',gate,'/does-not-exist'],input=b'',capture_output=True,timeout=5)
        self.assertEqual(r.returncode,125)
    def test_real_local_process_identity(self):
        if sys.platform!='darwin':self.skipTest('Mac libproc only')
        birth=live.identity(os.getpid())
        self.assertEqual(birth[-1],os.getuid());self.assertGreater(birth[0],0)


if __name__=='__main__':unittest.main()
