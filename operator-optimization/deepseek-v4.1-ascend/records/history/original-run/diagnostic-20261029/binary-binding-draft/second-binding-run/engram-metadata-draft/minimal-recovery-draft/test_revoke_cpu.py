"""Local temporary-file flock simulation and structural audit; no remote access."""
import ast,fcntl,multiprocessing as mp,os,tempfile,time,unittest
from unittest import mock
from pathlib import Path
HERE=Path(__file__).resolve().parent

def holder(root,ready,release,events):
    with open(root+'/fence','w') as f:
        fcntl.flock(f,fcntl.LOCK_EX);events.put('old-start-entered');ready.set();release.wait(4);events.put('old-start-finished')
def revoke(root,ready,events):
    ready.wait(4)
    with open(root+'/fence','w') as f:
        fcntl.flock(f,fcntl.LOCK_EX);Path(root+'/cancel').touch();events.put('revoked')
def late(root,events):
    with open(root+'/fence','w') as f:
        fcntl.flock(f,fcntl.LOCK_EX);events.put('refused' if Path(root+'/cancel').exists() else 'WRONG_START')

class Tests(unittest.TestCase):
    def test_fence_orders_inflight_then_revoke_then_late_start(self):
        with tempfile.TemporaryDirectory() as root:
            ready=mp.Event();release=mp.Event();q=mp.Queue()
            a=mp.Process(target=holder,args=(root,ready,release,q));b=mp.Process(target=revoke,args=(root,ready,q))
            a.start();b.start();self.assertEqual(q.get(timeout=4),'old-start-entered');release.set()
            a.join(4);b.join(4);self.assertEqual((a.exitcode,b.exitcode),(0,0))
            self.assertEqual([q.get(timeout=4),q.get(timeout=4)],['old-start-finished','revoked'])
            c=mp.Process(target=late,args=(root,q));c.start();c.join(4);self.assertEqual(q.get(timeout=4),'refused')
    def test_cancel_check_is_inside_each_mutating_remote_block(self):
        mapping={'engram_adapter_start.py':['REMOTE'],'engram_adapter_supervise.py':['CAPTURE','RESTORE'],'engram_adapter_recover.py':['REMOTE']}
        for filename,names in mapping.items():
            src=(HERE/filename).read_text();self.assertIn('DISABLED:',src)
            tree=ast.parse(src)
            for name in names:
                value=next(n.value.value for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name and isinstance(n.value,ast.Constant))
                self.assertLess(value.index('acquire_owner_fence('),value.index("marker=marker_dir+'/cancel-'+RUN_ID"))
                self.assertIn('cannot persist cancellation marker' if filename.endswith('recover.py') else 'run cancelled or cancellation state unavailable',value)
    def test_unconditional_revoke_then_late_stop_recovery(self):
        tree=ast.parse((HERE/'watchdog.py').read_text())
        loop=next(n for n in tree.body if isinstance(n,ast.While) and isinstance(n.test,ast.Constant) and n.test.value is True)
        events=[]
        observations=iter([{'running':True,'health':200},{'running':False,'health':None},{'running':True,'health':200},{'running':True,'health':200},{'running':True,'health':200}])
        ticks=iter([0,0,150,300,450,600])
        fake_time=mock.Mock();fake_time.monotonic.side_effect=lambda:next(ticks);fake_time.sleep.return_value=None
        def observe():events.append('observe');return next(observations)
        def recover(*a,**kw):events.append('recover');return mock.Mock(returncode=0)
        fake_sub=mock.Mock();fake_sub.run.side_effect=recover
        env={'healthy_since':None,'need_fence_recovery':True,'save':lambda state:events.append(state),'subprocess':fake_sub,'sys':__import__('sys'),'here':HERE,'env':{},'observe':observe,'time':fake_time}
        with self.assertRaises(SystemExit) as done:
            exec(compile(ast.Module(body=[loop],type_ignores=[]),'watchdog-loop-cpu','exec'),env)
        self.assertEqual(done.exception.code,0)
        self.assertLess(events.index('recover'),events.index('observe'))
        self.assertEqual(events.count('recover'),2)
        self.assertIn('BASELINE_STOPPED_RECOVERY_RETRY',events)
        self.assertIn('BASELINE_HEALTHY_OBSERVED_5MIN',events)
    def test_watchdog_never_signals_process_and_heartbeat_is_irrelevant(self):
        s=(HERE/'watchdog.py').read_text()
        self.assertNotIn('killpg',s);self.assertNotIn('signal.',s);self.assertNotIn('heartbeat',s)
        self.assertIn("'engram_adapter_recover.py'",s)
        self.assertIn('DISABLED:',s)
if __name__=='__main__':unittest.main()
