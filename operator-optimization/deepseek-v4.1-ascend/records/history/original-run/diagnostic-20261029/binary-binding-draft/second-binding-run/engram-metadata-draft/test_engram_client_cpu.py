"""Test local gate and frozen client control flow using fake HTTP responses."""
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

os.environ['WIO_ENGRAM_RUN_ID']='a'*32
c=importlib.import_module('engram_client')
HERE=Path(__file__).resolve().parent


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.p=Path(self.tmp.name)
        self.input=self.p/'inputs';folder=self.input/'measure1';folder.mkdir(parents=True)
        (folder/'warmup.json').write_text(json.dumps([{'prompt':'seed'}]*8))
        (folder/'sample.requests.jsonl').write_text(''.join(json.dumps({'question':'private-prompt-'+str(i)})+'\n' for i in range(8)))
        (folder/'frozen-vllm-details.jsonl').write_text(''.join(json.dumps({'data_id':i,'prediction':'private-answer'})+'\n' for i in range(8)))
        self.meta=self.p/'metadata';self.meta.mkdir()
        self.patch1=mock.patch.object(c,'INPUTS',self.input);self.patch1.start()
        self.patch2=mock.patch.object(c,'GATE',self.meta/'PROFILE_ACTIVE');self.patch2.start()
        hashes={'seed':folder/'warmup.json','requests':folder/'sample.requests.jsonl','references':folder/'frozen-vllm-details.jsonl'}
        self.patch3=mock.patch.object(c,'FROZEN_FILES',{'measure1':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in hashes.items()}});self.patch3.start()
    def tearDown(self):self.patch3.stop();self.patch2.stop();self.patch1.stop();self.tmp.cleanup()
    def test_gate_exact_identity_and_safe_foreign_refusal(self):
        c.gate_open('b'*32)
        with self.assertRaises(RuntimeError):c.gate_close('c'*32)
        self.assertTrue(c.GATE.exists())
        c.gate_close('b'*32);self.assertFalse(c.GATE.exists())
    def test_gate_refuses_existing_foreign_marker(self):
        c.GATE.write_text('foreign')
        with self.assertRaises(RuntimeError):c.gate_open('b'*32)
        self.assertEqual(c.GATE.read_text(),'foreign')
    def test_measurement_gate_after_seed_and_no_text_in_summary(self):
        def parallel(rows,phase,stream=False):
            if phase.endswith('-seed'):
                self.assertFalse(c.GATE.exists())
                return [{'usage':{'prompt_tokens':29696,'completion_tokens':1},'prediction':'private-answer'}]*8
            self.assertTrue(c.GATE.exists());self.assertTrue(stream)
            gate=json.loads(c.GATE.read_text());self.assertEqual(gate['run_id'],'a'*32);self.assertEqual(gate['window_id'],'b'*32)
            return [{'usage':{'prompt_tokens':32768,'completion_tokens':1024},'prediction':'private-answer'}]*8
        delta={x:0.0 for x in c.METRICS};delta.update(prompt_tokens_total=262144,generation_tokens_total=8192,
          request_success_total=8,prefix_cache_queries_total=262144,prefix_cache_hits_total=235904)
        with mock.patch.object(c,'parallel',side_effect=parallel),mock.patch.object(c,'metrics',side_effect=[{k:0 for k in c.METRICS},delta]):
            out=c.phase('measure1','b'*32)
        self.assertFalse(c.GATE.exists());self.assertEqual(out['exact_frozen_prediction_matches'],8)
        self.assertNotIn('private-',json.dumps(out))
    def test_failed_requests_close_gate(self):
        def parallel(rows,phase,stream=False):
            if phase.endswith('-seed'):return [{'usage':{'prompt_tokens':29696,'completion_tokens':1}}]*8
            self.assertTrue(c.GATE.exists());raise RuntimeError('request failure')
        with mock.patch.object(c,'parallel',side_effect=parallel),mock.patch.object(c,'metrics',return_value={}):
            with self.assertRaises(RuntimeError):c.phase('measure1','b'*32)
        self.assertFalse(c.GATE.exists())
    def test_profiler_endpoints_absent(self):
        source=(HERE/'engram_client.py').read_text()
        self.assertNotIn('/start_profile',source);self.assertNotIn('/stop_profile',source)


class StaticAdapterTests(unittest.TestCase):
    def test_all_copies_disabled_and_unique_paths(self):
        for p in HERE.glob('engram_adapter_*.py'):
            source=p.read_text();compile(source,str(p),'exec')
            self.assertIn('DISABLED: review the independent adapter',source)
            self.assertNotIn('REPLACEME',source)
    def test_remote_guard_and_absence_of_old_injection(self):
        start=(HERE/'engram_adapter_start.py').read_text()
        sup=(HERE/'engram_adapter_supervise.py').read_text()
        recover=(HERE/'engram_adapter_recover.py').read_text()
        self.assertIn("HERE.parent/'remote_guard.py'",start)
        self.assertIn("HERE.parent/'remote_guard.py'",sup)
        self.assertIn("HERE.parent/'remote_guard.py'",recover)
        for s in (start,sup,recover):self.assertNotIn('binding-diagnostic-selective2',s)
        self.assertNotIn('profile_client.py',sup);self.assertIn('engram_client.py',sup)
        self.assertIn('OTHER_NPU_PROCESSES_DURING_CAPTURE',sup)
        self.assertIn('OTHER_NPU_PROCESSES_DURING_LOAD',sup)
        self.assertIn('LOCK_ROOT',recover)
    def test_embedded_remote_python_compiles(self):
        for file, names in [('engram_adapter_start.py',('REMOTE',)),('engram_adapter_supervise.py',('STATUS','CAPTURE','SUMMARY','RESTORE')),('engram_adapter_recover.py',('REMOTE','HEALTH'))]:
            tree=ast.parse((HERE/file).read_text())
            for name in names:
                nodes=[x for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in x.targets) and isinstance(x.value,ast.Constant)]
                self.assertEqual(len(nodes),1,(file,name))
                fragment=ast.literal_eval(nodes[0].value)
                compile(fragment,file+':'+name,'exec')
                if file=='engram_adapter_start.py' and name=='REMOTE':
                    inner=ast.parse(fragment)
                    prep=next(x for x in inner.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='prepare' for t in x.targets))
                    compile(ast.literal_eval(prep.value),'stage-prepare','exec')
    def test_execute_flags_stop_before_remote_or_local_mutation(self):
        for p in HERE.glob('engram_adapter_*.py'):
            args=[sys.executable,str(p),'--execute']
            if p.name=='engram_adapter_watchdog.py':args+=['a'*32,str(os.getpid())]
            result=subprocess.run(args,capture_output=True,text=True,timeout=5)
            self.assertNotEqual(result.returncode,0,p.name)
            self.assertIn('DISABLED:',result.stdout+result.stderr,p.name)
    def test_optimized_python_refused_before_remote_fragments(self):
        for file, names in [('engram_adapter_start.py',('REMOTE',)),('engram_adapter_supervise.py',('STATUS','CAPTURE','SUMMARY','RESTORE')),('engram_adapter_recover.py',('REMOTE','HEALTH'))]:
            tree=ast.parse((HERE/file).read_text())
            for name in names:
                n=next(x for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in x.targets) and isinstance(x.value,ast.Constant))
                code=ast.literal_eval(n.value)
                self.assertFalse(any(isinstance(x,ast.Assert) for x in ast.walk(ast.parse(code))),(file,name))
                for optimized in ('flag','environment'):
                    env=dict(os.environ);args=[sys.executable]
                    if optimized=='flag':args.append('-O')
                    else:env['PYTHONOPTIMIZE']='1'
                    r=subprocess.run(args+['-c',code],env=env,capture_output=True,text=True,timeout=5)
                    self.assertNotEqual(r.returncode,0,(file,name,optimized))
                    self.assertIn('optimized Python is forbidden',r.stderr,(file,name,optimized))


if __name__=='__main__':unittest.main()
