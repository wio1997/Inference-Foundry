"""Actual SG67 cleanup function over synchronous CPU-only framework stubs."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json, os, tempfile, unittest
import sg_layer_event_anchor as m

class CleanupTests(unittest.TestCase):
 def run_cleanup(self, mismatch=False, drain_failure=False):
  log=[];rows=[]
  class Graph:
   def reset(self):log.append('graph_reset')
  graph=Graph();backend=SimpleNamespace(_device_id=3,_graphs={'bucket1':graph},_outputs={'bucket1':'held'},
   _update_executor=SimpleNamespace(shutdown=lambda **kw:log.append(('shutdown',kw))))
  key=(id(backend),id(graph));m._GRAPH_EVENT_HOLDS[key]={'graph':graph}
  class ImmediateThread:
   def __init__(self, target, **kwargs):self.target=target
   def start(self):self.target()
  def emit(event,**kw):rows.append((event,kw));log.append(event)
  def outer_drain(**kw):
   log.append('outer_drain');return {'status':'drain_failed_references_retained' if drain_failure else 'drained'}
  def outer_release(g,*,reset_receipt):
   assert g is graph and reset_receipt['graph_python_id']==id(graph)
   assert reset_receipt['device_drained_before_and_after_reset']
   assert 'bucket1'not in backend._graphs
   log.append('outer_release');return {'outer_event_count_released':2}
  npu=SimpleNamespace(set_device=lambda d:log.append(('device',d)),synchronize=lambda:log.append('sync'))
  with tempfile.TemporaryDirectory()as temp:
   path=Path(temp)/'request.json';path.write_text(json.dumps({'task_id':'wrong'if mismatch else 'Q27-FULL-MAP-20261025','worker_pids':[os.getpid()]}))
   with patch.dict(os.environ,{'Q27_EVENT_CLEANUP_REQUEST':str(path)}),patch.object(m.threading,'Thread',ImmediateThread),patch.dict('sys.modules',{
    'torch':SimpleNamespace(npu=npu),'sg_replay_outer_events':SimpleNamespace(drain_outer_replay_anchors=outer_drain,release_drained_outer_events=outer_release)}):
    m.arm_owned_graph_cleanup(backend,graph,emit)
  return log,rows,backend,key
 def tearDown(self):m._GRAPH_EVENT_HOLDS.clear()
 def test_success_orders_drains_reset_release_and_forbids_further_serving(self):
  log,rows,b,key=self.run_cleanup()
  self.assertEqual(log,[('device',3),('shutdown',{'wait':True,'cancel_futures':True}),'sync','outer_drain',
   'sg_replay_outer_summary','graph_reset','sync','outer_release','sync','layer_public_event_cleanup'])
  self.assertNotIn(key,m._GRAPH_EVENT_HOLDS);self.assertEqual(b._graphs,{});self.assertEqual(b._outputs,{})
  self.assertFalse(rows[-1][1]['serving_after_cleanup_allowed']);self.assertEqual(rows[-1][1]['events_released'],128)
 def test_wrong_task_identity_preserves_graph_holds_without_device_actions(self):
  log,rows,b,key=self.run_cleanup(mismatch=True)
  self.assertEqual(log,['layer_public_event_cleanup']);self.assertIn(key,m._GRAPH_EVENT_HOLDS)
  self.assertIn('bucket1',b._graphs);self.assertEqual(rows[-1][1]['status'],'failed_preserve_holds')
 def test_failed_outer_drain_preserves_graph_and_prevents_reset_release(self):
  log,rows,b,key=self.run_cleanup(drain_failure=True)
  self.assertNotIn('graph_reset',log);self.assertNotIn('outer_release',log)
  self.assertIn(key,m._GRAPH_EVENT_HOLDS);self.assertIn('bucket1',b._graphs)
  self.assertEqual(rows[-1][1]['status'],'failed_preserve_holds')

if __name__=='__main__':unittest.main()
