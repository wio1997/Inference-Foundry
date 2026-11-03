"""CPU-only reservation mock; Popen is intercepted and never starts a process."""
from pathlib import Path
import contextlib,io,json,runpy,shutil,sys,tempfile
from unittest import mock
src=Path(__file__).resolve().parent/'launch_second_background.py'
with tempfile.TemporaryDirectory() as d:
 p=Path(d)/src.name;shutil.copyfile(src,p)
 class Child:pid=123456
 calls=[]
 def fake_popen(*a,**kw):
  state=json.loads((Path(d)/'controller-reservation.json').read_text());assert state['pid'] is None and state['state']=='STARTING_VERIFY_BEFORE_MANUAL_RETRY'
  calls.append((a,kw));return Child()
 old=sys.argv;sys.argv=[str(p),'--execute']
 try:
  with mock.patch('subprocess.Popen',side_effect=fake_popen),contextlib.redirect_stdout(io.StringIO()):
   runpy.run_path(str(p),run_name='__main__')
   state=json.loads((Path(d)/'controller-reservation.json').read_text());assert state['pid']==123456
   try:runpy.run_path(str(p),run_name='__main__')
   except SystemExit as e:assert 'already exists' in str(e)
   else:raise AssertionError('duplicate controller allowed')
 finally:sys.argv=old
 assert len(calls)==1
print(json.dumps({'status':'PASS','duplicate_launch':'refused','reservation_before_popen':'PASS','scope':'CPU mock only'}))
