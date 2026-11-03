"""Fail-closed install and output-path tests, CPU only."""
import json, os, sys, tempfile
from pathlib import Path
import audited_observer as obs
if sys.flags.optimize:raise RuntimeError('tests require assertions enabled')
tempfile.tempdir=os.path.realpath(tempfile.gettempdir())
os.environ['WIO_ENGRAM_META_ENABLE']='1'
class Subject:
 def embed_gathered(self,gathered,num_tokens):return gathered
original=Subject.__dict__['embed_gathered'];real=obs._observer
try:
 for replacement in (lambda:None, lambda:(_ for _ in ()).throw(RuntimeError('forced init'))):
  obs._observer=replacement
  assert not obs.install(Subject,{'status':'verified_before_install'})
  assert Subject.__dict__['embed_gathered'] is original
 class BadStart:
  worker=None;loss=False
  def __init__(self):
   import threading
   self.stop=threading.Event()
  def start(self,verified):raise RuntimeError('forced post-patch start')
 bad=BadStart();obs._observer=lambda:bad
 assert not obs.install(Subject,{'status':'verified_before_install'})
 assert Subject.__dict__['embed_gathered'] is original and bad.loss and bad.stop.is_set() and Subject not in obs._INSTALLED
finally:obs._observer=real
with tempfile.TemporaryDirectory() as td:
 root=Path(td);private=root/'private';private.mkdir(mode=0o700);path=private/'engram-metadata.jsonl'
 assert obs._private_directory(str(private),str(path))
 path.write_text('');path.chmod(0o644);assert not obs._private_directory(str(private),str(path))
 path.chmod(0o600);assert obs._private_directory(str(private),str(path))
 path.unlink();path.symlink_to(root/'elsewhere');assert not obs._private_directory(str(private),str(path))
 path.unlink();alias=root/'alias';alias.symlink_to(private,target_is_directory=True);assert not obs._private_directory(str(alias),str(alias/'engram-metadata.jsonl'))
 private.chmod(0o755);assert not obs._private_directory(str(private),str(path))
print(json.dumps({'status':'PASS','observer_init_failure_rollback':'PASS','post_patch_start_failure_rollback':'PASS','symlink_and_broad_permissions':'rejected','assertions_enabled':True}))
