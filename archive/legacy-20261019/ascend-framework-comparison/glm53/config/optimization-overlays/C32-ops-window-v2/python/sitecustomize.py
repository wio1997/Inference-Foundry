"""Task-scoped D-only hook; fail closed before worker startup on invalid configuration."""
import os
if os.environ.get('GLM53_DEEPEP_METADATA_BATCH')=='1':
 try:
  import hashlib,pathlib
  path=pathlib.Path(__file__).with_name('glm53_deepep_metadata_import_hook_v1.py')
  assert hashlib.sha256(path.read_bytes()).hexdigest()=='c4c429ea517c7ac726bf937cc3ab4c0db337c9e92666340af5b075dbec9b2bcb'
  import glm53_deepep_metadata_import_hook_v1 as hook
  hook.activate()
 except BaseException as error:
  os.write(2,('GLM53_METADATA_HOOK_FATAL='+repr(error)+'\n').encode());os._exit(86)

if os.environ.get('GLM53_C32_OPS_AUDIT')=='1':
 try:
  import hashlib,pathlib
  path=pathlib.Path(__file__).with_name('glm53_C32_ops_window_audit_v2.py')
  assert hashlib.sha256(path.read_bytes()).hexdigest()=='2e3ba957fab37574fd69ed76fdfeb71c3fe18acb08adb8d0407f4d83a7d24177'
  import glm53_C32_ops_window_audit_v2 as ops_audit
  ops_audit.activate()
 except BaseException as error:
  os.write(2,('GLM53_C32_OPS_HOOK_FATAL='+repr(error)+'\n').encode());os._exit(86)
