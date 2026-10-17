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
  path=pathlib.Path(__file__).with_name('glm53_C32_ops_audit_v1.py')
  assert hashlib.sha256(path.read_bytes()).hexdigest()=='a8bb32d4cde5b3159c606b0bc37333c5819b85e1a225aec07a4ffe41563c462f'
  import glm53_C32_ops_audit_v1 as ops_audit
  ops_audit.activate()
 except BaseException as error:
  os.write(2,('GLM53_C32_OPS_HOOK_FATAL='+repr(error)+'\n').encode());os._exit(86)
