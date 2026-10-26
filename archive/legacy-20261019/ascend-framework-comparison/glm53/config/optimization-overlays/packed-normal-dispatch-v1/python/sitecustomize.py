"""Task-owned packed D lazy hook; legacy L04 branch remains available."""
import os
if os.environ.get('GLM53_PACKED_DISPATCH')=='1':
 try:
  import hashlib,pathlib
  assert os.environ.get('GLM53_DEEPEP_METADATA_BATCH')=='1'
  path=pathlib.Path(__file__).with_name('glm53_deepep_packed_import_hook_v1.py')
  assert hashlib.sha256(path.read_bytes()).hexdigest()=='64bfa11e04920ab1ed6b945d359848fed503599094d88e0037317fb0a458bfc5'
  import glm53_deepep_packed_import_hook_v1 as hook
  hook.activate()
 except BaseException as error:
  os.write(2,('GLM53_PACKED_HOOK_FATAL='+repr(error)+'\n').encode());os._exit(86)
else:
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
