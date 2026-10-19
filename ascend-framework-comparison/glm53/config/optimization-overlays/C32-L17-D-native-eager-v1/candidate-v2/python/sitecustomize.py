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

if os.environ.get('GLM53_C32_TARGET_GRAPH_ALIGN')=='1':
 try:
  import hashlib,pathlib
  p=pathlib.Path(__file__).with_name('glm53_C32_target_graph_fullmesh_v2.py')
  assert hashlib.sha256(p.read_bytes()).hexdigest()=='feb69fcc0020acca1fa2c3ce2774aefa79483a79d8f5221c7c265bff8b37185b'
  import glm53_C32_target_graph_fullmesh_v2 as align
  align.activate()
 except BaseException as error:
  os.write(2,('GLM53_C32_ALIGNMENT_FATAL='+repr(error)+'\n').encode());os._exit(86)


if os.environ.get('GLM53_EAGER_INPUT_AUDIT')=='1':
 try:
  if os.environ.get('SGLANG_EAGER_INPUT_NO_COPY')!='1': raise RuntimeError('SGLANG_EAGER_INPUT_NO_COPY is not 1')
  if os.environ.get('GLM53_NODE') not in ('36','41'): raise RuntimeError('GLM53_NODE is not D36/D41')
  import hashlib,pathlib
  audit_path=pathlib.Path(__file__).with_name('glm53_C32_L17_eager_input_audit_v2.py')
  if hashlib.sha256(audit_path.read_bytes()).hexdigest()!='0df4276b5e5f8d10aec1faf25cd24e847dc3ef9a57570f70f257c0ebf539d56f': raise RuntimeError('L17 v2 audit module SHA mismatch')
  import glm53_C32_L17_eager_input_audit_v2 as eager_input_audit
  eager_input_audit.activate()
 except BaseException as error:
  os.write(2,('GLM53_EAGER_INPUT_AUDIT_FATAL='+repr(error)+'\n').encode());os._exit(86)
