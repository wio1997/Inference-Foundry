"""Strict target+NextN registration using only the installed native classes.

Candidate only: no scheduler/model replacement and no runtime file mutation.
Environment supplies an explicit, pre-generated runtime SHA manifest.
"""
import hashlib,importlib,inspect,json,os
from pathlib import Path

root=Path(os.environ['GLM53_RUNTIME_ROOT']).resolve()
manifest=json.loads(Path(os.environ['GLM53_NATIVE_SHA_MANIFEST']).read_text())
classes=[]
for module_name,class_name in [
 ('sglang.srt.models.glm5_next','Glm5NextForConditionalGeneration'),
 ('sglang.srt.models.glm5_next_nextn','Glm5NextForConditionalGenerationNextN')]:
 module=importlib.import_module(module_name)
 cls=getattr(module,class_name);file=Path(inspect.getfile(cls)).resolve();rel=str(file.relative_to(root))
 if cls.__module__!=module_name or hashlib.sha256(file.read_bytes()).hexdigest()!=manifest[rel]:
  raise RuntimeError('Unexpected native model source: '+str(file))
 loader=cls.load_weights;ledger=root/'glm53_sglang_adapter/ledger.py'
 if not hasattr(loader,'__wrapped__') or Path(loader.__code__.co_filename).resolve()!=ledger:
  raise RuntimeError('Strict ledger wrapper missing: '+class_name)
 if hashlib.sha256(ledger.read_bytes()).hexdigest()!=manifest[str(ledger.relative_to(root))]:
  raise RuntimeError('Unexpected ledger implementation')
 classes.append(cls)
EntryClass=classes
