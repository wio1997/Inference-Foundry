"""Offline full-class streaming identity candidate; other original bodies retained."""
from L036_streaming_components_v2 import *
OriginalSessionSlot=ns['SessionSlot']
p=R/'candidates/L036-streaming-slot-identity/streaming_session.py'
root=ast.parse(p.read_text());nodes=[x for x in root.body if isinstance(x,ast.ClassDef) and x.name in ('SessionSlot','StreamingSession')]
assert len(nodes)==2
for node in nodes:records.append({'path':str(p.relative_to(R)),'symbol':node.name,'AST_sha256':__import__('hashlib').sha256(ast.dump(node,include_attributes=False).encode()).hexdigest(),'candidate':True,'whole_candidate_class':True})
module=ast.Module(body=[ast.parse('from __future__ import annotations').body[0]]+nodes,type_ignores=[]);ast.fix_missing_locations(module);exec(compile(module,str(p),'exec'),ns)
BOUNDARIES_L036=BOUNDARIES_L036+['Offline candidate full SessionSlot/StreamingSession classes: identity tuple saved onfirstfinish; changed or unknownlegacy identity releases oldslot before raw fallback. No rawtree/storage generation fix, no production application.']
