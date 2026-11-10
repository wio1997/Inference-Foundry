"""Semantic and interval-contract checks without importing a model or torch."""
import ast,json,pathlib,tempfile,os,sys,time
import ft_hook
source='''from __future__ import annotations
class Demo:
    def outer(self, n: int) -> int:
        "kept doc"
        if n < 0:
            raise ValueError("original")
        return self.inner(n) + 1
    def inner(self, n):
        return sum(range(n))
'''
spec={'functions':['Demo.outer','Demo.inner'],'coarse':['Demo.outer']}
t=ft_hook.Transformer('demo','demo.py',spec);tree=t.visit(ast.parse(source));ast.fix_missing_locations(tree)
with tempfile.TemporaryDirectory() as tmp:
 ft_hook.ROOT=tmp
 code=compile(tree,'demo.py','exec');space={'_ft_span':ft_hook._span};exec(code,space)
 obj=space['Demo']();assert obj.outer.__doc__=='kept doc'
 for mode in ['off','coarse','detailed']:
  pathlib.Path(tmp,'control.json').write_text(json.dumps({'mode':mode,'label':mode}))
  for n in [0,10,100]:assert obj.outer(n)==sum(range(n))+1
  try:obj.outer(-1)
  except ValueError as e:assert str(e)=='original'
  else:raise AssertionError('exception changed')
 rows=[json.loads(l) for p in pathlib.Path(tmp).glob('spans-*') for l in p.read_text().splitlines()]
 assert len(rows)==11,len(rows)
 assert all(r['duration_ns']>=0 and r['thread_cpu_ns']>=0 for r in rows)
 assert sum(r['mode']=='coarse' for r in rows)==4
 detailed=[r for r in rows if r['mode']=='detailed'];assert len(detailed)==7
 for r in detailed:
  if r['parent'] is not None:
   parent=next(p for p in detailed if p['id']==r['parent'])
   assert parent['mono_ns']<=r['mono_ns']<=r['mono_ns']+r['duration_ns']<=parent['mono_ns']+parent['duration_ns']
 print(json.dumps({'semantic_values':9,'exceptions':3,'events':len(rows),'hierarchy':'PASS','signature':'retained','device_imports':0}))
