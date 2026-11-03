"""Turn security assertions into explicit checks in disabled adapter copies.

CPU/local only. No command dispatch. The build fails if the installed local
Python cannot parse a generated executable fragment or transform its asserts.
"""
import ast
from pathlib import Path
import sys
import textwrap

if sys.flags.optimize:raise RuntimeError('optimized Python forbidden during hardening')

HERE=Path(__file__).resolve().parent
PROGRAM_NAMES={'REMOTE','HEALTH','STATUS','CAPTURE','SUMMARY','RESTORE','prepare','health','check','code'}
OPT_GUARD="import sys\nif sys.flags.optimize:\n    raise RuntimeError('optimized Python is forbidden for diagnostic safety checks')\n"


class ExplicitChecks(ast.NodeTransformer):
    def visit_Assert(self,n):
        self.generic_visit(n)
        args=[] if n.msg is None else [n.msg]
        return ast.copy_location(ast.If(test=ast.UnaryOp(op=ast.Not(),operand=n.test),
            body=[ast.Raise(exc=ast.Call(func=ast.Name(id='AssertionError',ctx=ast.Load()),args=args,keywords=[]),cause=None)],orelse=[]),n)

    def visit_Assign(self,n):
        self.generic_visit(n)
        names={x.id for x in n.targets if isinstance(x,ast.Name)}
        if names & PROGRAM_NAMES and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str):
            n.value=ast.copy_location(ast.Constant(value=harden_program(n.value.value)),n.value)
        return n

    def visit_Constant(self,n):
        # Recovery identity readback is constructed from a list of source lines.
        if isinstance(n.value,str) and n.value.lstrip().startswith('assert '):
            prefix=n.value[:len(n.value)-len(n.value.lstrip())]
            transformed=harden_program(textwrap.dedent(n.value),guard=False).rstrip('\n')
            return ast.copy_location(ast.Constant(value=textwrap.indent(transformed,prefix)),n)
        return n


def harden_program(source,guard=True):
    t=ast.parse(source)
    t=ExplicitChecks().visit(t);ast.fix_missing_locations(t)
    out=ast.unparse(t)+'\n'
    if guard:out=OPT_GUARD+out
    compile(out,'hardened-remote','exec')
    if any(isinstance(n,ast.Assert) for n in ast.walk(ast.parse(out))):raise RuntimeError('unconverted assertion')
    return out


def harden_source(source,path='<generated>'):
    tree=ast.parse(source);lines=source.splitlines(keepends=True)
    if source.startswith(OPT_GUARD):raise RuntimeError('adapter already hardened')
    offsets=[0]
    for line in lines:offsets.append(offsets[-1]+len(line))
    updates=[]
    for n in ast.walk(tree):
        if isinstance(n,ast.Assert):
            msg='' if n.msg is None else ast.unparse(n.msg)
            code='if not ('+ast.unparse(n.test)+'): raise AssertionError('+msg+')'
            updates.append((offsets[n.lineno-1]+n.col_offset,offsets[n.end_lineno-1]+n.end_col_offset,code))
        elif isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in PROGRAM_NAMES for t in n.targets) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str):
            v=n.value;updates.append((offsets[v.lineno-1]+v.col_offset,offsets[v.end_lineno-1]+v.end_col_offset,repr(harden_program(v.value))))
    for a,b,s in sorted(updates,reverse=True):source=source[:a]+s+source[b:]
    source=OPT_GUARD+source
    compile(source,str(path),'exec')
    if any(isinstance(n,ast.Assert) for n in ast.walk(ast.parse(source))):raise RuntimeError('outer assertion left')
    return source


if __name__=='__main__':
    raise SystemExit('Use generate_adapter_drafts.py; direct in-place hardening is disabled.')
