"""Static structural guard for the archived AISBench base API source.

This checks AST structure only; it does not import AISBench or prove live behavior.
"""
import ast
from pathlib import Path


class SourceStructureError(ValueError):
    """The archived source lacks the required retry structure."""


def _is_self_call(node, method):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == 'self'
            and node.func.attr == method)


def validate_source(path):
    path = Path(path)
    try:
        tree = ast.parse(path.read_text(encoding='utf-8'))
    except (UnicodeError, SyntaxError) as exc:
        raise SourceStructureError('source is not valid Python text') from exc
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BaseAPIModel']
    if len(classes) != 1:
        raise SourceStructureError('expected exactly one top-level BaseAPIModel class')
    methods = [n for n in classes[0].body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'generate']
    if len(methods) != 1:
        raise SourceStructureError('expected exactly one async BaseAPIModel.generate method')
    loops = []
    for n in methods[0].body:
        if not isinstance(n, ast.For) or not isinstance(n.iter, ast.Call):
            continue
        call = n.iter
        if not (isinstance(call.func, ast.Name) and call.func.id == 'range'
                and len(call.args) == 1 and not call.keywords):
            continue
        arg = call.args[0]
        if (isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name)
                and arg.value.id == 'self' and arg.attr == 'retry'):
            loops.append(n)
    if len(loops) != 1:
        raise SourceStructureError('generate must contain one direct for-loop over range(self.retry)')
    tries = [n for n in loops[0].body if isinstance(n, ast.Try)]
    if len(tries) != 1:
        raise SourceStructureError('retry loop must directly contain one try block')
    checked = tries[0]
    calls = {method for method in ('stream_infer', 'text_infer')
             if any(isinstance(n, ast.Await) and _is_self_call(n.value, method)
                    for stmt in checked.body for n in ast.walk(stmt))}
    handles_exception = any(isinstance(h.type, ast.Name) and h.type.id == 'Exception'
                            for h in checked.handlers)
    has_success_break = any(isinstance(n, ast.Break) for stmt in checked.body for n in ast.walk(stmt))
    if calls != {'stream_infer', 'text_infer'} or not handles_exception or not has_success_break:
        raise SourceStructureError('retry try block lacks expected infer calls, error handling, or success break')
    return {'class': 'BaseAPIModel', 'method': 'async generate',
            'retry_iterator': 'range(self.retry)', 'checks': 'AST structure only'}
