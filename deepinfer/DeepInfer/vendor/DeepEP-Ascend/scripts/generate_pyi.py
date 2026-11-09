import re
from pathlib import Path
from typing import NamedTuple

import tree_sitter_cpp
from tree_sitter import Language, Node, Parser, Query, QueryCursor


class Param(NamedTuple):
    name: str
    type: str
    default: str | None = None


class Sig(NamedTuple):
    """A callable's signature: parameters, Python return type and docstring."""
    params: list[Param]
    ret: str
    doc: str | None = None


CPP = Language(tree_sitter_cpp.language())
PARSER = Parser(CPP)
EXTS = {'.cpp', '.cc', '.cxx', '.c', '.hpp', '.h', '.cuh'}

# tree-sitter pattern matches: function declarators, `constexpr auto` aliases,
# and `m.def(...)` / `m.attr(...)` pybind statements.
Q_FUNC = Query(CPP, '(function_declarator declarator: [(identifier) (field_identifier)] @name parameters: (parameter_list) @params)')
Q_ALIAS = Query(CPP, '(init_declarator declarator: (identifier) @name value: [(identifier) (template_function)] @value)')
Q_CALL = Query(CPP, '(call_expression function: (field_expression field: (field_identifier) @method) arguments: (argument_list) @args)')
Q_ATTR = Query(CPP, '(assignment_expression'
                    ' left: (call_expression arguments: (argument_list) @alias)'
                    ' right: (call_expression function: (field_expression field: (field_identifier) @rf) arguments: (argument_list) @target))')

# C++ template -> Python generic, built from the already-converted argument list.
TEMPLATE_MAP = {
    'optional': lambda args: f'Optional[{args[0]}]',
    'vector': lambda args: f'list[{args[0]}]',
    'pair': lambda args: f'tuple[{", ".join(args)}]',
    'tuple': lambda args: f'tuple[{", ".join(args)}]',
}
# C++ leaf type name (last token) -> Python type.
TYPE_MAP = {
    'void': 'None', 'bool': 'bool',
    'float': 'float', 'double': 'float',
    'char': 'str', 'string': 'str', 'Tensor': 'torch.Tensor',
    'int': 'int', 'long': 'int', 'short': 'int', 'unsigned': 'int', 'signed': 'int',
    'size_t': 'int', 'ssize_t': 'int', 'ptrdiff_t': 'int',
    'int8_t': 'int', 'int16_t': 'int', 'int32_t': 'int', 'int64_t': 'int',
    'uint8_t': 'int', 'uint16_t': 'int', 'uint32_t': 'int', 'uint64_t': 'int',
}


def matches(query: Query, node: Node) -> list[dict[str, list[Node]]]:
    return [m for _, m in QueryCursor(query).matches(node)]


def text(node: Node) -> str:
    return node.text.decode()


def last_ident(s: str) -> str:
    """`&deep_gemm::foo<int>` -> `foo`."""
    return re.sub(r'<.*', '', s.lstrip('&')).split('::')[-1].strip()


def first_string(arg_list: Node) -> str:
    """`("x", ...)` -> `x`."""
    return text(arg_list.named_children[0]).strip('"')


def convert_py_type(node: Node | None) -> str:
    """Map a C++ type node directly to a Python type string."""
    if node is None:
        return 'Any'
    if node.type == 'type_descriptor':
        return convert_py_type(node.child_by_field_name('type'))
    if node.type == 'qualified_identifier':
        return convert_py_type(node.named_children[-1])
    if node.type == 'template_type':
        name = last_ident(text(node.child_by_field_name('name')))
        args = [convert_py_type(c) for c in node.child_by_field_name('arguments').named_children]
        return TEMPLATE_MAP[name](args) if name in TEMPLATE_MAP else 'Any'
    return TYPE_MAP.get(text(node).split()[-1].split('::')[-1], 'Any')


def convert_py_value(node: Node) -> str:
    """Map a C++ default-value node to a Python literal string."""
    s = text(node).strip()
    if 'nullopt' in s or s in ('nullptr', 'NULL'):
        return 'None'
    if s in ('true', 'false'):
        return s.capitalize()
    if s.startswith('"'):
        return s
    m = re.search(r'make_tuple\s*\((.*)\)', s, re.S) or re.match(r'std::\w+\s*<[^>]*>\s*\(\s*\{(.*)\}\s*\)', s, re.S)
    if m:
        return f'({m.group(1).strip()})'
    if re.match(r'^[+-]?[\d.]', s):
        return s.rstrip('fF')
    return 'None'


def get_ident(decl: Node | None) -> str | None:
    if decl is None:
        return None
    if decl.type == 'identifier':
        return text(decl)
    return next((get_ident(c) for c in decl.children if get_ident(c)), None)


def get_params(param_list: Node) -> list[Param]:
    decls = [p for p in param_list.named_children
             if p.type in ('parameter_declaration', 'optional_parameter_declaration')]
    return [Param(get_ident(p.child_by_field_name('declarator')) or f'arg{i}',
                  convert_py_type(p.child_by_field_name('type')))
            for i, p in enumerate(decls)]


def get_doc(node: Node) -> str | None:
    while node.parent and node.parent.type not in ('translation_unit', 'declaration_list', 'compound_statement', 'namespace_definition'):
        node = node.parent
    comments, sib = [], node.prev_sibling
    while sib and sib.type == 'comment':
        comments.append(re.sub(r'^/[/*]+ ?|\s*\*+/$', '', text(sib)).rstrip())
        sib = sib.prev_sibling
    lines = list(reversed(comments))
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return '\n'.join(lines) or None


def _returns_value(node: Node) -> bool:
    """True if a lambda body contains a value-returning `return <expr>;`.

    Nested lambdas are skipped so their returns don't leak into the outer one.
    """
    if node.type == 'lambda_expression':
        return False
    if node.type == 'return_statement':
        return any(c.is_named for c in node.children)
    return any(_returns_value(c) for c in node.children)


def parse_lambda(node: Node) -> Sig:
    """Parse an inline `[&](...) {...}` lambda into a Sig."""
    decl = node.child_by_field_name('declarator')
    body = node.child_by_field_name('body')
    # The trailing return type (`-> T`) is an unnamed child of the declarator.
    trailing = next((c for c in decl.children if c.type == 'trailing_return_type'), None) if decl else None
    params = get_params(decl.child_by_field_name('parameters')) if decl else []
    if trailing:
        # Explicit `-> T` wins.
        ret = convert_py_type(trailing.named_children[0])
    elif body is not None and not _returns_value(body):
        # No value-returning `return` => the lambda is void.
        ret = 'None'
    else:
        ret = 'Any'
    return Sig(params, ret, get_doc(node))


def get_signature(target: Node, funcs: dict[str, Sig]) -> Sig:
    """The signature of an `m.def` callable: an inline lambda or a named C++ function."""
    if target.type == 'lambda_expression':
        return parse_lambda(target)
    return funcs.get(last_ident(text(target)), Sig([], 'Any'))


def parse_pyargs(args: list[Node]) -> tuple[list[Param], str | None]:
    """Parse trailing `py::arg("x")[= default]` / docstring arguments of a `.def`."""
    pyargs: list[Param] = []
    doc = None
    for a in args:
        if a.type == 'string_literal':                          # pybind docstring
            doc = text(a).strip('"')
        elif a.type == 'assignment_expression':                 # py::arg("x") = default
            name = first_string(a.child_by_field_name('left').child_by_field_name('arguments'))
            pyargs.append(Param(name, '', convert_py_value(a.child_by_field_name('right'))))
        elif a.type == 'call_expression':                       # py::arg("x")
            pyargs.append(Param(first_string(a.child_by_field_name('arguments')), ''))
    return pyargs, doc


def merge_pyargs(pyargs: list[Param], sig: Sig) -> list[Param]:
    """Names come from py::arg; types come from the C++ signature. Without py::arg the
    callable (e.g. a lambda) keeps its own parameter names."""
    if not pyargs:
        return list(sig.params)
    return [p._replace(type=sig.params[i].type if i < len(sig.params) else 'Any')
            for i, p in enumerate(pyargs)]


def build_binding(m: dict[str, list[Node]], funcs: dict[str, Sig]) -> Sig:
    """Turn one `m.def("name", callable, py::arg...)` match into the bound Sig."""
    args = m['args'][0].named_children
    sig = get_signature(args[1], funcs)
    pyargs, doc = parse_pyargs(args[2:])
    return Sig(merge_pyargs(pyargs, sig), sig.ret, doc or sig.doc)


def init_target(target: Node) -> Node | None:
    """Return the `py::init<...>` template_function node if `target` is a pybind constructor."""
    if target.type != 'call_expression':
        return None
    tf = find_template_function(target.child_by_field_name('function'))
    if tf is not None and last_ident(text(tf.child_by_field_name('name'))) == 'init':
        return tf
    return None


def build_method(m: dict[str, list[Node]], funcs: dict[str, Sig], cpp_name: str) -> tuple[str, Sig]:
    """Turn one class-level `.def("name", &Cls::method, ...)` or `.def(py::init<...>())`
    match into a `(python_name, Sig)` pair with an explicit leading `self`."""
    args = m['args'][0].named_children
    init = init_target(args[0])
    if init is not None:
        # `.def(py::init<...>())` has no name string: prefer the matching C++ constructor
        # (named parameters), else fall back to the template argument types with `argN`.
        ctor = funcs.get(cpp_name)
        targs = init.child_by_field_name('arguments')
        types = [convert_py_type(c) for c in targs.named_children] if targs else []
        if ctor is not None and len(ctor.params) == len(types):
            params = list(ctor.params)
        else:
            params = [Param(f'arg{i}', t) for i, t in enumerate(types)]
        pyargs, doc = parse_pyargs(args[1:])
        params = merge_pyargs(pyargs, Sig(params, 'None')) if pyargs else params
        return '__init__', Sig([Param('self', '')] + params, 'None', doc)

    name = first_string(m['args'][0])
    sig = get_signature(args[1], funcs)
    pyargs, doc = parse_pyargs(args[2:])
    return name, Sig([Param('self', '')] + merge_pyargs(pyargs, sig), sig.ret, doc or sig.doc)


def chain_root(call: Node) -> Node | None:
    """Follow a `obj.def(...).def(...)` method chain down to its receiver object.

    Returns the module identifier (`m`) for a free function, or the
    `py::class_<...>(m, "Name")` constructor call for a bound method.
    """
    node = call
    while True:
        fn = node.child_by_field_name('function')
        if fn is None or fn.type != 'field_expression':
            return None
        obj = fn.child_by_field_name('argument')
        inner = obj.child_by_field_name('function') if obj.type == 'call_expression' else None
        if inner is not None and inner.type == 'field_expression':
            node = obj                                          # keep walking up the chain
            continue
        return obj


def find_template_function(node: Node | None) -> Node | None:
    """Locate the `template_function` node inside a (possibly qualified) callee."""
    if node is None or node.type == 'template_function':
        return node
    return next((r for c in node.named_children if (r := find_template_function(c))), None)


def class_base_info(base: Node | None) -> tuple[str, str] | None:
    """`py::class_<deep_ep::Foo>(m, "Foo")` -> `("Foo", "Foo")` (python name, C++ name)."""
    if base is None or base.type != 'call_expression':
        return None
    tf = find_template_function(base.child_by_field_name('function'))
    if tf is None or last_ident(text(tf.child_by_field_name('name'))) != 'class_':
        return None
    cpp = last_ident(text(tf.child_by_field_name('arguments').named_children[0]))
    py = next((text(a).strip('"') for a in base.child_by_field_name('arguments').named_children
               if a.type == 'string_literal'), cpp)
    return py, cpp


def index_functions(roots: list[Node]) -> dict[str, Sig]:
    """All C++ functions by name, with `constexpr auto` aliases flattened in."""
    funcs: dict[str, Sig] = {}
    aliases: dict[str, str] = {}
    for root in roots:
        for m in matches(Q_FUNC, root):
            defn = m['name'][0]
            while defn is not None and defn.type not in ('function_definition', 'declaration', 'field_declaration'):
                defn = defn.parent
            if defn is None:
                continue
            funcs.setdefault(text(m['name'][0]), Sig(get_params(m['params'][0]),
                                                      convert_py_type(defn.child_by_field_name('type')), get_doc(defn)))
        for m in matches(Q_ALIAS, root):
            aliases[text(m['name'][0])] = last_ident(text(m['value'][0]))
    for alias, target in aliases.items():
        while target in aliases and target not in funcs:
            target = aliases[target]
        if target in funcs:
            funcs.setdefault(alias, funcs[target])
    return funcs


def render_callable(name: str, sig: Sig, indent: str = '') -> str:
    """Render a single `def` stub (optionally indented for a class body)."""
    lines = []
    for p in sig.params:
        ann = f': {p.type}' if p.type else ''
        default = f' = {p.default}' if p.default is not None else ''
        lines.append(f'{indent}    {p.name}{ann}{default}')
    body = '\n' + ',\n'.join(lines) + f'\n{indent}' if lines else ''
    head = f'{indent}def {name}({body}) -> {sig.ret}:'
    if not sig.doc:
        return f'{head} ...'
    if '\n' in sig.doc:
        doc = '\n'.join(f'{indent}    {ln}'.rstrip() for ln in sig.doc.split('\n'))
        return f'{head}\n{indent}    """\n{doc}\n{indent}    """\n{indent}    ...'
    return f'{head}\n{indent}    """{sig.doc}"""\n{indent}    ...'


def render_class(py_name: str, members: list[tuple[str, Sig]]) -> str:
    if not members:
        return f'class {py_name}: ...'
    body = '\n\n'.join(render_callable(name, sig, indent='    ') for name, sig in members)
    return f'class {py_name}:\n{body}'


def generate_pyi_file(name: str, root: str, output_dir: str = '.') -> None:
    roots = [PARSER.parse(p.read_bytes()).root_node
             for p in sorted(Path(root).rglob('*')) if p.is_file() and p.suffix.lower() in EXTS]
    funcs = index_functions(roots)

    # `m.def` -> module function stub; `py::class_<T>(m, "X").def(...)` -> class + methods;
    # `m.attr(alias) = m.attr(target)` clones a module binding.
    bindings: dict[str, Sig] = {}
    classes: dict[str, list[tuple[str, Sig]]] = {}
    attrs: dict[str, str] = {}
    for root_node in roots:
        for m in matches(Q_CALL, root_node):
            if m['method'][0].text != b'def':
                continue
            info = class_base_info(chain_root(m['args'][0].parent))
            if info is not None:
                py_name, cpp_name = info
                classes.setdefault(py_name, []).append(build_method(m, funcs, cpp_name))
            else:
                bindings[first_string(m['args'][0])] = build_binding(m, funcs)
        for m in matches(Q_ATTR, root_node):
            if m['rf'][0].text == b'attr':
                attrs[first_string(m['alias'][0])] = first_string(m['target'][0])
    for alias, target in attrs.items():
        while target in attrs and target not in bindings:
            target = attrs[target]
        if target in bindings and alias not in bindings:
            bindings[alias] = bindings[target]

    stubs = [render_class(py_name, members) for py_name, members in classes.items()]
    stubs += [render_callable(fn_name, f) for fn_name, f in bindings.items()]

    content = (f'# Stubs for module: {name}\n\n'
               'from typing import Any, Optional\n'
               'import torch\n\n\n' + '\n\n\n'.join(stubs) + '\n')

    output_path = Path(output_dir) / f'{name}.pyi'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding='utf-8')
    print(f'.pyi file generated: {output_path}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate a .pyi stub from pybind11 sources.')
    parser.add_argument('--name', default='_C')
    parser.add_argument('--root', default='./csrc')
    parser.add_argument('--output-dir', default='./deep_ep')
    args = parser.parse_args()
    generate_pyi_file(name=args.name, root=args.root, output_dir=args.output_dir)
