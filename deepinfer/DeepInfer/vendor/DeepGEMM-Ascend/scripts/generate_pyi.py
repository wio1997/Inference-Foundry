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
Q_FUNC = Query(CPP, '(function_declarator declarator: (identifier) @name parameters: (parameter_list) @params)')
Q_ALIAS = Query(CPP, '(init_declarator declarator: (identifier) @name value: [(identifier) (template_function)] @value)')
Q_CALL = Query(CPP, '(call_expression function: (field_expression argument: (_) @object field: (field_identifier) @method)'
                    ' arguments: (argument_list) @args)')
Q_ATTR = Query(CPP, '(assignment_expression'
                    ' left: (call_expression function: (field_expression argument: (_) @alias_object field: (field_identifier) @lf)'
                    ' arguments: (argument_list) @alias)'
                    ' right: (call_expression function: (field_expression argument: (_) @target_object field: (field_identifier) @rf)'
                    ' arguments: (argument_list) @target))')
MODULE_TYPES = {'pybind11::module_', 'py::module_'}

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
    args = [c for c in arg_list.named_children if c.type != 'comment']
    if not args or args[0].type != 'string_literal':
        raise ValueError(f'Cannot resolve binding name at line {arg_list.start_point.row + 1}: {text(arg_list)}')
    return text(args[0]).strip('"')


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


def is_module_object(node: Node) -> bool:
    """Resolve a receiver from module parameters and local aliases, respecting shadowing."""
    if node.type == 'parenthesized_expression':
        return is_module_object(node.named_children[0])
    if node.type == 'call_expression':
        function = node.child_by_field_name('function')
        return (function.type == 'field_expression' and text(function.child_by_field_name('field')) == 'def'
                and is_module_object(function.child_by_field_name('argument')))
    if node.type != 'identifier':
        return False

    name = text(node)
    while node.parent:
        scope = node.parent
        if scope.type == 'compound_statement':
            for statement in reversed(scope.named_children):
                if statement.end_byte > node.start_byte or statement.type != 'declaration':
                    continue
                for decl in statement.children_by_field_name('declarator'):
                    if get_ident(decl.child_by_field_name('declarator') or decl) != name:
                        continue
                    dtype = text(statement.child_by_field_name('type'))
                    value = decl.child_by_field_name('value')
                    return dtype in MODULE_TYPES or (dtype == 'auto' and value is not None and is_module_object(value))
        if scope.type in ('function_definition', 'lambda_expression'):
            declarator = scope.child_by_field_name('declarator')
            params = declarator.child_by_field_name('parameters') if declarator else None
            for param in params.named_children if params else []:
                if get_ident(param.child_by_field_name('declarator')) == name:
                    return text(param.child_by_field_name('type')) in MODULE_TYPES
            if scope.type == 'function_definition':
                return False
        node = scope
    return False


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


def build_binding(m: dict[str, list[Node]], funcs: dict[str, Sig]) -> Sig:
    """Turn one `m.def("name", callable, py::arg...)` match into the bound Sig."""
    args = m['args'][0].named_children
    sig = get_signature(args[1], funcs)

    pyargs: list[Param] = []
    doc = None
    for a in args[2:]:
        if a.type == 'string_literal':                          # pybind docstring
            doc = text(a).strip('"')
        elif text(a).startswith('py::kw_only'):
            pyargs.append(Param('*', ''))
        elif a.type == 'assignment_expression':                 # py::arg("x") = default
            name = first_string(a.child_by_field_name('left').child_by_field_name('arguments'))
            pyargs.append(Param(name, '', convert_py_value(a.child_by_field_name('right'))))
        elif text(a).startswith('py::arg'):                     # py::arg("x")
            pyargs.append(Param(first_string(a.child_by_field_name('arguments')), ''))

    # Names come from py::arg; types/return from C++. Lambdas have no py::arg and
    # keep their own parameter names.
    params = []
    sig_index = 0
    for param in pyargs:
        if param.name == '*':
            params.append(param)
            continue
        param_type = sig.params[sig_index].type if sig_index < len(sig.params) else 'Any'
        params.append(param._replace(type=param_type))
        sig_index += 1
    params = params or sig.params
    return Sig(params, sig.ret, doc or sig.doc)


def index_functions(roots: list[Node]) -> dict[str, Sig]:
    """All C++ functions by name, with `constexpr auto` aliases flattened in."""
    funcs: dict[str, Sig] = {}
    aliases: dict[str, str] = {}
    for root in roots:
        for m in matches(Q_FUNC, root):
            defn = m['name'][0]
            while defn.type not in ('function_definition', 'declaration'):
                defn = defn.parent
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


def generate_pyi_file(name: str, root: str, output_dir: str = '.') -> None:
    roots = [PARSER.parse(p.read_bytes()).root_node
             for p in sorted(Path(root).rglob('*')) if p.is_file() and p.suffix.lower() in EXTS]
    funcs = index_functions(roots)

    # Each m.def becomes a stub; each `m.attr(alias) = m.attr(target)` clones one.
    bindings: dict[str, Sig] = {}
    attrs: dict[str, str] = {}
    for root_node in roots:
        for m in matches(Q_CALL, root_node):
            if m['method'][0].text == b'def' and is_module_object(m['object'][0]):
                bindings[first_string(m['args'][0])] = build_binding(m, funcs)
        for m in matches(Q_ATTR, root_node):
            if (m['lf'][0].text == m['rf'][0].text == b'attr'
                    and is_module_object(m['alias_object'][0]) and is_module_object(m['target_object'][0])):
                attrs[first_string(m['alias'][0])] = first_string(m['target'][0])
    for alias, target in attrs.items():
        while target in attrs and target not in bindings:
            target = attrs[target]
        if target in bindings and alias not in bindings:
            bindings[alias] = bindings[target]

    stubs = []
    for fn_name, f in bindings.items():
        lines = [f'    {p.name}: {p.type}' + (f' = {p.default}' if p.default is not None else '')
                 if p.name != '*' else '    *' for p in f.params]
        body = '\n' + ',\n'.join(lines) + '\n' if lines else ''
        head = f'def {fn_name}({body}) -> {f.ret}:'
        if not f.doc:
            stubs.append(f'{head} ...')
        elif '\n' in f.doc:
            doc = '\n'.join(f'    {ln}'.rstrip() for ln in f.doc.split('\n'))
            stubs.append(f'{head}\n    """\n{doc}\n    """\n    ...')
        else:
            stubs.append(f'{head}\n    """{f.doc}"""\n    ...')

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
    parser.add_argument('--root', default='./csrc/apis')
    parser.add_argument('--output-dir', default='./stubs')
    args = parser.parse_args()
    generate_pyi_file(name=args.name, root=args.root, output_dir=args.output_dir)
