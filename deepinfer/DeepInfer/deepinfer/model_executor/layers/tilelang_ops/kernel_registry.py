"""Index backend factories without importing their compiler or device dependencies."""
import ast
from pathlib import Path


def factory_modules(package_paths, package_name):
    result = {}
    for directory in package_paths:
        for path in sorted(Path(directory).glob('*.py')):
            if path.stem == '__init__':
                continue
            names = set()
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    names.add(node.name)
                elif isinstance(node, ast.ImportFrom):
                    names.update(alias.asname or alias.name for alias in node.names)
            for name in sorted(names):
                if not name.startswith('_') and name.endswith('_tl'):
                    result.setdefault(name, []).append(f'{package_name}.{path.stem}')
    return {name: tuple(modules) for name, modules in result.items()}
