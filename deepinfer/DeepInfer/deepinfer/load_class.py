import importlib.util


def load_class_from_file(file_path: str, class_name: str) -> type:
    spec = importlib.util.spec_from_file_location("module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)
