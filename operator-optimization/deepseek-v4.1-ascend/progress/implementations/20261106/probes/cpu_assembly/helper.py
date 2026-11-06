"""Exact-layout dataclass assembly; unknown layouts retain dataclasses.replace."""
from dataclasses import fields, replace


def make_replacer(cls):
    schema = fields(cls)
    assert not cls.__dataclass_params__.frozen
    assert len(schema) == len(cls.__dataclass_fields__) and all(f.init for f in schema)
    assert not hasattr(cls, '__post_init__')
    assert cls.__new__ is object.__new__ and cls.__setattr__ is object.__setattr__
    assert cls.__getattribute__ is object.__getattribute__
    assert '__slots__' not in cls.__dict__
    assert cls.__init__.__code__.co_filename == '<string>'
    names = tuple(f.name for f in schema)
    allowed = frozenset(names)

    def assemble(value, bound_replace=None, /, **changes):
        if type(value) is not cls:
            return bound_replace(**changes) if bound_replace is not None else replace(value, **changes)
        source = value.__dict__
        # Order, extra attributes and missing instance defaults also affect
        # replace's result. Keep its original behavior for all such cases.
        if tuple(source) != names or not changes.keys() <= allowed:
            return bound_replace(**changes) if bound_replace is not None else replace(value, **changes)
        result = object.__new__(cls)
        result.__dict__ = source.copy()
        result.__dict__.update(changes)
        return result

    return assemble


def publish_owned(value, cls, global_metadata, cp_token_range):
    """Only for the pinned base builder's fresh, unescaped return object."""
    if type(value) is not cls:
        return replace(value, global_metadata=global_metadata, cp_token_range=cp_token_range)
    value.global_metadata = global_metadata
    value.cp_token_range = cp_token_range
    return value
