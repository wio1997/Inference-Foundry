from typing import Any, Optional


def value_or(value: Optional[Any], default: Any) -> Any:
    return default if value is None else value
