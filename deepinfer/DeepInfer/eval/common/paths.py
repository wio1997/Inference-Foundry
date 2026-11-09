from __future__ import annotations

import os
import sys
from pathlib import Path

COMMON_DIR = Path(__file__).resolve().parent
EVAL_DIR = COMMON_DIR.parent
DATA_DIR = EVAL_DIR / "data"
DEFAULT_OUT_ROOT = EVAL_DIR / "outputs"


def _looks_like_repo(path: Path) -> bool:
    return (path / "examples" / "_config.py").is_file() and (path / "deepinfer").is_dir()


def resolve_repo() -> Path:
    """Locate the repository root.

    By default it is inferred from the location of eval/; set `DEEPINFER_REPO=<repo root>` when the
    directory layout is unusual.
    """
    candidates: list[Path] = []
    if os.environ.get("DEEPINFER_REPO"):
        candidates.append(Path(os.environ["DEEPINFER_REPO"]).expanduser())
    candidates += [EVAL_DIR.parent, Path.cwd()]
    for cand in candidates:
        try:
            if _looks_like_repo(cand):
                return cand.resolve()
        except OSError:
            continue
    print(
        f"[eval] warning: could not infer the repository root from {EVAL_DIR} (no examples/ "
        "and deepinfer/); set DEEPINFER_REPO=<absolute repo root>",
        file=sys.stderr,
    )
    return EVAL_DIR.parent


def ensure_sys_path() -> Path:
    """Put the repository root and the eval directory on sys.path; return the repository root."""
    repo = resolve_repo()
    for entry in (str(repo), str(EVAL_DIR)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    return repo


def resolve_out_dir(out_dir: str | None, *, bench: str, tag: str) -> str:
    return out_dir or str(DEFAULT_OUT_ROOT / bench / tag)
