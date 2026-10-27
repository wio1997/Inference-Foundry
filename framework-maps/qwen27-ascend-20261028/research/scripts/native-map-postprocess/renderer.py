#!/usr/bin/env python3
"""Render the interactive evidence map; CPU lifecycle shards load over HTTP from sibling assets."""
from __future__ import annotations
import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent


def _json_for_script(data):
    # Escape HTML-significant characters so attacker-controlled content cannot
    # close the script element or introduce HTML while parsed as JSON.
    return (json.dumps(data, ensure_ascii=False, separators=(",", ":"))
            .replace("&", "\\u0026").replace("<", "\\u003c")
            .replace(">", "\\u003e").replace("\u2028", "\\u2028")
            .replace("\u2029", "\\u2029"))


def render(data, template=None):
    if not isinstance(data, dict):
        raise ValueError("combined input must be a JSON object")
    required = ("metadata", "baseline", "mechanisms", "design", "new_measurements")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError("combined input is missing top-level keys: " + ", ".join(missing))
    template_path = pathlib.Path(template) if template else HERE / "template.html"
    source = template_path.read_text(encoding="utf-8")
    token = "__Q27_COMBINED_JSON__"
    if source.count(token) != 1:
        raise ValueError("template must contain exactly one JSON insertion point")
    return source.replace(token, _json_for_script(data))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Combined map JSON (does not modify it)")
    ap.add_argument("--output", required=True, help="Output standalone HTML path")
    ap.add_argument("--template", help="Optional HTML template path")
    args = ap.parse_args(argv)
    data = json.loads(pathlib.Path(args.input).read_text(encoding="utf-8"))
    html = render(data, args.template)
    pathlib.Path(args.output).write_text(html, encoding="utf-8")
    print(f"wrote {args.output} ({len(html.encode('utf-8'))} bytes)")


if __name__ == "__main__":
    main()
