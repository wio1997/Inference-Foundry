"""Local JSON model estimation and synthetic CPU replay. No device calls."""

import argparse
import json
from pathlib import Path
import sys

# Support direct invocation without installing a package or model framework.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from execution_lab.core import Problem, Plan, estimate, compare_replays
from execution_lab.toy import fixture, callbacks


def _read(path):
    # Python's JSON parser accepts NaN/Infinity by default; reject them explicitly.
    def invalid(value):
        raise ValueError(f"invalid JSON number: {value}")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle, parse_constant=invalid)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo")
    command = commands.add_parser("estimate")
    command.add_argument("problem")
    command.add_argument("plan")
    args = parser.parse_args(argv)
    try:
        if args.command == "estimate":
            result = estimate(Problem.from_dict(_read(args.problem)), Plan.from_dict(_read(args.plan)))
        else:
            problem, baseline, candidate = fixture()
            result = dict(fixture="toy_shared_routed", all_costs="synthetic", device_calls=False,
                          verdict="screening_only", estimates=[estimate(problem, p)
                          for p in (baseline, candidate)],
                          replay=compare_replays(problem, baseline, candidate, callbacks(), [1, 2, 3]))
        print(json.dumps(result, separators=(",", ":"), allow_nan=False))
        return 0
    except (ValueError, TypeError, OSError, KeyError, OverflowError) as error:
        print(json.dumps(dict(error=str(error), status="invalid"), separators=(",", ":")))
        return 2


if __name__ == "__main__":
    sys.exit(main())
