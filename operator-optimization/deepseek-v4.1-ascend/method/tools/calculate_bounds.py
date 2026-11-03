"""Exact arithmetic for declared lower-bound terms; NOT a proof checker."""
import argparse
from decimal import Decimal, localcontext
from fractions import Fraction
import json
from pathlib import Path


def number(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("Use an integer or exact rational/decimal string, or null")
    result = Fraction(value)
    if result < 0:
        raise ValueError("Negative values cannot describe this nonnegative cost model")
    return result


def display(value):
    if value is None:
        return None
    with localcontext() as ctx:
        ctx.prec = 24
        approximate = str(Decimal(value.numerator) / Decimal(value.denominator))
    return {"exact_fraction": str(value), "decimal_approx_not_certificate": approximate}


def calculate(data):
    if not data.get("scope") or not data.get("time_unit") or not data.get("work_unit"):
        raise ValueError("scope, time_unit and work_unit are required")
    q = number(data.get("effective_work"))
    if q is not None and q <= 0:
        raise ValueError("effective_work must be positive or null")
    terms = {}
    for term in data.get("terms", []):
        key = term["id"]
        if key in terms:
            raise ValueError("Duplicate term id: " + key)
        if term["kind"] == "resource":
            work, rate = number(term.get("work_lower")), number(term.get("rate_upper"))
            if rate == 0:
                raise ValueError("Zero capacity is not supported; retain unknown as null")
            value = work / rate if work is not None and rate is not None else None
        elif term["kind"] == "delay":
            value = number(term.get("time_lower"))
        else:
            raise ValueError("Unknown term kind")
        terms[key] = (value, term)
    result, prior, chosen = [], Fraction(0), set()
    for layer in data.get("layers", []):
        selected = set(layer["term_ids"])
        if len(selected) != len(layer["term_ids"]):
            raise ValueError("Duplicate term in layer")
        if not chosen.issubset(selected) or not selected.issubset(terms):
            raise ValueError("Layers must retain earlier terms and use known term ids")
        known = [terms[key][0] for key in selected if terms[key][0] is not None]
        lower = max([prior] + known)
        upper = q / lower if q is not None and lower > 0 else None
        result.append({
            "id": layer["id"],
            "time_lower_from_known_terms": display(lower),
            "throughput_upper_from_known_terms": display(upper),
            "unresolved_term_ids": sorted(key for key in selected if terms[key][0] is None),
            "selected_terms": [terms[key][1] for key in sorted(selected)],
            "proof_status": "NOT_CHECKED_BY_CALCULATOR",
            "no_finite_numeric_ceiling": upper is None,
        })
        chosen, prior = selected, lower
    return {
        "scope": data["scope"], "time_unit": data["time_unit"],
        "work_unit": data["work_unit"], "effective_work": display(q),
        "status": "ARITHMETIC_ONLY",
        "notice": "Exact fractions certify arithmetic only. Units, source facts, scope coverage, necessity, and proof status require separate review. Unknown terms are omitted only from the partial numeric bound, never treated as established zero cost.",
        "layers": result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    print(json.dumps(calculate(json.loads(args.input.read_text())), ensure_ascii=False, indent=2))
