"""Reduce host PD completion boundaries; never estimates removable device time.

Input manifest explicitly selects one D host/boot domain and required contributor
IDs. All requests remain in the result, including failed/incomplete requests.
Device readiness and a missed legal scheduling opportunity require separate raw
proof. CPU observations alone deliberately return INCONCLUSIVE.
"""
import argparse
import json
from pathlib import Path


def reduce_events(events, manifest):
    domain = (manifest["d_host"], manifest["d_boot_id"])
    required = set(manifest["required_contributors"])
    if not required or len(required) != len(manifest["required_contributors"]):
        raise ValueError("unique, nonempty contributor declaration required")
    requests = manifest["request_ids"]
    if not requests or len(set(requests)) != len(requests):
        raise ValueError("unique, nonempty request declaration required")
    rows = sorted((e for e in events if (e["host"], e["boot_id"]) == domain),
                  key=lambda e: e["monotonic_ns"])
    bad_trace = any(e["event"] in {"trace.truncated", "trace.extractor_error"}
                    for e in events)
    foreign_domains = sorted({(e["host"], e["boot_id"]) for e in events} - {domain})
    result = []
    for request_id in requests:
        captures = {}
        duplicate = False
        for event in rows:
            if event["event"] == "get_finished.end" and request_id in event["recv_ids"]:
                contributor = event["contributor"]
                if contributor in captures:
                    duplicate = True
                else:
                    captures[contributor] = event["monotonic_ns"]
        failures = [e["event"] for e in rows
                    if (e.get("request_id") == request_id
                        and (e["event"] == "_mark_failed_recv_request.begin"
                             or e["event"].endswith(".error")))
                    or (e["event"].endswith(".error")
                        and "request_id" not in e)]
        missing = sorted(required - set(captures))
        unexpected = sorted(set(captures) - required)
        all_captured = max(captures.values()) if captures and not missing else None

        def first(name, predicate):
            return next((e["monotonic_ns"] for e in rows
                         if e["event"] == name and predicate(e)), None)

        aggregated = first("aggregate.end", lambda e: request_id in e["recv_ids"])
        observed = first("_update_from_kv_xfer_finished.end",
                         lambda e: request_id in e["recv_ids"])
        promoted = first("_try_promote_blocked_waiting_request.end",
                         lambda e: e.get("request_id") == request_id and e["promoted"])
        scheduled = next((e["monotonic_ns"] for e in rows
                          if e["event"] == "schedule.end"
                          and e["tokens"].get(request_id, 0) > 0
                          and promoted is not None and e["monotonic_ns"] >= promoted), None)
        expected_counts = {e["expected"] for e in rows
                           if e["event"] == "aggregate.end" and request_id in e["recv_ids"]}
        chain = [all_captured, aggregated, observed, promoted, scheduled]
        monotonic = all(x is not None for x in chain) and chain == sorted(chain)
        complete = (not bad_trace and not failures and not duplicate
                    and not missing and not unexpected and monotonic
                    and expected_counts == {len(required)})

        def elapsed(a, b):
            return (b - a) / 1e6 if complete else None

        result.append(dict(
            request_id=request_id,
            host_chain_complete=complete,
            missing_contributors=missing, unexpected_contributors=unexpected,
            duplicate_completion=duplicate, failures=failures,
            all_worker_capture_ns=all_captured,
            aggregate_ns=aggregated, scheduler_observed_ns=observed,
            promoted_ns=promoted, scheduled_tokens_ns=scheduled,
            capture_to_aggregate_ms=elapsed(all_captured, aggregated),
            aggregate_to_scheduler_ms=elapsed(aggregated, observed),
            scheduler_to_promotion_ms=elapsed(observed, promoted),
            promotion_to_scheduled_tokens_ms=elapsed(promoted, scheduled),
            # No implicit device-ready, legal-opportunity, execution or public
            # output claim. Scheduled tokens may still await committed work.
            device_ready=None, missed_legal_opportunity=None,
            removable_ms=None, e2e_gain=None,
            verdict="INCONCLUSIVE"))
    return dict(simulation=bool(manifest.get("simulation", False)),
                domain=dict(host=domain[0], boot_id=domain[1]),
                ignored_clock_domains=foreign_domains, trace_loss=bad_trace,
                requests=result, effective_public_output_tokens=None,
                verdict="INCONCLUSIVE",
                missing_proof=["successful device-visible KV on every required contributor",
                               "otherwise legal uncommitted D admission opportunity",
                               "effective execution/commit and valid public output"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("traces", nargs="+")
    args = parser.parse_args()
    events = []
    for path in args.traces:
        # Malformed/truncated lines are an error; never silently drop them.
        events.extend(json.loads(line) for line in Path(path).read_text().splitlines())
    print(json.dumps(reduce_events(events, json.loads(Path(args.manifest).read_text())), indent=2))


if __name__ == "__main__":
    main()
