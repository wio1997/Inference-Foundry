"""Semantic validation, resource contention, and actual CPU reference execution."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from execution_lab.core import Cost, Node, Problem, Plan, estimate, replay, compare_replays
from execution_lab.toy import fixture, callbacks
from execution_lab.cli import main


class ExecutionLabTests(unittest.TestCase):
    def setUp(self):
        self.problem, self.serial, self.concurrent = fixture()

    def problem_changed(self, change):
        raw = self.problem.to_dict()
        change(raw)
        return Problem.from_dict(raw)

    def plan_changed(self, change):
        raw = self.concurrent.to_dict()
        change(raw)
        plan = Plan.from_dict(raw)
        plan.validate(self.problem)
        return plan

    def test_required_cycle_and_unknown_edges(self):
        for edge in (("join", "input"), ("absent", "join"), ("input", "input")):
            with self.subTest(edge=edge), self.assertRaises(ValueError):
                self.problem_changed(lambda d: d["required_edges"].append(edge))

    def test_plan_cannot_delete_or_change_dependencies(self):
        for change in (
            lambda d: d["required_edges"].pop(),
            lambda d: d["required_edges"].append(["shared", "routed"]),
            lambda d: d["required_edges"].append(["unknown", "routed"]),
            lambda d: d["serial_edges"].append(["join", "input"]),
            lambda d: d["serial_edges"].append(["unknown", "join"]),
            lambda d: d["serial_edges"].append(["input", "shared"]),
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.plan_changed(change)

    def test_node_and_schema_validation(self):
        for change in (
            lambda d: d["node_ids"].pop(),
            lambda d: d["node_ids"].append("unknown"),
            lambda d: d["node_ids"].__setitem__(0, "shared"),
            lambda d: d.__setitem__("problem_id", "other"),
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.plan_changed(change)
        with self.assertRaises(ValueError):
            self.problem_changed(lambda d: d["nodes"].append(d["nodes"][0]))
        with self.assertRaises(ValueError):
            self.problem_changed(lambda d: d.__setitem__("unrecognized", 1))

    def test_capacity_and_demand_validation(self):
        for value in (0, -1, True, 1.5, float("inf"), float("nan"), "2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.problem_changed(lambda d: d["capacities"].__setitem__("toy_compute", value))
            with self.subTest(demand=value), self.assertRaises(ValueError):
                self.problem_changed(lambda d: d["nodes"][0]["resources"].__setitem__("toy_compute", value))
        with self.assertRaises(ValueError):
            self.problem_changed(lambda d: d["nodes"][0]["resources"].__setitem__("toy_compute", 3))
        with self.assertRaises(ValueError):
            self.problem_changed(lambda d: d["nodes"][0].__setitem__("resources", {"missing": 1}))

    def test_cost_validation_and_unknown_is_not_zero(self):
        for value in (0, -1, True, float("nan"), float("inf"), -float("inf"), "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Cost(value, "fixture", "fixed synthetic condition", "synthetic")
        for field in ("source", "condition", "kind"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.problem_changed(lambda d: d["nodes"][0]["cost"].__setitem__(field, ""))
        unknown = self.problem_changed(lambda d: d["nodes"][1]["cost"].__setitem__("duration_ms", None))
        with self.assertRaisesRegex(ValueError, "unknown cost"):
            estimate(unknown, self.concurrent)
        # Correctness replay does not require a made-up timing cost.
        self.assertEqual(replay(unknown, self.serial, callbacks(), [1])["outputs"]["join"], (6,))

    def test_parallel_timeline_and_references(self):
        a = estimate(self.problem, self.serial)
        b = estimate(self.problem, self.concurrent)
        self.assertEqual(a["makespan_ms"], 9)
        self.assertEqual(b["makespan_ms"], 6)
        self.assertEqual(b["critical_path_ms"], 6)
        self.assertEqual(b["resource_work_ms"], {"toy_compute": 4.5})
        self.assertEqual(b["resource_relaxation_reference_ms"], 6)
        self.assertEqual(b["hardware_lower_bound"], "unknown")
        times = {v["node"]: (v["start_ms"], v["end_ms"]) for v in b["timeline"]}
        self.assertEqual(times, {"input": (0, 1), "routed": (1, 4), "shared": (1, 5), "join": (5, 6)})

    def test_competing_branches_serialize_at_one_slot(self):
        def change(d):
            d["capacities"]["toy_compute"] = 1
            for node in d["nodes"]:
                node["cost"]["condition"] = "synthetic one-slot fixture; fixed costs"
        problem = self.problem_changed(change)
        result = estimate(problem, self.concurrent)
        self.assertEqual(result["makespan_ms"], 9)
        self.assertEqual(result["resource_relaxation_reference_ms"], 9)
        branches = {v["node"]: v for v in result["timeline"]}
        self.assertGreaterEqual(branches["shared"]["start_ms"], branches["routed"]["end_ms"])

    def test_multiresource_capacity_constraints(self):
        def change(d):
            d["capacities"]["toy_memory"] = 1
            for node in d["nodes"]:
                node["resources"]["toy_memory"] = 1
                node["cost"]["condition"] = "synthetic two-resource fixture; fixed costs"
        problem = self.problem_changed(change)
        self.assertEqual(estimate(problem, self.concurrent)["makespan_ms"], 9)

    def test_cost_arithmetic_cannot_silently_drop_time(self):
        problem = self.problem_changed(lambda d: d["nodes"][0]["cost"].__setitem__("duration_ms", 1e300))
        with self.assertRaisesRegex(ValueError, "precision"):
            estimate(problem, self.concurrent)

    def test_waiter_starts_on_capacity_release_while_other_task_runs(self):
        nodes = tuple(Node(name, name, resources, Cost(duration, "synthetic event fixture",
                      "compute=2; memory=1; fixed costs", "synthetic"))
                      for name, duration, resources in (
                          ("long", 10, {"compute": 1}),
                          ("short", 2, {"compute": 1, "memory": 1}),
                          ("waiting", 3, {"compute": 1, "memory": 1})))
        problem = Problem("toy_release", nodes, (), {"compute": 2, "memory": 1})
        result = estimate(problem, Plan.concurrent(problem))
        times = {v["node"]: (v["start_ms"], v["end_ms"]) for v in result["timeline"]}
        self.assertEqual(times, {"long": (0, 10), "short": (0, 2), "waiting": (2, 5)})
        self.assertEqual(result["makespan_ms"], 10)

    def test_reference_callbacks_execute_different_legal_orders(self):
        calls = []
        registry = {}
        for name, callback in callbacks().items():
            def tracked(deps, inputs, name=name, callback=callback):
                calls.append(name)
                return callback(deps, inputs)
            registry[name] = tracked
        result = compare_replays(self.problem, self.serial, self.concurrent, registry, [1, 2, 3])
        self.assertTrue(result["equal"])
        self.assertEqual(result["outputs"]["join"], (6, 9, 12))
        self.assertNotEqual(*result["orders"])
        self.assertEqual(calls, result["orders"][0] + result["orders"][1])
        self.assertEqual(result["status"], "CPU correctness_only")
        with self.assertRaisesRegex(ValueError, "missing registered callback"):
            replay(self.problem, self.serial, {}, [])

    def test_order_sensitive_callback_fails_equivalence(self):
        counter = iter(range(20))
        registry = callbacks()
        registry["shared"] = lambda deps, inputs: (next(counter),)
        with self.assertRaisesRegex(ValueError, "outputs differ"):
            compare_replays(self.problem, self.serial, self.concurrent, registry, [1])

    def test_mutated_parent_alias_cannot_fake_replay_equivalence(self):
        def shared(deps, inputs):
            deps["input"][0] = 1
            return deps["input"]
        def routed(deps, inputs):
            deps["input"][0] += 2
            return deps["input"]
        def join(deps, inputs):
            deps["shared"][0] = 0
            return deps["shared"]
        registry = dict(input=lambda deps, inputs: [0], shared=shared, routed=routed, join=join)
        with self.assertRaisesRegex(ValueError, "mutated required parent outputs"):
            compare_replays(self.problem, self.serial, self.concurrent, registry, [])

    def test_mutated_inputs_rejected(self):
        registry = callbacks()
        def mutate(deps, inputs):
            inputs[0] = 99
            return tuple(inputs)
        registry["input"] = mutate
        with self.assertRaisesRegex(ValueError, "mutated inputs"):
            compare_replays(self.problem, self.serial, self.concurrent, registry, [1])

    def test_return_snapshot_survives_later_hidden_alias_mutation(self):
        storage = [0]
        def shared(deps, inputs):
            storage[0] = 99  # Hidden side effect is unchecked, but cannot rewrite old outputs.
            return tuple(deps["input"])
        registry = callbacks()
        registry.update(input=lambda deps, inputs: storage, shared=shared)
        result = replay(self.problem, self.serial, registry, [1])
        self.assertEqual(storage, [99])
        self.assertEqual(result["outputs"]["input"], [0])
        self.assertEqual(result["outputs"]["shared"], (0,))

    def test_cli_demo_and_estimate_json(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["demo"]), 0)
        data = json.loads(output.getvalue())
        self.assertEqual(data["all_costs"], "synthetic")
        self.assertFalse(data["device_calls"])
        with tempfile.TemporaryDirectory() as tmp:
            p, q = Path(tmp) / "p.json", Path(tmp) / "q.json"
            p.write_text(json.dumps(self.problem.to_dict()))
            q.write_text(json.dumps(self.concurrent.to_dict()))
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(["estimate", str(p), str(q)]), 0)
            self.assertEqual(json.loads(output.getvalue())["makespan_ms"], 6)
            p.write_text(p.read_text().replace('"duration_ms": 1', '"duration_ms": NaN', 1))
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(["estimate", str(p), str(q)]), 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "invalid")


if __name__ == "__main__":
    unittest.main()
