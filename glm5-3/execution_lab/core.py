"""Small dependency/resource model and explicitly registered CPU reference replay.

Times describe the supplied conditional-cost model only. The resource relaxation
is a reference within that model, never a hardware bound or measured NPU latency.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType


def _name(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _positive(value, label, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a positive {'integer' if integer else 'number'}")
    try:
        valid = isfinite(value) and value > 0
    except OverflowError:
        valid = False
    if not valid or (integer and not isinstance(value, int)):
        raise ValueError(f"{label} must be finite and positive")
    return value


def _keys(data, required, optional=()):
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    missing = set(required) - set(data)
    unknown = set(data) - set(required) - set(optional)
    if missing or unknown:
        raise ValueError(f"schema keys: missing={sorted(missing)}, unknown={sorted(unknown)}")


def _advance(start, duration):
    end = start + duration
    if not isfinite(end) or end <= start:
        raise ValueError("cost arithmetic overflow or insufficient time precision")
    return end


def _edges(raw, names):
    if not isinstance(raw, (list, tuple)):
        raise ValueError("edges must be a list")
    result = []
    for pair in raw:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("each edge must be [parent, child]")
        parent, child = pair
        _name(parent, "edge parent")
        _name(child, "edge child")
        if parent not in names or child not in names:
            raise ValueError(f"unknown dependency: {pair}")
        if parent == child or (parent, child) in result:
            raise ValueError(f"self or duplicate dependency: {pair}")
        result.append((parent, child))
    return tuple(result)


def _topology(names, edges):
    parents = {name: set() for name in names}
    for parent, child in edges:
        parents[child].add(parent)
    done, order = set(), []
    while len(order) < len(names):
        ready = [name for name in names if name not in done and parents[name] <= done]
        if not ready:
            raise ValueError("dependency cycle")
        # One at a time preserves the declared node order as a deterministic policy.
        done.add(ready[0])
        order.append(ready[0])
    return tuple(order), parents


@dataclass(frozen=True)
class Cost:
    duration_ms: float | None
    source: str
    condition: str
    kind: str

    def __post_init__(self):
        if self.duration_ms is not None:
            _positive(self.duration_ms, "duration_ms")
        _name(self.source, "cost source")
        _name(self.condition, "cost condition")
        if self.kind not in ("synthetic", "proxy", "measurement"):
            raise ValueError("cost kind must be synthetic, proxy, or measurement")

    @classmethod
    def from_dict(cls, data):
        _keys(data, ("duration_ms", "source", "condition", "kind"))
        return cls(**data)

    def to_dict(self):
        return dict(duration_ms=self.duration_ms, source=self.source,
                    condition=self.condition, kind=self.kind)


@dataclass(frozen=True)
class Node:
    id: str
    callback: str
    resources: dict
    cost: Cost

    def __post_init__(self):
        _name(self.id, "node id")
        _name(self.callback, "callback")
        if not isinstance(self.resources, dict) or not self.resources:
            raise ValueError("node resources must be a nonempty mapping")
        for name, amount in self.resources.items():
            _name(name, "resource")
            _positive(amount, "resource demand", integer=True)
        if not isinstance(self.cost, Cost):
            raise ValueError("node cost must be Cost")
        object.__setattr__(self, "resources", MappingProxyType(dict(self.resources)))


@dataclass(frozen=True)
class Problem:
    id: str
    nodes: tuple
    required_edges: tuple
    capacities: dict

    def __post_init__(self):
        _name(self.id, "problem id")
        if not isinstance(self.nodes, (tuple, list)) or not self.nodes:
            raise ValueError("nodes must be a nonempty list")
        if any(not isinstance(node, Node) for node in self.nodes):
            raise ValueError("nodes must contain Node objects")
        names = tuple(node.id for node in self.nodes)
        if len(set(names)) != len(names):
            raise ValueError("duplicate node id")
        if not isinstance(self.capacities, dict) or not self.capacities:
            raise ValueError("capacities must be a nonempty mapping")
        for name, capacity in self.capacities.items():
            _name(name, "resource")
            _positive(capacity, "resource capacity", integer=True)
        for node in self.nodes:
            for resource, demand in node.resources.items():
                if resource not in self.capacities or demand > self.capacities[resource]:
                    raise ValueError(f"unavailable capacity for {node.id}: {resource}")
        edges = _edges(self.required_edges, names)
        _topology(names, edges)
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "required_edges", edges)
        object.__setattr__(self, "capacities", MappingProxyType(dict(self.capacities)))

    @classmethod
    def from_dict(cls, data):
        _keys(data, ("id", "nodes", "required_edges", "capacities"))
        if not isinstance(data["nodes"], list):
            raise ValueError("nodes must be a list")
        nodes = []
        for raw in data["nodes"]:
            _keys(raw, ("id", "callback", "resources", "cost"))
            nodes.append(Node(raw["id"], raw["callback"], raw["resources"],
                              Cost.from_dict(raw["cost"])))
        return cls(data["id"], tuple(nodes), data["required_edges"], data["capacities"])

    def to_dict(self):
        return dict(id=self.id, nodes=[dict(id=n.id, callback=n.callback,
                    resources=dict(n.resources), cost=n.cost.to_dict()) for n in self.nodes],
                    required_edges=[list(e) for e in self.required_edges],
                    capacities=dict(self.capacities))


@dataclass(frozen=True)
class Plan:
    id: str
    problem_id: str
    node_ids: tuple
    required_edges: tuple
    serial_edges: tuple

    def validate(self, problem):
        _name(self.id, "plan id")
        if self.problem_id != problem.id:
            raise ValueError("plan problem_id mismatch")
        if not isinstance(self.node_ids, (tuple, list)):
            raise ValueError("node_ids must be a list")
        for name in self.node_ids:
            _name(name, "plan node id")
        names = tuple(n.id for n in problem.nodes)
        if len(self.node_ids) != len(names) or set(self.node_ids) != set(names):
            raise ValueError("plan must contain every problem node exactly once")
        required = _edges(self.required_edges, names)
        if set(required) != set(problem.required_edges):
            raise ValueError("plan cannot delete or change necessary dependencies")
        serial = _edges(self.serial_edges, names)
        if set(required) & set(serial):
            raise ValueError("serial_edges must describe only additional order")
        return _topology(tuple(self.node_ids), required + serial)

    @classmethod
    def from_dict(cls, data):
        _keys(data, ("id", "problem_id", "node_ids", "required_edges", "serial_edges"))
        return cls(**data)

    @classmethod
    def concurrent(cls, problem):
        return cls("dependency_ready", problem.id, tuple(n.id for n in problem.nodes),
                   problem.required_edges, ())

    @classmethod
    def serial(cls, problem):
        names = tuple(n.id for n in problem.nodes)
        order, _ = _topology(names, problem.required_edges)
        extra = tuple((a, b) for a, b in zip(order, order[1:])
                      if (a, b) not in problem.required_edges)
        return cls("serial_baseline", problem.id, order, problem.required_edges, extra)

    def to_dict(self):
        return dict(id=self.id, problem_id=self.problem_id, node_ids=list(self.node_ids),
                    required_edges=[list(e) for e in self.required_edges],
                    serial_edges=[list(e) for e in self.serial_edges])


def estimate(problem, plan):
    """Nonpreemptive greedy list schedule; supplied node order breaks ready ties.

    Fixed conditional costs do not capture changing overlap/traffic costs. A
    calibration or adapter must supply those costs; this is candidate screening.
    """
    order, parents = plan.validate(problem)
    nodes = {n.id: n for n in problem.nodes}
    duration = {}
    for node in problem.nodes:
        if node.cost.duration_ms is None:
            raise ValueError(f"unknown cost for {node.id}; cannot estimate")
        duration[node.id] = node.cost.duration_ms
    required_order, required_parents = _topology(tuple(nodes), problem.required_edges)
    finish = {}
    for name in required_order:
        finish[name] = _advance(max((finish[p] for p in required_parents[name]), default=0),
                                duration[name])
    critical_path = max(finish.values())
    work = {r: sum(duration[n.id] * n.resources.get(r, 0) for n in problem.nodes)
            / capacity for r, capacity in problem.capacities.items()}
    if not all(isfinite(v) for v in [*finish.values(), *work.values()]):
        raise ValueError("cost arithmetic overflow")
    active, done, started, timeline = {}, set(), set(), []
    now = 0
    while len(done) < len(nodes):
        for name in order:
            if name in started or not parents[name] <= done:
                continue
            node = nodes[name]
            fits = all(sum(nodes[a].resources.get(r, 0) for a in active) + demand
                       <= problem.capacities[r] for r, demand in node.resources.items())
            if fits:
                end = _advance(now, duration[name])
                active[name] = end
                started.add(name)
                timeline.append(dict(node=name, start_ms=now, end_ms=end,
                                     resources=dict(node.resources), cost=node.cost.to_dict()))
        if not active:
            raise ValueError("schedule cannot progress")
        now = min(active.values())
        completed = {n for n, end in active.items() if end == now}
        done.update(completed)
        active = {n: end for n, end in active.items() if n not in completed}
    return dict(problem=problem.id, plan=plan.id, status="model_estimate",
                makespan_ms=now, critical_path_ms=critical_path,
                resource_work_ms=work,
                resource_relaxation_reference_ms=max(critical_path, *work.values()),
                hardware_lower_bound="unknown", npu_measurement=False,
                verdict="screening_only", timeline=timeline)


def replay(problem, plan, callbacks, inputs):
    """Execute trusted Python callbacks sequentially in a legal reference order.

    A callback gets (required-parent outputs, inputs). Extra serial predecessors
    affect order only. Inputs and required-parent values must support deepcopy
    and equality. Their before/after snapshots reject persistent mutation, and
    returned values are independently snapshotted. Hidden globals are unchecked;
    registry contents are trusted caller code, never loaded from JSON/eval/shell.
    """
    order, _ = plan.validate(problem)
    nodes = {n.id: n for n in problem.nodes}
    missing = [n.callback for n in problem.nodes if not callable(callbacks.get(n.callback))]
    if missing:
        raise ValueError(f"missing registered callback: {missing}")
    _, parents = _topology(tuple(nodes), problem.required_edges)
    outputs = {}
    for name in order:
        parent_values = {p: outputs[p] for p in sorted(parents[name])}
        prior_inputs, prior_parents = deepcopy(inputs), deepcopy(parent_values)
        result = callbacks[nodes[name].callback](MappingProxyType(parent_values), inputs)
        if inputs != prior_inputs:
            raise ValueError(f"callback {name} mutated inputs")
        if parent_values != prior_parents:
            raise ValueError(f"callback {name} mutated required parent outputs")
        outputs[name] = deepcopy(result)
    return dict(status="CPU correctness_only", npu_measurement=False,
                execution_order=list(order), outputs=outputs)


def compare_replays(problem, first, second, callbacks, inputs):
    # Independent input copies prevent one run from contaminating the next.
    a = replay(problem, first, callbacks, deepcopy(inputs))
    b = replay(problem, second, callbacks, deepcopy(inputs))
    if a["outputs"] != b["outputs"]:
        raise ValueError("reference replay outputs differ across legal plans")
    return dict(status="CPU correctness_only", equal=True, npu_measurement=False,
                orders=[a["execution_order"], b["execution_order"]], outputs=a["outputs"])
