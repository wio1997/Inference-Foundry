"""Entirely synthetic shared/routed branches; this is not a GLM layer."""

from .core import Cost, Node, Problem, Plan


def fixture():
    nodes = tuple(Node(name, name, {"toy_compute": 1},
                       Cost(cost, "synthetic fixture v1", "toy_compute capacity 2; fixed costs",
                            "synthetic"))
                  for name, cost in (("input", 1), ("shared", 4), ("routed", 3), ("join", 1)))
    problem = Problem("toy_shared_routed", nodes,
                      (("input", "shared"), ("input", "routed"),
                       ("shared", "join"), ("routed", "join")), {"toy_compute": 2})
    baseline = Plan.serial(problem)
    candidate = Plan("dependency_ready", problem.id, ("input", "routed", "shared", "join"),
                     problem.required_edges, ())
    return problem, baseline, candidate


def callbacks():
    return {
        "input": lambda deps, inputs: tuple(inputs),
        "shared": lambda deps, inputs: tuple(x * 2 for x in deps["input"]),
        "routed": lambda deps, inputs: tuple(x + 3 for x in deps["input"]),
        "join": lambda deps, inputs: tuple(a + b for a, b in zip(deps["shared"], deps["routed"])),
    }
