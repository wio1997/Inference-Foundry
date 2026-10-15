# Independent execution-plan toy

This local standard-library experiment is a synthetic shared/routed branch DAG,
not a GLM layer or an Ascend runtime. It imports neither vLLM nor SGLang and makes
no device calls. Times are conditional model estimates for candidate screening;
they do not establish hardware bounds, NPU performance, or a performance KEEP.

```text
          shared
         /      \
input --          -- join
         \      /
          routed
```

The four arrows are necessary dependencies. The serial baseline adds
`shared -> routed` as an extra order, while the dependency-ready candidate leaves
the branches independent. `required_edges` must repeat the problem's necessary
edges exactly, `serial_edges` declares only additional order, and `node_ids`
sets deterministic ready-task priority. A plan cannot omit nodes or necessary
dependencies. Its total graph must remain acyclic.

```sh
python3 glm5-3/execution_lab/cli.py demo
python3 glm5-3/execution_lab/cli.py estimate \
  glm5-3/examples/execution_lab/toy_problem.json \
  glm5-3/examples/execution_lab/toy_concurrent.json
python3 -m unittest discover -s glm5-3/tests -p test_execution_lab.py -v
```

The synthetic fixture estimates 9 ms serial and 6 ms with two toy compute slots.
The scheduler admits ready nodes only when all resource demands fit; a one-slot
variant serializes competing branches. It reports the necessary-DAG critical
path and the maximum of critical path and aggregate resource work/capacity as a
relaxation reference within the supplied model. Extra plan order affects the
schedule, while those references concern the underlying problem. Scheduling is
greedy and nonpreemptive; its makespan is not an optimality certificate.

Each cost carries `duration_ms`, `source`, `condition`, and `kind` (`synthetic`,
`proxy`, or `measurement`). A null duration represents an unknown cost and blocks
estimation. Positive finite costs and positive integer demands/capacities are
required. The tool records supplied conditions; it cannot verify calibration,
source truth, or whether an overlap changes costs. Users must supply a matching
cost model before drawing conclusions. Changing shapes, traffic, or resources
can invalidate fixed costs. Real cost collection and an Ascend adapter remain
future work; this is not a complete inference framework.

For callable CPU reference replay, put `glm5-3` on Python's import path:

```python
from execution_lab.core import compare_replays, estimate
from execution_lab.toy import fixture, callbacks
problem, serial, concurrent = fixture()
estimate(problem, concurrent)
compare_replays(problem, serial, concurrent, callbacks(), [1, 2, 3])
```

Replay executes explicitly registered Python functions sequentially in a legal
order. Callbacks receive `(required_parent_outputs, inputs)`; extra serial edges
do not provide semantic inputs. The demo checks all intermediate outputs under
two different legal branch orders and returns `CPU correctness_only`. Registry
code is trusted caller code, never imported from JSON or evaluated as commands.
Callbacks must be pure, with inputs and outputs supporting deepcopy and equality.
Before/after snapshots reject persistent changes to callback inputs and necessary
parent values, including mutable values inside the read-only dependency mapping.
Each returned value is independently snapshotted so later alias mutations cannot
rewrite earlier outputs. Repeated runs receive independent input copies. These
checks cannot detect hidden global state or transient changes restored before a
callback returns; caller purity remains a requirement. Replay checks CPU semantics only and says
nothing about device numerics, synchronization, distributed state, or speed.
