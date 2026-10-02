from pathlib import Path
import sys,unittest,json
root=Path("/data/tiankuan/wio/Inference-Foundry/glm5-3");sys.path.insert(0,str(root/"runtime"));sys.path.insert(0,str(root/"tests"))
import test_response_budget_v9 as suite
tests=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(c)for c in[suite.EstimateContracts,suite.GatewayBudgetContracts,suite.CLIBudgetContract]])
from response_affinity_gateway_v8 import create_app
import test_prefill_workload_v4 as weighted
weighted.create_app=create_app
tests.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(weighted.BytePlacementContracts))
tests.addTest(weighted.ByteGatewayContract('test_actual_ASGI_three_held_requests_reserve_lighter_owner_without_body_changes'))
import test_work_seconds as work
work.create_app=create_app
tests.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(work.WorkContracts))
tests.addTest(work.WorkGatewayContract("test_actual_ASGI_three_held_requests_balance_prefill_and_decode_without_body_changes"))
import test_native_pd as pd
tests.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(pd.PDTransportTests))
r=unittest.TextTestRunner(verbosity=2).run(tests)
out=dict(tests_run=r.testsRun,failures=len(r.failures),errors=len(r.errors),passed=r.wasSuccessful(),actual_native_requests=0,scope="ActualgatewayV8/nativePDtransport CLI+ASGI/prefill-bytes/nativeResponsesRequest/native39 typedstorefixture/mocktransport; retained9ownercontracts + nativebudget hints/bodywire/leasebackground boundary")
(Path(__file__).parent/"race_reduction.json").write_text(json.dumps(out,indent=2)+"\n");assert r.wasSuccessful();print(json.dumps(out))
