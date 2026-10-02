from pathlib import Path
import sys,unittest,json
root=Path("/data/tiankuan/wio/Inference-Foundry/glm5-3");sys.path.insert(0,str(root/"runtime"));sys.path.insert(0,str(root/"tests"))
import test_response_budget_v5 as suite
tests=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(c)for c in[suite.EstimateContracts,suite.GatewayBudgetContracts,suite.CLIBudgetContract]])
r=unittest.TextTestRunner(verbosity=2).run(tests)
out=dict(tests_run=r.testsRun,failures=len(r.failures),errors=len(r.errors),passed=r.wasSuccessful(),actual_native_requests=0,scope="ActualgatewayV4 CLI+ASGI/nativeResponsesRequest/native39 typedstorefixture/mocktransport; retained9ownercontracts + nativebudget hints/bodywire/leasebackground boundary")
(Path(__file__).parent/"race_reduction.json").write_text(json.dumps(out,indent=2)+"\n");assert r.wasSuccessful();print(json.dumps(out))
