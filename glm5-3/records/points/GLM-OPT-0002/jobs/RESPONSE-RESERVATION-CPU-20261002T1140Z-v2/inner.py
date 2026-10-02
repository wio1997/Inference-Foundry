from pathlib import Path
import unittest,sys,json
root=Path("/data/tiankuan/wio/Inference-Foundry/glm5-3")
sys.path.insert(0,str(root/"runtime"));sys.path.insert(0,str(root/"tests"))
import test_response_affinity_v3 as suite
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(suite.V2Tests))
out=dict(tests_run=result.testsRun,errors=len(result.errors),failures=len(result.failures),passed=result.wasSuccessful(),actual_native_requests=0,scope="ActualgatewayV2ASGI/native39typedfixture/mocktransport; 6 inherited contracts and 3 ownership regressions")
(Path(__file__).parent/"race_reduction.json").write_text(json.dumps(out,indent=2)+"\n")
assert result.wasSuccessful();print(json.dumps(out))
