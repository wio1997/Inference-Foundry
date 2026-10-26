import pathlib,hashlib,json,subprocess,sys,ast
root=pathlib.Path('/data/wio/Inference_Foundry');out=root/'evidence/20260927_loop080_bound/run552';out.mkdir(parents=True,exist_ok=True)
client=root/'scripts/loop080_frontier_client.py';validator=root/'scripts/loop080_frontier_client_validate.py';test=root/'evidence/20260927_loop080_bound/run551/frontier_client_cpu_test.py'
source=test.read_text();source=source.replace('import types','import types\nimport time')
source=source.replace('class Timeout:', '''semaphore_trace = {}
RealSemaphore = asyncio.Semaphore
class AuditedSemaphore(RealSemaphore):
    async def acquire(self):
        result = await super().acquire()
        semaphore_trace.setdefault(asyncio.current_task(), {})["actual_acquired_by"] = time.monotonic_ns()
        return result
    def release(self):
        t = semaphore_trace.setdefault(asyncio.current_task(), {})
        t["actual_release_not_before"] = time.monotonic_ns()
        result = super().release()
        t["actual_release_completed_by"] = time.monotonic_ns()
        return result
asyncio.Semaphore = AuditedSemaphore
class Timeout:''')
source=source.replace('assert url == "fake://no-network"','semaphore_trace[asyncio.current_task()]["i"] = int(json["messages"][0]["content"][1:])\n        assert url == "fake://no-network"')
source=source.replace('assert lineage["unique_parent_edges_certified"] == 0','''assert lineage["unique_parent_edges_certified"] == 0
    actual = {x["i"]:x for x in semaphore_trace.values()}
    assert len(actual) == 48
    for row in summary["requests"]:
        t = actual[row["i"]]
        assert row["c12_acquire_attempt_monotonic_ns"] <= t["actual_acquired_by"] <= row["c12_acquired_monotonic_ns"]
        assert row["end_monotonic_ns"] <= t["actual_release_not_before"] <= t["actual_release_completed_by"] <= row["c12_release_completed_by_monotonic_ns"]
    # Invalid generated markers must reject even with otherwise valid SSE rows.
    import copy
    for label, mutate in [
       ("acquired_after_POST",lambda r:r.update(c12_acquired_monotonic_ns=r["start_monotonic_ns"]+1)),
       ("DONE_after_end",lambda r:r.update(sse_done_monotonic_ns=r["end_monotonic_ns"]+1)),
       ("release_upper_before_end",lambda r:r.update(c12_release_completed_by_monotonic_ns=r["end_monotonic_ns"]-1)),
       ("claim_unique_parent",lambda r:r.update(c12_unique_predecessor_certified=True)),
       ("noninteger_clock",lambda r:r.update(c12_acquired_monotonic_ns=float(r["c12_acquired_monotonic_ns"]))),
    ]:
        changed=copy.deepcopy(summary["requests"]);mutate(changed[0])
        try:check.validate(changed,12)
        except (ValueError,TypeError,KeyError):pass
        else:raise AssertionError(label)
''')
source=source.replace('"status": "pass", "scope":','"actual_semaphore_acquire_release_brackets_verified_requests":48, "extra_marker_negatives":5, "status": "pass", "scope":')
probe=out/'actual_semaphore_cpu_test.py';probe.write_text(source)
commands=[([sys.executable,str(probe),str(client),str(validator),str(out/'actual_semaphore_cpu.json')],out/'actual_semaphore_cpu.log'),([sys.executable,str(root/'evidence/20260927_loop079_identity/run528/independent_cpu_review.py'),str(client),str(out/'transport_cpu.json')],out/'transport_cpu.log')]
for cmd,log in commands:
 p=subprocess.run(cmd,cwd=root,capture_output=True,text=True);log.write_text(p.stdout+p.stderr);assert p.returncode==0,(cmd,p.stderr)
old=ast.parse((root/'scripts/loop079_formal_ledger_client.py').read_text());new=ast.parse(client.read_text())
fn=lambda t,n:next(x for x in t.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name==n)
assert ast.dump(fn(old,'body'),include_attributes=False)==ast.dump(fn(new,'body'),include_attributes=False)
sys.path.insert(0,str(root/'scripts'));import loop080_frontier_client_validate as v
# Explicitly verify allowed ambiguity: release upper bound may be later than a successor acquisition.
rows=[]
for i in range(48):
 acq=1000000+(i//12)*10000+i%12;end=acq+1000
 rows.append({'i':i,'c12_acquire_attempt_monotonic_ns':acq-1,'c12_acquired_monotonic_ns':acq,'start_monotonic_ns':acq+1,'sse_done_monotonic_ns':end-1,'end_monotonic_ns':end,'c12_release_completed_by_monotonic_ns':end+100000,'c12_unique_predecessor_certified':False,'http_status':200,'output_tokens':1024,'error':None})
res=v.validate(rows,12);assert res['unique_parent_edges_certified']==0
files=[client,validator,test,root/'evidence/20260927_loop080_bound/run551/source_preflight.md',root/'scripts/loop079_formal_ledger_client.py',root/'scripts/bench.py',root/'runtime/fixed_serving.py',root/'runtime/extreme_decode.py',pathlib.Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/worker/model_runner_v1.py'),root/'evidence/20260927_loop079_identity/run528/independent_cpu_review.py']
checks={'status':'PASS_COMPONENT_ONLY','actual_semaphore_bracket_requests':48,'additional_marker_negatives':5,'existing_validator_synthetic':v.self_test(),'existing_transport_cases':len(json.loads((out/'transport_cpu.json').read_text())['cases']),'body_AST_unchanged':True,'wide_release_brackets_admitted_without_unique_parent':True,'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'service_or_NPU_execution':False,'production_source_mutation':False}
(out/'independent_checks.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))
