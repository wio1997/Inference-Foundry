"""Exercise actual launch AST with a subprocess double; no Docker/NPU call."""
import ast,hashlib,json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
HERE=Path(__file__).resolve().parent;source=HERE/'pd_rope_correctness.py'
nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('launch','epoch_root','runtime_witness_root')]
with TemporaryDirectory() as temporary:
 root=Path(temporary);(root/'witnesses').mkdir()
 for n in ('candidate','baseline_recovery'):(root/'epochs'/n).mkdir(parents=True)
 for n in ('mc2_mode.bin','stream_order_mode.bin','mtp_graph_mode.bin'):(root/n).write_bytes(b'\x01' if n=='mc2_mode.bin' else b'\x00')
 envfile=root/'environment.json';envfile.write_text(json.dumps(dict(environment={})))
 calls=[]
 plan=dict(baseline_env={'167':str(envfile)},environment={'167':{}},native_candidate=dict(library='/fixture/lib/libtorch_npu.so',Python_backend='/fixture/python'))
 ns=dict(ROOT=root,PLAN=plan,Path=Path,__file__=str(source),json=json,verify_artifact=lambda:True,m=NS(device_owners=lambda:(set(),None),PYTHON='/fixture/python3'),subprocess=NS(run=lambda argv,**kw:calls.append(argv)))
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
 ns['launch']('candidate');candidate_root=ns['runtime_witness_root']()
 (candidate_root/'witnesses/h11_capability_rank0.json').write_text('immutable candidate record')
 before=(candidate_root/'witnesses/h11_capability_rank0.json').read_bytes()
 ns['launch']('baseline_recovery');recovery_root=ns['runtime_witness_root']()
 assert recovery_root!=candidate_root and recovery_root==root/'epochs/baseline_recovery'
 assert (recovery_root/'witnesses').is_dir() and not any((recovery_root/'witnesses').iterdir())
 assert all((recovery_root/n).read_bytes()==b'\x00' for n in ('stream_order_mode.bin','mtp_graph_mode.bin'))
 assert (candidate_root/'witnesses/h11_capability_rank0.json').read_bytes()==before
 for argv,wroot in zip(calls,(candidate_root,recovery_root)):
  assert 'GLM_STREAM_ORDER_DIAGNOSTIC_ROOT='+str(wroot) in argv
  assert 'GLM_MTP_GRAPH_DIAGNOSTIC_ROOT='+str(wroot) in argv
 assert 'GLM_ROPE_LAYOUT_DIAGNOSTIC_ROOT='+str(root) in calls[0]
 assert not any(x.startswith('GLM_ROPE_LAYOUT_DIAGNOSTIC_ROOT=') for x in calls[1])
row=dict(passed=True,actual_launch_AST=True,epoch_witness_paths_disjoint=True,candidate_records_preserved=True,recovery_H11_H9_off=True,recovery_H12_observer_absent=True,subprocess_calls_intercepted=len(calls),NPU_requests=0,driver_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),limitations='CPU lifecycle double, not device proof')
p=HERE/'epoch_namespace_CPU_result.json';assert not p.exists();p.write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row))
