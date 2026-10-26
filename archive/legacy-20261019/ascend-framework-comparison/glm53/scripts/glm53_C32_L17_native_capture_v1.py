"""DRAFT_PENDING_PRIMARY_REVIEW: L17 mixed-root native capture factory.

Local rendering/import only.  ``plans()`` reads frozen local sources and the two
immutable capacity records, renders the four host programs and writes nothing.
``execute()`` is the future once-only remote entry point; it requires an
independently approved actual review whose host-code hashes match exactly.  This
draft performs zero SSH/network/Docker/NPU/SDK/model/Git actions.

The actual base capture is the accepted v3 source (8cad4c...).  Its log archive,
output codec, persistent SDMA audit, alignment audit, SDK snapshots, Graph
counters and profiler-metadata preservation are reused.  This factory changes
only the root/CID/identity wiring, adds D eager-input first-use evidence, and
keeps the P native optional-extension map check.
"""
import ast
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import pathlib
import shlex

import glm53_C32_L17_pipeline_contract_v1 as contract

ROOT = contract.ROOT
STATUS = contract.STATUS

MODES = ("startup", "postfunctional", "formal")
MODE_CAPTURE = {
    "startup": contract.STARTUP_CAPTURE,
    "postfunctional": contract.FUNCTIONAL_CAPTURE,
    "formal": contract.FORMAL_CAPTURE,
}
MODE_REVIEW = {
    "startup": contract.STARTUP_CAPTURE_REVIEW,
    "postfunctional": contract.FUNCTIONAL_CAPTURE_REVIEW,
    "formal": contract.FORMAL_CAPTURE_REVIEW,
}

D_EAGER_BLOCK = r'''
out['D_eager_input_audit']={'applicable':NATIVE_D,'status':'NOT_APPLICABLE' if not NATIVE_D else 'UNREAD'}
if NATIVE_D:
 import collections
 sdk_pids={event['pid'] for item in out['persistent_audit'].values() for event in item['rows'] if event['event']=='initialize_complete'}
 init={event['pid']:event for item in out['persistent_audit'].values() for event in item['rows'] if event['event']=='initialize_complete'}
 assert len(sdk_pids)==16
 owned={}
 for row in rows:
  assert row['namespace_pid'] not in owned
  owned[row['namespace_pid']]=row
 workers=[]
 for pid in sorted(sdk_pids):
  assert pid in owned
  event=init[pid]
  host_pid=owned[pid]['host_pid']
  proc=pathlib.Path('/proc')/str(host_pid)
  assert proc.exists()
  status=(proc/'status').read_text()
  nspid=None
  for line in status.splitlines():
   if line.startswith('NSpid:'):
    nspid=[int(x) for x in line.split()[1:]];break
  assert nspid and nspid[-1]==pid
  cgroup=(proc/'cgroup').read_text()
  assert CID in cgroup
  stat=(proc/'stat').read_text();rest=stat.rsplit(')',1)[1].split()
  argv=[x for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
  alive=bool(argv and rest[0] not in (None,'Z','X'))
  assert alive
  raw_env=(proc/'environ').read_bytes()
  env={}
  for item in raw_env.split(b'\0'):
   if b'=' not in item:continue
   key,value=item.split(b'=',1)
   try:name=key.decode('utf-8')
   except Exception:continue
   if name in EAGER_ENV_KEYS:env[name]=value.decode('utf-8','replace')
  assert env.get('SGLANG_EAGER_INPUT_NO_COPY')=='1'
  assert env.get('GLM53_EAGER_INPUT_AUDIT')=='1'
  assert env.get('GLM53_EAGER_INPUT_AUDIT_ROOT')==EAGER_ROOT_IN_CONTAINER
  assert env.get('GLM53_NODE')==str(NODE)
  assert env.get('TASK_QUEUE_ENABLE')=='0'
  assert env.get('HCCL_BUFFSIZE')=='256'
  workers.append({'node':NODE,'namespace_pid':pid,'host_pid':host_pid,'device_id':event['device_id'],'session':event.get('session'),'ns_pid_chain':nspid,'cgroup_contains_current_cid':True,'alive':True,'cmdline_argc':len(argv),'selected_environment':env,'other_environment_retained':False,'full_environment_hashed':False})
 assert len({x['host_pid'] for x in workers})==16
 assert sorted(x['device_id'] for x in workers)==list(range(16))
 audit_root=pathlib.Path(EAGER_AUDIT_HOST_ROOT)
 files=[];records=[];file_errors=[]
 if audit_root.exists():
  for path in sorted(audit_root.glob('pid-*.jsonl')):
   try:
    assert path.is_file() and not path.is_symlink()
    raw=path.read_bytes();assert len(raw)<=8*1024**2
    assert not raw or raw.endswith(b'\n')
    parsed=[]
    for number,line in enumerate(raw.splitlines(),1):
     try:
      row=json.loads(line)
      assert row.get('schema')=='glm53.L17.eager_input_audit.v2'
      assert row.get('audit_version')=='v2' and row.get('project_id')=='glm53'
      assert row.get('source_sha256')==EAGER_SOURCE_SHA==row.get('eager_source_sha256')
      assert row.get('environ_source_sha256')==ENVIRON_SOURCE_SHA
      assert row.get('node')==NODE and row.get('pid') in sdk_pids
      assert row.get('native_no_copy_get') is True
      assert row.get('clone_distinct') is True
      assert type(row.get('is_draft_worker')) is bool
      assert type(row.get('batch_size')) is int and row['batch_size']>=1
      assert isinstance(row.get('forward_mode'),str) and row['forward_mode']
      assert type(row.get('dataclass_field_count')) is int and row['dataclass_field_count']>0
      verdicts=row.get('field_verdicts');assert isinstance(verdicts,dict) and verdicts
      assert all(type(x) is int and x>=0 for x in verdicts.values())
      assert sum(verdicts.values())==row['dataclass_field_count']
      assert isinstance(row.get('field_names_sha256'),str) and len(row['field_names_sha256'])==64
      allow=row.get('allowlisted');assert isinstance(allow,dict)
      assert set(allow)==set(AUDIT_ALLOWLIST_FIELDS)
      parsed.append(row)
     except Exception as error:
      file_errors.append({'path':str(path),'line':number,'error':repr(error),'raw_line':line.decode('utf-8','replace')[:512]})
    files.append({'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'complete_line_ending':True,'rows':len(parsed)})
    for number,row in enumerate(parsed,1):records.append({'path':str(path),'line':number,'row':row})
   except Exception as error:
    file_errors.append({'path':str(path),'error':repr(error)})
 assert len(files)<=64
 by_pid=collections.defaultdict(list)
 for item in records:by_pid[item['row']['pid']].append(item)
 assert not (set(by_pid)-sdk_pids)
 def productive(row):
  return row.get('is_draft_worker') is True and row.get('forward_mode')=='DECODE' and row.get('batch_size')==2 and row.get('native_no_copy_get') is True and row.get('clone_distinct') is True
 def tensor_meta(row,name):
  value=row.get('allowlisted',{}).get(name)
  if not isinstance(value,dict) or value.get('tensor') is not True:return None
  shape=value.get('shape');stride=value.get('stride');dtype=value.get('dtype');device=value.get('device')
  if not (isinstance(shape,list) and isinstance(stride,list) and len(shape)==len(stride) and all(type(x) is int and x>=0 for x in shape) and all(type(x) is int for x in stride)):return None
  if not (isinstance(dtype,str) and dtype and isinstance(device,str) and device):return None
  return {'shape':shape,'dtype':dtype,'device':device,'stride':stride}
 flag_true=[pid for pid in sdk_pids if any(item['row'].get('native_no_copy_get') is True for item in by_pid.get(pid,[]))]
 clone_ok=[pid for pid in sdk_pids if any(item['row'].get('clone_distinct') is True and item['row'].get('native_no_copy_get') is True for item in by_pid.get(pid,[]))]
 bs2_rows=[]
 for pid in sdk_pids:
  for item in by_pid.get(pid,[]):
   if productive(item['row']):bs2_rows.append(item)
 bs2=[pid for pid in sdk_pids if any(productive(item['row']) for item in by_pid.get(pid,[]))]
 out_cache=[]
 for item in bs2_rows:
  meta=tensor_meta(item['row'],'out_cache_loc')
  if meta is not None:out_cache.append({'pid':item['row']['pid'],'file':item['path'],'line':item['line'],'out_cache_loc':meta})
 out_cache_pids={x['pid'] for x in out_cache}
 coverage=bool(len(workers)==16 and not file_errors and set(flag_true)==sdk_pids and set(clone_ok)==sdk_pids and set(bs2)==sdk_pids and out_cache_pids==sdk_pids)
 out['D_eager_input_audit']={'applicable':True,'audit_root_host':str(audit_root),'audit_root_container':EAGER_ROOT_IN_CONTAINER,'files':files,'file_errors':file_errors,'worker_environment_and_identity':workers,'records':records,'coverage':{'workers':len(workers),'flag_true_workers':len(flag_true),'clone_ok_workers':len(clone_ok),'draft_decode_bs2_workers':len(bs2),'out_cache_loc_workers':len(out_cache_pids),'bs2_decode_rows':len(bs2_rows),'out_cache_loc_rows':len(out_cache),'all_16_workers':len(workers)==16,'all_flag_true':set(flag_true)==sdk_pids,'all_clone_ok':set(clone_ok)==sdk_pids,'all_bs2_draft_decode':set(bs2)==sdk_pids,'all_out_cache_loc_tensor_metadata':out_cache_pids==sdk_pids,'no_row_errors':not file_errors},'out_cache_loc_tensors':out_cache,'coverage_pass':coverage,'coverage_status':'COMPLETE_LOCAL16_BS2_DRAFT_DECODE' if coverage else 'PENDING_OR_INCOMPLETE_LOCAL16_BS2_DRAFT_DECODE','coverage_scope':'NODE_LOCAL_16_ACTUAL_SDK_WORKERS','global32_join_scope':'COMPOSITE_NODE_PLUS_NAMESPACE_PID_ONLY; namespace PIDs may overlap across D nodes and are never pooled by PID alone','scope':'Only actual per-PID first-use rows are joined within this node; no row is inferred, filled or clamped. out_cache_loc shape/dtype/stride are reported as recorded, with no contiguity claim.'}
 if D_EAGER_REQUIRED:
  assert coverage,'D eager-input all32 bs2 draft-DECODE coverage is required for this capture'
'''

P_MAP_BLOCK = r'''
out['native_optional_extension_worker_maps']=[]
if NATIVE_P:
 sdk_pids={event['pid'] for item in out['persistent_audit'].values() for event in item['rows'] if event['event']=='initialize_complete'}
 for row in rows:
  if row['namespace_pid'] not in sdk_pids:continue
  raw=(pathlib.Path('/proc')/str(row['host_pid'])/'maps').read_bytes();assert len(raw)<=8*1024**2
  paths=sorted({line.split(maxsplit=5)[5].strip() for line in raw.decode().splitlines() if 'libmf_hybm_accoffload.so' in line})
  assert paths==['/usr/local/python3.12.13/lib/python3.12/site-packages/memfabric_hybrid/lib/libmf_hybm_accoffload.so']
  out['native_optional_extension_worker_maps'].append({'namespace_pid':row['namespace_pid'],'host_pid':row['host_pid'],'loaded_extension_paths':paths,'process_maps_sha256':hashlib.sha256(raw).hexdigest()})
 assert len(sdk_pids)==16 and len(out['native_optional_extension_worker_maps'])==16
'''


def _load_base():
    return contract.load_module("l17_old_capture_base", contract.OLD_CAPTURE_BASE, contract.OLD_CAPTURE_BASE_SHA)


def _load_identity():
    return contract.load_module("l17_mixed_identity", contract.MIXED_IDENTITY, contract.MIXED_IDENTITY_SHA)


def _host_template(old):
    text = old.HOST.replace(
        "root=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261018T-L10-sglang-C32-target-graph-fullmesh-v3');",
        "root=pathlib.Path(NODE_REMOTE_ROOT);",
    )
    assert text != old.HOST
    stream = old.STREAM.replace(
        "if 'GLM53_C32_PACKED_DISPATCH_' in line",
        "if 'GLM53_C32_NATIVE_MC_MAPPING_REUSED=' in line or 'GLM53_C32_NATIVE_HICACHE_' in line or 'GLM53_NATIVE_HICACHE_FATAL' in line or 'GLM53_EAGER_INPUT' in line or 'GLM53_C32_PACKED_DISPATCH_' in line",
    )
    assert stream != old.STREAM
    text = text + P_MAP_BLOCK + D_EAGER_BLOCK
    return text, stream


def plans(capture_id, mode):
    assert mode in MODES, mode
    assert pathlib.Path(capture_id).name == capture_id
    assert capture_id.startswith("20261019T-L17-")
    old = _load_base()
    identity = _load_identity()
    p_capacity, d_capacity = contract.check_capacity_pair()
    text, stream = _host_template(old)
    codes = {}
    rows = contract.combined_providers()
    for node in contract.ALL_NODES:
        item = rows[str(node)]
        is_d = node in contract.D_NODES
        remote_root = item["remote_root"]
        values = {
            "CID": item["container_id"],
            "NAME": item["name"],
            "IMAGE": contract.IMAGE,
            "NODE": node,
            "IDENTITY": identity.render(node, p_capacity, d_capacity, "candidate"),
            "IDENTITY_ENV": {},
            "STREAM": stream,
            "SDK_SOURCE": old.SDK_SOURCE,
            "PORT": 30500 if node in contract.P_NODES else 30600,
            "META": contract.META_CID,
            "ARCHIVE": "/data/wio/inference-foundry/glm53/runs/" + capture_id,
            "FORMAL_RID": contract.FORMAL_RUN,
            "CONCURRENCY": contract.CONCURRENCY,
            "NEED_FORMAL": mode == "formal",
            "NODE_REMOTE_ROOT": remote_root,
            "NATIVE_P": node in contract.P_NODES,
            "NATIVE_D": is_d,
            "D_EAGER_REQUIRED": is_d and mode in ("postfunctional", "formal"),
            "D_EAGER_REQUIRED_ANY": mode in ("postfunctional", "formal"),
            "EAGER_AUDIT_HOST_ROOT": remote_root + "/eager-input-audit",
            "EAGER_ROOT_IN_CONTAINER": contract.EAGER_ROOT_IN_CONTAINER,
            "EAGER_SOURCE_SHA": contract.EAGER_SOURCE_SHA,
            "ENVIRON_SOURCE_SHA": contract.ENVIRON_SOURCE_SHA,
            "EAGER_ENV_KEYS": contract.D_EAGER_ENV_KEYS,
            "AUDIT_ALLOWLIST_FIELDS": contract.AUDIT_ALLOWLIST_FIELDS,
            "CAPTURE_MODE": mode,
            "CAPTURE_ID": capture_id,
        }
        code = old.render(text, values)
        ast.parse(code)
        codes[str(node)] = code
    cpu = None
    if mode == "formal":
        cpu = old.render(old.CPU_HOST, {
            "ARCHIVE": "/data/wio/inference-foundry/glm53/runs/" + capture_id,
            "FORMAL_RID": contract.FORMAL_RUN,
            "CONCURRENCY": contract.CONCURRENCY,
        })
        ast.parse(cpu)
    return old, codes, cpu


def plan_hashes(capture_id, mode):
    _, codes, cpu = plans(capture_id, mode)
    hashes = {node: hashlib.sha256(code.encode()).hexdigest() for node, code in codes.items()}
    if cpu is not None:
        hashes["43"] = hashlib.sha256(cpu.encode()).hexdigest()
    return hashes


def _invoke_with_raw_failure(old, node, code, archive, run):
    pod, host = old.base.NODES[node]
    mapped = old.base.invoke(
        ["kubectl get pod -n default " + shlex.quote(pod) + " -o jsonpath='{.status.hostIP}'"],
        timeout=30,
    )
    (run / ("node-" + str(node) + "-mapping-wire.json")).write_text(json.dumps(mapped, indent=2) + "\n")
    assert mapped["rc"] == 0 and mapped["stdout"] == "172.16.12." + str(node)
    code = "import socket\nassert socket.gethostname()==" + repr(host) + "\n" + code + old.codec.ENCODER
    ast.parse(code)
    command = shlex.join([
        "kubectl", "exec", "-i", "-n", "default", pod, "-c", "hostctl", "--",
        "nsenter", "-t", "1", "-m", "--", "python3", "-",
    ])
    wire = old.base.invoke([command], code, timeout=900)
    (run / ("node-" + str(node) + "-raw-wire.json")).write_text(json.dumps(wire, indent=2) + "\n")
    if wire["rc"] != 0:
        return wire
    return old.codec.decode(wire)


def execute(capture_id, review_path, mode):
    assert capture_id == MODE_CAPTURE[mode]
    assert review_path == MODE_REVIEW[mode]
    old, codes, cpu = plans(capture_id, mode)
    hashes = {node: hashlib.sha256(code.encode()).hexdigest() for node, code in codes.items()}
    if cpu is not None:
        hashes["43"] = hashlib.sha256(cpu.encode()).hexdigest()
    contract.capture_review_check(review_path, hashes)
    run = ROOT / "runs" / capture_id
    run.mkdir(parents=True, exist_ok=False)
    with (run / "execution-once.lock").open("x") as file:
        file.write(str(os.getpid()))
    receipt = {
        "project_id": contract.PROJECT,
        "capture_id": capture_id,
        "mode": mode,
        "model_requests": 0,
        "SDK_init_calls": 0,
        "max_SSH": 2,
        "nodes": {},
        "D_eager_input_required": mode in ("postfunctional", "formal"),
    }
    archive = "/data/wio/inference-foundry/glm53/runs/" + capture_id
    jobs = {node: code for node, code in codes.items()}
    if cpu is not None:
        jobs["43"] = cpu
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            pending = {
                pool.submit(_invoke_with_raw_failure, old, int(node), code, archive, run): node
                for node, code in jobs.items()
            }
            for future in concurrent.futures.as_completed(pending):
                node = pending[future]
                try:
                    result = future.result()
                except Exception as error:
                    result = {"error": repr(error)}
                (run / ("node-" + str(node) + ".json")).write_text(json.dumps(result, indent=2) + "\n")
                receipt["nodes"][str(node)] = result.get("rc", result.get("error"))
                print(node, receipt["nodes"][str(node)], flush=True)
        expected_nodes = {str(n) for n in contract.ALL_NODES}
        if cpu is not None:
            expected_nodes.add("43")
        assert set(receipt["nodes"]) == expected_nodes
        assert all(result == 0 for result in receipt["nodes"].values())
        receipt["pass"] = True
    finally:
        (run / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    result = {
        "project_id": contract.PROJECT,
        "status": STATUS,
        "modes": list(MODES),
        "capture_ids": MODE_CAPTURE,
        "review_paths": MODE_REVIEW,
        "planned_hashes": {mode: plan_hashes(capture_id, mode) for mode, capture_id in MODE_CAPTURE.items()},
        "runtime_authorized": False,
        "model_requests": 0,
        "SDK_init_calls": 0,
        "max_SSH_future_execution": 2,
        "SSH": 0,
        "network": 0,
        "Docker": 0,
        "NPU": 0,
        "Git": 0,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
