from pathlib import Path
import os,sys,subprocess,json
j=Path(sys.argv[1]);runtime="/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime"
env=dict(os.environ,PYTHONPATH=runtime+":"+os.environ.get("PYTHONPATH",""))
argv=[sys.executable,"/data/tiankuan/wio/glm52-pd/deploy/plugins/dp_run89/native_acl_lifecycle.py","script",runtime+"/service_entry.py","--help"]
z=subprocess.run(argv,env=env,capture_output=True,timeout=90);(j/"child.stdout").write_bytes(z.stdout);(j/"child.stderr").write_bytes(z.stderr);z.check_returncode()
acks=[json.loads(x)for x in z.stdout.decode().splitlines()if x.startswith('{"event":')];assert acks[0]["event"]=="task_acl_init"and acks[-1]["event"]=="task_acl_finalize"
assert "--config"in z.stdout.decode()and "--port"in z.stdout.decode()
(j/"child_proof.json").write_text(json.dumps(dict(argv=argv,child_exit=z.returncode,SDK_init=acks[0]["returncode"],SDK_finalize=acks[-1]["returncode"],preserve_CANN_PYTHONPATH=True,native_inference=0)))
