import json,socket,subprocess
host=socket.gethostname()
assert host in ('B4-33U-Atlas900RCKA3-32','B5-5U-Atlas900RCKA3-33')
smi=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=25)
containers=subprocess.run(['docker','ps','--no-trunc','--format','{{json .}}'],capture_output=True,text=True,timeout=20)
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'hostname':host,'npu_smi_rc':smi.returncode,'npu_smi_stdout':smi.stdout,'npu_smi_stderr':smi.stderr,'running_containers':[json.loads(r) for r in containers.stdout.splitlines()],'read_only':True,'models_started':False},indent=2)+'\nGLM53_RESULT_JSON_END')
