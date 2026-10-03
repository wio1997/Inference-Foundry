import json,time
print(json.dumps(dict(event="native_client_CPU_probe",simulation=False,inference=0,NPU_tensors=0,models_created=0)),flush=True)
time.sleep(2)
