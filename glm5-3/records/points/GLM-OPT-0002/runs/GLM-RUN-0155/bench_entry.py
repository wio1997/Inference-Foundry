"""Actual AISBench adapter with phase-scoped raw receipts on retained public service."""
import json,os,sys,hashlib
from pathlib import Path
g=Path("/data/tiankuan/wio/Inference-Foundry/glm5-3");sys.path.insert(0,str(g/"runtime"));sys.path.insert(0,str(g/"adapters/aisbench"))
import prefix_bench
from protocol_receipts import verify_protocol_receipts
# Reuse immutable152 inputs, keeping native tokenizer validation in prefix_bench.
SALT="GLM-RUN-0155-prefix-tail4096-20261003T0624Z"
dataset=Path("/data/tiankuan/wio/Inference-Foundry/glm5-3/records/points/GLM-OPT-0002/runs/GLM-RUN-0152/formal/benchmark/dataset")
DATA=[("prefix-GSM8K-in73728-num2-GLM-5.2-w8a8.jsonl",943912,"2df32b5ebf0e9d58c1d5cd3a39463b7b47911874b72910dcaa70d8b4b7e93d6d"),("GSM8K-in81920-num4-GLM-5.2-w8a8-repeatRate0.9.jsonl",2097102,"c8cbc3f51fbc452882f031497d65ca6163fd1bd774f17e9d9106ec1e48d7fb4a")]
def reuse_dataset(**kwargs):
 assert kwargs["input_len"]==81920and kwargs["number"]==4and kwargs["dp"]==2and kwargs["seed"]==20260930and kwargs["repeat_rate"]==0.9
 out=[]
 for name,size,sha in DATA:
  file=dataset/name;b=file.read_bytes();assert len(b)==size and hashlib.sha256(b).hexdigest()==sha;out.append(str(file))
 Path("dataset_reuse.json").write_text(json.dumps(dict(source_run="GLM-RUN-0152",files=DATA,tokenizer_validation_preserved=True))+"\n")
 return tuple(out)
prefix_bench.create_prefix_dataset=reuse_dataset
native_config=prefix_bench.modify_aisbench_api
def salted_config(*args,**kwargs):
 native_config(*args,**kwargs)
 f=Path("temp_api.py");s=f.read_text();assert s.count("ignore_eos=True")==1
 s=s.replace("ignore_eos=True","ignore_eos=True,\n            cache_salt="+repr(SALT))
 __import__("ast").parse(s);f.write_text(s)
 phase="warmup"if kwargs["output_len"]==1else"full";Path(phase+"_temp_api.py").write_text(s)
prefix_bench.modify_aisbench_api=salted_config
original=prefix_bench.validate_phase_details
def validate(log_dir,phase,expected_count,output_len):
 acceptance=original(log_dir,phase,expected_count,output_len)
 cursor=json.loads(Path(os.environ["GLM_FORMAL_TRACE_CURSOR"]).read_text());source=Path(cursor["path"]);raw=source.read_bytes()
 assert hashlib.sha256(raw[:cursor["bytes"]]).hexdigest()==cursor["sha256"]and raw[cursor["bytes"]-1:cursor["bytes"]]==b"\n"
 scoped=Path(os.environ["GLM_FORMAL_ROUTER_TRACE_PATH"]);scoped.write_bytes(raw[cursor["bytes"]:])
 receipt=verify_protocol_receipts(scoped,expected_count,73740 if phase=="warmup"else 81932,output_len,["D0","D1"])
 counts={key:sum(x["replica"]==key for x in receipt["requests"])for key in ["D0","D1"]}
 assert counts=={"D0":1,"D1":1}if phase=="warmup"else counts=={"D0":2,"D1":2}
 (Path(os.environ["GLM_FORMAL_RECEIPTS_PATH"])/(phase+"_wire_acceptance.json")).write_text(json.dumps(receipt,indent=2)+"\n")
 return acceptance
prefix_bench.validate_phase_details=validate
if __name__=="__main__":sys.exit(prefix_bench.main())
