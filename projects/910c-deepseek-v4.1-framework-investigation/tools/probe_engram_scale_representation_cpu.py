#!/usr/bin/env python3
"""Exhaustive CPU representation check in frozen Torch; no NPU context."""
import hashlib
import json
from pathlib import Path
import subprocess

PROBE = r'''
import hashlib,json,torch
raw=torch.arange(256,dtype=torch.int16).to(torch.uint8)
original=raw.view(torch.float8_e8m0fnu).float()
e=raw.to(torch.int32)
bits=e<<23
bits=torch.where(e==0,torch.full_like(bits,0x00400000),bits)
bits=torch.where(e==255,torch.full_like(bits,0x7fc00000),bits)
decoded=bits.view(torch.float32)
diff=(original.view(torch.int32)!=decoded.view(torch.int32)).nonzero().flatten().tolist()
corrected=bits.clone()
for i in diff:corrected[i]=original.view(torch.int32)[i]
def digest(t):return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
codes=torch.arange(-128,128,dtype=torch.int16).float()
old=(codes[:,None]*original[None,:]).bfloat16()
new=(codes[:,None]*corrected.view(torch.float32)[None,:]).bfloat16()
print(json.dumps(dict(torch_version=torch.__version__,input_scale_encodings=256,
 old_loaded_FP32_sha256=digest(original),existing_MXFP8_decoder_FP32_sha256=digest(decoded),
 existing_decoder_bit_mismatches=[dict(encoding=i,loaded_FP32_bits=hex(int(original.view(torch.int32)[i])&0xffffffff),decoder_FP32_bits=hex(int(bits[i])&0xffffffff)) for i in diff],
 corrected_decoder_bit_mismatches=int((original.view(torch.int32)!=corrected).sum()),
 corrected_FP32_sha256=digest(corrected),
 signed_int8_by_scale_pairs=65536,CPU_BF16_product_bit_mismatches=int((old.view(torch.int16)!=new.view(torch.int16)).sum()),
 product_sha256=digest(old),npu_initialized=False,
 limits=['CPU representation enumeration only; no Triton/NPU primitive equivalence, actual UVA allocation, timing or model claim.','Existing decoder is for MXFP8 codes; new INT8+E8M0 route needs independent code-type/scale-type selectors.'])))
'''
REMOTE = r'''
import json,subprocess,sys
p=subprocess.run(['docker','exec','-e','TORCH_DEVICE_BACKEND_AUTOLOAD=0','-e','PYTHONDONTWRITEBYTECODE=1','-e','OMP_NUM_THREADS=1','wio-vllm-v41-repro171-20261102','python3','-B','-c',sys.argv[1]],capture_output=True,text=True,timeout=40)
assert p.returncode==0,p.stderr
print(p.stdout)
'''
if __name__ == '__main__':
    p=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-c',REMOTE,PROBE],capture_output=True,text=True,timeout=55)
    assert p.returncode==0,p.stderr
    d=json.loads(p.stdout);d.update(probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                  remote_writes=[],new_NPU_work=False,service_requests=0)
    dest=Path('audit/execution-search/engram-scale-representation-cpu.json')
    dest.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(d,ensure_ascii=False))
