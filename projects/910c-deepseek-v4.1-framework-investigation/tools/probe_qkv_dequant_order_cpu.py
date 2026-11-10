#!/usr/bin/env python3
"""Finite scalar falsification of scale reassociation; not a CANN/NPU bit gate."""
from hashlib import sha256
from pathlib import Path
import json,random,struct

def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def bits(x):return struct.unpack('<I',struct.pack('<f',x))[0]
def bf16(x):
    u=bits(x);return (u+0x7fff+((u>>16)&1))>>16
def frombf16(u):return struct.unpack('<f',struct.pack('<I',u<<16))[0]

def main():
    rng=random.Random(20261110);found=None;count=0
    for count in range(1,1000001):
        dot=rng.randint(1,20000000)*(1 if rng.randrange(2) else -1)
        weight=frombf16(rng.randrange(0x3900,0x3d80))
        maximum=frombf16(rng.randrange(0x3f80,0x4181))
        token=f32(maximum*f32(1/127))
        integer=f32(dot)
        dequant_first=f32(f32(integer*weight)*token)
        scales_first=f32(integer*f32(weight*token))
        a,b=bf16(dequant_first),bf16(scales_first)
        if a==b:continue
        full,rem=divmod(abs(dot),127*127);q2,r2=divmod(rem,127);products=[(127,127)]*full+([(q2,127)] if q2 else [])+([(r2,1)] if r2 else [])
        assert len(products)<=1280 and sum(x*y for x,y in products)==abs(dot)
        found=dict(dot_int32=dot,integer_cast_fp32=integer,weight_scale_bf16_hex=hex(bits(weight)>>16),weight_scale=weight,input_max_bf16_hex=hex(bits(maximum)>>16),input_max=maximum,token_scale_fp32_hex=hex(bits(token)),token_scale=token,
            dequant_then_token=dict(fp32_hex=hex(bits(dequant_first)),bf16_hex=hex(a)),scales_then_dot=dict(fp32_hex=hex(bits(scales_first)),bf16_hex=hex(b)),
            dot_constructible_from_INT8_K1280=dict(nonzero_products=len(products),absolute_dot=sum(x*y for x,y in products),max_operand=127,padding_zeros=1280-len(products),sign_of_one_operand=-1 if dot<0 else 1))
        break
    assert found is not None,'No finite final-BF16 counterexample found in fixed search'
    out=dict(status='CPU_FINITE_FINAL_BF16_COUNTEREXAMPLE_TO_REASSOCIATION',trials=count,counterexample=found,
        arithmetic='IEEE FP32 rounding emulated via struct after every binary op; double exact for products of two FP32 finite values in sampled range, then FP32 RNE. BF16 RNE conversion. No FMA.',
        qualification='Disproves universal equality of pointwise typed formulas, not native CANN AscendDequant implementation or loaded original/new key. Not model-observed, not NPU test or admitted replacement primitive. Token scale chosen from a finite BF16 max times rounded FP32(1/127); does not certify a complete original norm/DQ invocation.',
        source_motivation='Installed CANN9.2 pertoken/basic calls AscendDequant then token multiplication; pertoken-opt precombines scales then multiplies FP32-cast INT32 accumulator. Active key must be bound; preserve original ordering.',
        bound=dict(K=1280,max_abs_INT8_product=16384,absolute_partial_sum_bound=20971520,int32_no_overflow=True,all_integer_values_FP32_exact=False),
        new_NPU_work=0,new_primitive_admissions=0,every_replacement_bitwise=True,whole_chain_complete=False,probe_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    p=Path('audit/execution-search/qkv-dequant-order-cpu.json');p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
