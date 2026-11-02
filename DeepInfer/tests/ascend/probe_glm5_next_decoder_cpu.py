# SPDX-License-Identifier: Apache-2.0
"""CPU-only preparation probe: actual mHC/norms and bounded dense metadata.

No NPU execution and no dequantization/full-model forward. This does not
validate the quantized MLP or whole decoder on hardware.
"""
import argparse
import datetime
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

import torch
from deepinfer.hugging_face.modelslim910c import ModelSlimCheckpoint
from deepinfer.model_executor.models.glm5_next.mhc import GlmMHC,rms_norm
from deepinfer.model_executor.models.glm5_next.decoder import Glm5NextDenseDecoder

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--reference-mhc',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True)
cli=parser.parse_args();cli.output_dir.mkdir(parents=True,exist_ok=True)
assert not list(Path('/dev').glob('davinci[0-9]*')), 'CPU preparation requires a device-free container'
assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0'
torch.set_num_threads(2);torch.manual_seed(5302)
spec=importlib.util.spec_from_file_location('glm_mhc_reference',cli.reference_mhc)
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
reader=ModelSlimCheckpoint(cli.checkpoint)
src=Path(__file__).resolve().parents[2]
report={'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'architecture':platform.machine(),
        'execution':'CPU only; no NPU devices or torch_npu import','torch':torch.__version__,
        'NPU_execution':False,'whole_decoder_validated':False,
        'reference_sha256':hashlib.sha256(cli.reference_mhc.read_bytes()).hexdigest(),
        'config_sha256':hashlib.sha256((cli.checkpoint/'config.json').read_bytes()).hexdigest(),
        'checks':{},'dense_contracts':{},'source_sha256':{}}
for rel in ['deepinfer/model_executor/models/glm5_next/mhc.py','deepinfer/model_executor/models/glm5_next/decoder.py','tests/ascend/probe_glm5_next_decoder_cpu.py']:
    report['source_sha256'][rel]=hashlib.sha256((src/rel).read_bytes()).hexdigest()

def check(name,actual,expected,atol=0.,rtol=0.):
    diff=(actual.float()-expected.float()).abs()
    report['checks'][name]={'shape':list(actual.shape),'max_abs':float(diff.max()) if diff.numel() else 0.,'atol':atol,'rtol':rtol}
    torch.testing.assert_close(actual,expected,atol=atol,rtol=rtol)
    report['checks'][name]['passed']=True

try:
    for layer_id in (0,2):
        for kind in ('attn','ffn'):
            module=GlmMHC.from_checkpoint(reader,layer_id,kind)
            residual=torch.randn(3,4,4096).bfloat16()
            expected_post,expected_comb,expected_input=reference.mhc_pre_torch(residual,module.fn,module.scale,module.base,1e-5,1e-6,1e-6,2.0,20)
            actual_input,mix=module.pre(residual);prefix=f'layer{layer_id}_{kind}'
            check(prefix+'_pre_input',actual_input,expected_input)
            check(prefix+'_post_coeff',mix.post,expected_post)
            check(prefix+'_comb_coeff',mix.comb,expected_comb)
            output=torch.randn(3,4096).bfloat16()
            expected=reference.mhc_post_torch(output,residual,expected_post,expected_comb)
            check(prefix+'_post_output',module.post(output,residual,mix),expected)
            suffix='input_layernorm.weight' if kind=='attn' else 'post_attention_layernorm.weight'
            norm=reader.tensor(f'model.language_model.layers.{layer_id}.{suffix}')[0]
            check(prefix+'_rmsnorm',rms_norm(actual_input,norm,1e-5),
                  (expected_input.float()*torch.rsqrt(expected_input.float().square().mean(-1,keepdim=True)+1e-5)*norm.float()).bfloat16())
    for layer_id in (0,1,2):
        contract={}
        for name in ('gate_proj','up_proj','down_proj'):
            key=f'model.language_model.layers.{layer_id}.mlp.{name}.weight'
            desc=reader.descriptor(key)[2]
            expected=[4096,12288] if name=='down_proj' else [12288,4096]
            assert desc['shape']==expected and desc['dtype']=='I8' and reader.quant[key]=='W8A8_DYNAMIC'
            contract[name]={'shape':desc['shape'],'storage':desc['dtype'],'recipe':reader.quant[key]}
        report['dense_contracts'][str(layer_id)]=contract
        try:Glm5NextDenseDecoder.from_checkpoint(reader,layer_id,'cpu',kda_backend='torch')
        except ValueError as exc:
            assert 'no CPU dequantization fallback' in str(exc)
            report['checks'][f'layer{layer_id}_CPU_INT8_guard']={'passed':True,'reason':str(exc)}
        else:raise AssertionError('quantized CPU decoder unexpectedly instantiated')
    try:Glm5NextDenseDecoder.from_checkpoint(reader,3,'cpu',kda_backend='torch')
    except ValueError:report['checks']['DSA_MoE_layer_rejected']={'passed':True}
    else:raise AssertionError('unsupported layer accepted')
    assert 'torch_npu' not in sys.modules
    report['status']='passed'
except BaseException as exc:
    report['status']='failed';report['failure']=repr(exc);raise
finally:
    report['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    p=cli.output_dir/'cpu-decoder-result.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(p)
    print(json.dumps(report,indent=2))
