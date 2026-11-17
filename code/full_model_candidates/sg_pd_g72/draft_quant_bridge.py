"""Embedded DSpark FLOAT weights must not inherit the target-only ascend tag."""
from pathlib import Path
import functools,hashlib,json,os,struct

NATIVE_SHA='0941ffadb4d8075768d6d97e13b06d9624806889c92be9d6ce13e76d5d87ed6e'
DESCRIPTION_SHA='1bc65590c213a620738477020127e99a8df5394cfec284198005df33d526178c'


@functools.lru_cache(maxsize=1)
def verified_embedded_draft(model_root):
    root=Path(model_root)
    data=(root/'quant_model_description.json').read_bytes()
    if hashlib.sha256(data).hexdigest()!=DESCRIPTION_SHA:raise RuntimeError('Pinned ModelSlim description changed')
    desc=json.loads(data);mtp={k:v for k,v in desc.items() if k.startswith('mtp.')}
    if len(mtp)!=1226 or set(mtp.values())!={'FLOAT'}:raise RuntimeError('Embedded draft is not the pinned unquantized 1226-entry draft')
    if {k.split('.')[1] for k in mtp}!={'0','1','2'}:raise RuntimeError('Expected all three embedded MTP blocks')
    index=json.loads((root/'quant_model_weights.safetensors.index.json').read_text())['weight_map']
    if set(mtp)!={k for k in index if k.startswith('mtp.')}:raise RuntimeError('Draft index and description disagree')
    headers={};dtypes={};logical_bytes=0
    for name in mtp:
        shard=index[name]
        if shard not in headers:
            with (root/shard).open('rb') as f:
                n=struct.unpack('<Q',f.read(8))[0]
                if n>268435456:raise RuntimeError('Unexpected safetensors header size')
                headers[shard]=json.loads(f.read(n))
        entry=headers[shard][name];dtype=entry['dtype']
        if dtype not in {'BF16','F32','F16','I64','I32','BOOL'}:raise RuntimeError('Packed/unknown dtype in unquantized draft: '+name+' '+dtype)
        dtypes[dtype]=dtypes.get(dtype,0)+1
        logical_bytes+=entry['data_offsets'][1]-entry['data_offsets'][0]
    return {'entries':len(mtp),'description_sha256':DESCRIPTION_SHA,'dtype_counts':dtypes,'logical_checkpoint_bytes':logical_bytes,'stages':[0,1,2]}


def install(root):
    import sglang.srt.configs.model_config as native
    if getattr(native.ModelConfig,'_ds41_embedded_float_installed',False):return
    if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()!=NATIVE_SHA:raise RuntimeError('Native ModelConfig source changed')
    original=native.ModelConfig._parse_quant_hf_config
    @functools.wraps(original)
    def parse(self):
        if (self.is_draft_model and self.speculative_algorithm=='DSPARK'
            and getattr(self.hf_config,'model_type',None)=='deepseek_v41'
            and Path(self.model_path).resolve()==Path('/models/ds41').resolve()):
            if self.quantization is not None:raise RuntimeError('Embedded FLOAT draft must retain native unquantized selection')
            report=verified_embedded_draft(self.model_path)
            report={**report,'native_source_sha256':NATIVE_SHA,'target_modelslim_unchanged':True,'scope':'Suppress target-only ascend quant tag for pinned embedded FLOAT draft; native unquantized layers and loaders retained','full_model_graph_PD_validated':False}
            (root/f'embedded_draft_quant.pid{os.getpid()}.json').write_text(json.dumps(report,indent=2))
            return None
        return original(self)
    native.ModelConfig._parse_quant_hf_config=parse
    native.ModelConfig._ds41_embedded_float_installed=True
