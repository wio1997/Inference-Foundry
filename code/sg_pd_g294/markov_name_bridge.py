"""Reuse the fixed image's generic DSpark mapping for two checkpoint Markov names."""
from pathlib import Path
import functools,hashlib,json,os


def install(root):
    import sglang.srt.layers.quantization.modelslim.modelslim as quant
    import sglang.srt.models.deepseek_v4_dspark as draft
    if getattr(draft,'_ds41_markov_names_installed',False):return
    expected=json.loads((root/'native_source_sha256.json').read_text())
    if hashlib.sha256(Path(draft.__file__).read_bytes()).hexdigest()!=expected['deepseek_v4_dspark.py']:raise RuntimeError('Native DSpark source changed')
    original=quant.remap_dspark_weight_name_npu
    pinned={'mtp.2.markov_head.embed.weight','mtp.2.markov_head.head.weight'}
    generic=draft.DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name
    @functools.wraps(original)
    def remap(name,*,for_quant_config=False):
        if name in pinned:
            mapped=generic(None,name)
            if mapped not in {'markov_head.markov_w1.weight','markov_head.markov_w2.weight'}:raise RuntimeError('Unexpected native generic Markov mapping')
            return mapped
        return original(name,for_quant_config=for_quant_config)
    assert draft.remap_dspark_weight_name_npu is original
    quant.remap_dspark_weight_name_npu=remap
    draft.remap_dspark_weight_name_npu=remap
    draft._ds41_markov_names_installed=True
    receipt={name:{'before':original(name),'after':remap(name)} for name in sorted(pinned)}
    (root/f'Markov_name_bridge.pid{os.getpid()}.json').write_text(json.dumps({'mapping':receipt,'native_generic_mapper_reused':True,'strict_load_coverage_unchanged':True,'model_graph_PD_or_quality_validated':False},indent=2))
