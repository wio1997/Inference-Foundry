"""Use the pinned native DSV4 name mapper for this checkpoint's descriptors."""
from pathlib import Path
import functools,hashlib,json,os
NATIVE_SHA='03fc670afe3cb769e05b4b459b6f08efa5c7001d4014b8812bdde87a15b6f7be'


def install(root):
    import sglang.srt.layers.quantization.modelslim.modelslim as module
    if getattr(module.ModelSlimConfig,'_ds41_pinned_aliases',False):return
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=NATIVE_SHA:raise RuntimeError('Native ModelSlim source changed')
    from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
    original=module.ModelSlimConfig.__init__
    @functools.wraps(original)
    def initialize(self,quant_config={}):
        hf=quant_config.get('hf_config')
        input_report={'input_key_count':len(quant_config),'input_keys':list(quant_config)[:20],'input_quant_method':quant_config.get('quant_method'),'hf_model_type':getattr(hf,'model_type',None)}
        if quant_config.get('quant_method')=='ascend' and getattr(hf,'model_type',None)=='deepseek_v41':
            from draft_quant_bridge import DESCRIPTION_SHA
            data=Path('/models/ds41/quant_model_description.json').read_bytes()
            if hashlib.sha256(data).hexdigest()!=DESCRIPTION_SHA:raise RuntimeError('Pinned ModelSlim description changed')
            resolved=json.loads(data)
            if 'layers.0.ffn.experts.0.w1.weight' not in resolved:raise RuntimeError('Missing target descriptor source')
            resolved['packed_modules_mapping']=quant_config.get('packed_modules_mapping',{})
            resolved['hf_config']=hf
            quant_config=resolved
            input_report['resolved_description_sha256']=DESCRIPTION_SHA
            input_report['resolved_descriptor_count']=len(resolved)
        (root/f'target_quant_input.pid{os.getpid()}.json').write_text(json.dumps(input_report,indent=2))
        if (quant_config.get('layers.0.ffn.experts.0.w1.weight')=='W4A8_DYNAMIC'
            and quant_config.get('layers.1.engram.embed.quant_type')=='INT8_SYM'
            and sum(k.startswith('mtp.') for k in quant_config if isinstance(k,str))==1226):
            mapped=dict(quant_config);aliases=0
            for key,value in quant_config.items():
                if not isinstance(key,str) or not key.startswith(('layers.','embed.','head.')) and key!='norm.weight':continue
                name=DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format(key)
                if name in mapped and mapped[name]!=value:raise RuntimeError('Conflicting quantization descriptor alias: '+name)
                if name not in mapped:mapped[name]=value;aliases+=1
            if mapped.get('model.layers.0.mlp.experts.0.gate_proj.weight')!='W4A8_DYNAMIC':raise RuntimeError('Routed expert W4A8 descriptor lost')
            if mapped.get('model.layers.0.self_attn.wq_a.weight')!='FLOAT':raise RuntimeError('FLOAT attention descriptor lost')
            (root/f'target_quant_aliases.pid{os.getpid()}.json').write_text(json.dumps({'native_source_sha256':NATIVE_SHA,'added_aliases':aliases,'existing_entries_retained':len(quant_config),'only_parameter_names_mapped':True,'full_weight_load_or_performance_validated':False},indent=2))
            return original(self,mapped)
        return original(self,quant_config)
    module.ModelSlimConfig.__init__=initialize
    module.ModelSlimConfig._ds41_pinned_aliases=True
