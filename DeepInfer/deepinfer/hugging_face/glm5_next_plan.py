# SPDX-License-Identifier: Apache-2.0
"""Device-free GLM text-backbone assembly and explicit component requirements.

This module imports no torch/native package and allocates no tensors. A plan is
not a model registration, authorization, dependency/hardware validation or
readiness claim. Assembly invokes only caller-supplied factories; no fallback
loads weights, dequantizes INT8 or substitutes CPU mathematical references.
Embedding/head, vision, MTP, distributed execution and serving are outside scope.
"""
from dataclasses import asdict,dataclass
import hashlib
import json
from types import MappingProxyType

from .glm5_next_config import Glm5NextConfig


@dataclass(frozen=True)
class BackboneOptions:
    logical_block_size: int = 128
    tensor_parallel_size: int = 1
    expert_parallel_size: int = 1
    requests_per_step: int = 1
    vision: bool = False
    mtp_tokens: int = 0
    graphs: bool = False
    prefix_cache: bool = False

    def validate(self):
        if type(self.logical_block_size) is not int or self.logical_block_size<128 or self.logical_block_size%128:
            raise ValueError('CANN backbone plan requires logical blocks divisible by128')
        for name in ('tensor_parallel_size','expert_parallel_size','requests_per_step'):
            if type(getattr(self,name)) is not int or getattr(self,name)!=1:
                raise ValueError(f'{name} is unsupported beyond the single-request unsharded component interface')
        if type(self.mtp_tokens) is not int or self.mtp_tokens!=0:raise ValueError('MTP is not supported by this backbone interface')
        for name in ('vision','graphs','prefix_cache'):
            if type(getattr(self,name)) is not bool or getattr(self,name):raise ValueError(f'{name} is not supported by this backbone interface')


def _fingerprint(config):
    # Includes actual source config and explicit ModelSlim sidecar method, but
    # contains no filesystem location, credentials or runtime resource claims.
    return hashlib.sha256(json.dumps(config.to_dict(),sort_keys=True,separators=(',',':')).encode()).hexdigest()


@dataclass(frozen=True)
class BackboneLayer:
    layer_id: int
    attention: str
    mlp: str
    checkpoint_prefix: str


@dataclass(frozen=True)
class GlmBackbonePlan:
    config_fingerprint: str
    options: BackboneOptions
    layers: tuple[BackboneLayer,...]
    hidden_size: int
    streams: int
    kda_heads: int
    kda_dim: int
    conv_width: int
    latent_dim: int
    index_dim: int
    pool_size: int
    declared_mtp_layers: int
    vision_config_present: bool

    @classmethod
    def from_config(cls,config,options=None):
        options=BackboneOptions() if options is None else options
        if not isinstance(options,BackboneOptions):raise ValueError('explicit BackboneOptions required')
        options.validate()
        if config.quantization_method!='W8A8_DYNAMIC':raise ValueError('explicit ModelSlim W8A8_DYNAMIC configuration is required')
        if config.architecture!='Glm5NextForConditionalGeneration' or config.model_type!='glm5_next':raise ValueError('expected GLM-Next conditional-generation checkpoint')
        if config.dtype!='bfloat16':raise ValueError('backbone component interface requires BF16 model activations')
        text=config.hf_config['text_config'];linear=config.linear_attn_config;dims=config.mla_dimensions;idx=config.indexer_config
        if not text.get('mhc') or type(text.get('hc_mult')) is not int or text['hc_mult']<1:raise ValueError('materialized mHC is required')
        if dims['qk_rope_head_dim']!=0 or not text.get('mla_use_nope'):raise ValueError('explicit noRoPE MLA is required')
        if idx.get('index_kpool')!=4 or not idx.get('index_kpool_compress') or not idx.get('index_kpool_always_select_tail'):
            raise ValueError('GLM compressed pool4 with causal tail is required')
        required={'scoring_func':'sigmoid','topk_method':'noaux_tc','norm_topk_prob':True,'n_group':1,'topk_group':1,'moe_router_dtype':'float32'}
        if any(text.get(k)!=v for k,v in required.items()):raise ValueError('unsupported GLM single-group routing configuration')
        if [s.layer_id for s in config.layers]!=list(range(config.num_hidden_layers)) or not config.layers:
            raise ValueError('backbone layer IDs must be complete, contiguous and ordered')
        layers=tuple(BackboneLayer(s.layer_id,s.attention,s.mlp,f'model.language_model.layers.{s.layer_id}') for s in config.layers)
        if any(s.attention not in ('linear_attention','deepseek_sparse_attention') or s.mlp not in ('dense','sparse') for s in layers):raise ValueError('unknown backbone component kind')
        return cls(_fingerprint(config),options,layers,config.hidden_size,text['hc_mult'],linear['num_heads'],linear['head_dim'],
                   linear['short_conv_kernel_size'],dims['kv_lora_rank'],idx['index_head_dim'],4,config.num_nextn_predict_layers,config.vision_config is not None)

    def to_dict(self):
        result=asdict(self)
        result['scope']='text_backbone_components_only'
        result['model_ready']=False
        result['counts']={kind:sum(getattr(layer,field)==kind for layer in self.layers) for field,kinds in (
            ('attention',('linear_attention','deepseek_sparse_attention')),('mlp',('dense','sparse'))) for kind in kinds}
        return result

    def logical_state_bytes(self,tokens,*,lookahead=0):
        """Arithmetic footprint of current unpaged state, NOT HBM/peak usage.

        Excludes position tags, tensor headers, padding, allocator/staging copies,
        weights, activations, operator scratch, shared pools and device overhead.
        """
        if type(tokens) is not int or tokens<0 or type(lookahead) is not int or lookahead<0:raise ValueError('token count/lookahead must be nonnegative integers')
        kda=sum(s.attention=='linear_attention' for s in self.layers);dsa=len(self.layers)-kda
        conv=kda*3*(self.conv_width-1)*self.kda_heads*self.kda_dim*2
        recurrent=kda*self.kda_heads*self.kda_dim*self.kda_dim*4
        latent=dsa*tokens*self.latent_dim*2
        index=dsa*(tokens//self.pool_size)*self.index_dim*2
        tail=dsa*2*(self.pool_size+lookahead)*self.index_dim*4
        return {'KDA_conv_BF16':conv,'KDA_recurrent_FP32':recurrent,'DSA_latent_BF16':latent,
                'DSA_compressed_BF16':index,'DSA_raw_tail_FP32':tail,'logical_total':conv+recurrent+latent+index+tail}

    def validate_bindings(self,config,*,attention_factories,ffn_factories,decoder_factory):
        """Validate all known prerequisites before the first factory side effect."""
        self.options.validate()
        if _fingerprint(config)!=self.config_fingerprint:raise ValueError('configuration changed after planning')
        # Also reject a manually modified/corrupt plan before factory calls.
        if self!=type(self).from_config(config,self.options):raise ValueError('plan no longer matches configuration')
        for factories,required,label in ((attention_factories,{s.attention for s in self.layers},'attention'),
                                          (ffn_factories,{s.mlp for s in self.layers},'FFN')):
            if not hasattr(factories,'keys') or set(factories)!=required or any(not callable(factories[k]) for k in required):
                raise ValueError(f'{label} factories must explicitly cover exactly the required component kinds')
        if not callable(decoder_factory):raise ValueError('explicit decoder factory is required')

    def assemble(self,config,*,attention_factories,ffn_factories,decoder_factory):
        """Construct caller-owned layers after complete metadata preflight.

        attention/FFN factories take BackboneLayer. decoder_factory takes
        (BackboneLayer, attention, FFN). No implicit weight reader or operator is
        called. Runtime factory failure does not roll back external allocations;
        callers own cleanup. Returning layers does not create a model/engine.
        """
        self.validate_bindings(config,attention_factories=attention_factories,ffn_factories=ffn_factories,decoder_factory=decoder_factory)
        layers={}
        for spec in self.layers:
            attention=attention_factories[spec.attention](spec)
            ffn=ffn_factories[spec.mlp](spec)
            if attention is None or ffn is None:raise ValueError('component factory returned no component')
            decoder=decoder_factory(spec,attention,ffn)
            if getattr(decoder,'layer_id',None)!=spec.layer_id or (getattr(decoder,'attention_kind',None),getattr(decoder,'mlp_kind',None))!=(spec.attention,spec.mlp):
                raise ValueError('decoder factory returned the wrong layer/kind binding')
            if getattr(decoder,'attention',None) is not attention or getattr(decoder,'mlp',None) is not ffn:
                raise ValueError('decoder must retain exactly the supplied attention/FFN components')
            for name in ('attn_hc','ffn_hc'):
                hc=getattr(decoder,name,None)
                if (getattr(hc,'hidden_size',None),getattr(hc,'streams',None))!=(self.hidden_size,self.streams):
                    raise ValueError('decoder factory returned the wrong hidden/stream geometry')
            layers[spec.layer_id]=decoder
        return MappingProxyType(layers)
