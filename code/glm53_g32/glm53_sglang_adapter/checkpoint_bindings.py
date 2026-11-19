"""Candidate GLM NextN checkpoint-owned input/output binding; native loaders only.

This exact candidate requires BOTH own matrices in the selected draft layer.
No absent-weight fallback, inferred alias, or guard suppression is implemented.
"""
import hashlib,json,os
from pathlib import Path

IDENTITY_SHA256='db653fdc513b9e21fd47c969d1028f7c3b35e3bf3d030725e3e53eef76d02787'
INDEX_SHA256='d08dc4b74f24ab912a7c7fd66c3c4172fb752b7044377e8e48e9c5ff089ff9ad'

def validate_model_contract(model):
    from sglang.srt.runtime_context import get_model
    from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
    root=Path(__file__).resolve().parents[1]
    identity=root/'nextn_checkpoint_identity.json'
    if hashlib.sha256(identity.read_bytes()).hexdigest()!=IDENTITY_SHA256:raise RuntimeError('NextN checkpoint identity metadata SHA mismatch')
    receipt=json.loads((root/'nextn_checkpoint_receipt.json').read_text())
    if not (receipt['identity_sha256']==IDENTITY_SHA256 and receipt['index_sha256']==INDEX_SHA256 and receipt['tensor_bytes_sha_verified'] is True):
        raise RuntimeError('NextN installer tensor identity proof missing')
    model_root=Path(get_model().model_path)
    index=model_root/'quant_model_weights.safetensors.index.json'
    if hashlib.sha256(index.read_bytes()).hexdigest()!=INDEX_SHA256:raise RuntimeError('Actual NextN checkpoint index SHA changed')
    cfg=model.config
    if not (cfg.num_hidden_layers==45 and cfg.num_nextn_predict_layers==1 and cfg.vocab_size==154880 and cfg.hidden_size==4096
            and getattr(cfg,'tie_word_embeddings',False) is False):raise RuntimeError('NextN checkpoint geometry/tie contract mismatch')
    for name,module in [('embed',model.model.embed_tokens),('head',model.lm_head)]:
        if type(module.quant_method) is not UnquantizedEmbeddingMethod or str(module.weight.dtype)!='torch.bfloat16':
            raise RuntimeError('Checkpoint-owned BF16 '+name+' requires actual native unquantized embedding method')
    model._glm53_checkpoint_contract={'identity_sha256':IDENTITY_SHA256,'index_sha256':INDEX_SHA256,'tensor_bytes_sha_verified_at_install':True,'own_embedding_head_unquantized':True}
    return model._glm53_checkpoint_contract

KEYS={'embed_tokens.weight':'model.embed_tokens.weight','shared_head.head.weight':'lm_head.weight'}

def route_checkpoint_owned(model,weights,layer_prefixes):
    params=dict(model.named_parameters());owned={};model._glm53_checkpoint_owned_bindings=owned
    for source,tensor in weights:
        prefix=next((p for p in layer_prefixes if source.startswith(p)),None)
        if prefix is None:continue
        suffix=source[len(prefix):]
        destination=KEYS.get(suffix)
        if destination is None:
            yield source,tensor
            continue
        if destination in owned:raise RuntimeError('Duplicate draft own binding '+destination)
        param=params[destination]
        if list(tensor.shape)!=[model.config.vocab_size,model.config.hidden_size]:
            raise RuntimeError('Draft own matrix checkpoint shape mismatch '+source)
        if str(tensor.dtype) not in ('torch.bfloat16','torch.float16','torch.float32'):
            raise RuntimeError('Draft own matrix must be floating unquantized '+source)
        # Preserve per-parameter TP/vocab sharding and native interception by the outer ledger.
        loader=getattr(param,'weight_loader',None)
        if not callable(loader):raise RuntimeError('Native own matrix loader missing '+destination)
        outcome=loader(param,tensor)
        if outcome is False:raise RuntimeError('Native own matrix loader rejected '+source)
        owned[destination]={'source_key':source,'parameter':param,'source_shape':list(tensor.shape),'source_dtype':str(tensor.dtype)}
    if set(owned)!=set(KEYS.values()):
        raise RuntimeError('This checkpoint-own candidate requires both draft embedding/head; missing '+str(set(KEYS.values())-set(owned)))

def keep_checkpoint_owned(model,target_embed,target_head):
    owned=getattr(model,'_glm53_checkpoint_owned_bindings',{})
    if set(owned)!=set(KEYS.values()):raise RuntimeError('Draft checkpoint ownership incomplete before native share hook')
    actual={'model.embed_tokens.weight':model.model.embed_tokens.weight,'lm_head.weight':model.lm_head.weight}
    targets={'model.embed_tokens.weight':target_embed,'lm_head.weight':target_head}
    for name,entry in owned.items():
        if actual[name] is not entry['parameter']:raise RuntimeError('Loaded draft own Parameter was replaced '+name)
        if actual[name] is targets[name]:raise RuntimeError('Unexpected target alias before own binding hook '+name)
    path=Path(model._glm53_weight_ledger_path);report=json.loads(path.read_text())
    if report['exception'] or report['missing_parameters'] or report['unexpected_undispatched']:
        raise RuntimeError('Strict ledger failed before draft own binding proof')
    evidence={}
    for name,entry in owned.items():
        if name not in report['loaded_parameters']:raise RuntimeError('Own matrix missing native dispatch coverage '+name)
        source=report['source_records'].get(entry['source_key'],{})
        if source.get('classification')!='dispatched' or not source.get('dispatches'):
            raise RuntimeError('Own source has no actual native loader dispatch '+name)
        evidence[name]={'ownership':'checkpoint_owned','source_key':entry['source_key'],'source_shape':entry['source_shape'],'source_dtype':entry['source_dtype'],'parameter_identity_preserved':True,'target_alias':False}
    # Existing fields remain truthful: own matrices are not target_shares.
    if report['deferred_target_shared_parameters']:raise RuntimeError('Unexpected unresolved deferred parameters')
    report['draft_bindings']=evidence;report['status']='dispatch_coverage_complete'
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');os.replace(tmp,path)
    summary=path.with_name(path.name.replace('.coverage.json','.summary.json'))
    if summary.exists():
        data=json.loads(summary.read_text());data['draft_bindings']=evidence;data['status']=report['status'];tmp=summary.with_suffix(summary.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');os.replace(tmp,summary)

def require_checkpoint_owned_bindings(model,report):
    evidence=report.get('draft_bindings',{});owned=getattr(model,'_glm53_checkpoint_owned_bindings',{})
    expected=set(KEYS.values())
    if set(evidence)!=expected or set(owned)!=expected:raise RuntimeError('Draft own binding proof incomplete')
    actual={'model.embed_tokens.weight':model.model.embed_tokens.weight,'lm_head.weight':model.lm_head.weight}
    for name in expected:
        e=evidence[name];source=report['source_records'].get(e.get('source_key'),{})
        if not (e.get('ownership')=='checkpoint_owned' and e.get('parameter_identity_preserved') is True and e.get('target_alias') is False
                and name in report['loaded_parameters'] and source.get('classification')=='dispatched' and source.get('dispatches')
                and actual[name] is owned[name]['parameter']):
            raise RuntimeError('Draft own binding evidence failed '+name)
