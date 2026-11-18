"""Scope the existing native query-row chunk budget to the validated NPU path."""
from pathlib import Path
import datetime,hashlib,inspect,json,os,types

BUDGET=128<<20

def install(root):
    import sglang.srt.layers.attention.dsv4.v41_indexer.scoring as scoring
    import sglang.srt.layers.attention.dsv4.dsv41_sparse as sparse
    import sglang.kernels.ops.quantization.fp4_fake_quant as quant
    proof=json.loads((root/'actual_FP4_score_gate_prerequisite.json').read_text())
    assert proof['actual_native_FP4_operand_domain_8ranks64cases_strict_scores_and_topk'] and proof['all8_released']
    for rel,mod in [('native/dsv4/v41_indexer/scoring.py',scoring),('native/dsv4/dsv41_sparse.py',sparse),('native/fp4_fake_quant.py',quant)]:
        assert hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()==proof['source_sha256'][rel],rel
    config=Path('/models/ds41/config.json');assert hashlib.sha256(config.read_bytes()).hexdigest()==proof['config_sha256']
    cfg=json.loads(config.read_text());text=cfg.get('text_config',cfg);assert cfg['model_type']=='deepseek_v41' and text['index_n_heads']==32 and text['index_head_dim']==128
    assert sparse.fake_quant_fp4 is quant.fake_quant_fp4
    if getattr(scoring,'_npu_query_budget_g587',None):
        assert scoring._npu_query_budget_g587['budget_bytes']==BUDGET
        return
    prior=scoring._score_chunks
    assert scoring._TORCH_SCORE_BUDGET_BYTES==1<<30 and '_TORCH_SCORE_BUDGET_BYTES' in prior.__code__.co_names and not prior.__code__.co_freevars
    namespace=dict(prior.__globals__);namespace['_TORCH_SCORE_BUDGET_BYTES']=BUDGET
    npu=types.FunctionType(prior.__code__,namespace,name=prior.__name__,argdefs=prior.__defaults__,closure=prior.__closure__);npu.__kwdefaults__=prior.__kwdefaults__
    assert npu.__code__ is prior.__code__ and npu.__globals__['ChunkScores'] is prior.__globals__['ChunkScores']
    recorded=False
    def scoped_chunks(**kwargs):
        nonlocal recorded
        q=kwargs['q'];indexer=kwargs['indexer'];key=kwargs['index_k'];weights=kwargs['weights']
        if q.device.type!='npu':
            yield from prior(**kwargs)
            return
        import torch
        assert isinstance(indexer,sparse.DeepseekV41Indexer)
        assert indexer.n_local_heads==32 and indexer.index_head_dim==128 and q.shape[1:]==(32,128)
        assert q.dtype==key.dtype==torch.bfloat16 and weights.dtype in (torch.bfloat16,torch.float32)
        assert sparse.fake_quant_fp4 is quant.fake_quant_fp4,'Actual native FP4 query/key producer changed'
        for chunk in npu(**kwargs):
            if not recorded and q.shape[0]>=8191:
                folder=root/'native_score_budget_hits';folder.mkdir(exist_ok=True)
                data={'pid':os.getpid(),'device_index':q.device.index,'query_shape':list(q.shape),'key_shape':list(key.shape),'weight_dtype':str(weights.dtype),'chunk_score_shape':list(chunk.scores.shape),'chunk_rows':chunk.scores.shape[0],'budget_bytes':BUDGET,'original_native_module_global_budget_bytes':scoring._TORCH_SCORE_BUDGET_BYTES,'native_code_object_preserved':npu.__code__ is prior.__code__,'native_score_source_sha256':proof['source_sha256']['native/dsv4/v41_indexer/scoring.py'],'scope':'Actual native first large-row score chunk returned; no correctness, request-completion or performance acceptance','at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'installer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
                (folder/f'worker{os.getpid()}.json').write_text(json.dumps(data,indent=2)+'\n');recorded=True
            yield chunk
    scoring._score_chunks=scoped_chunks
    scoring._npu_query_budget_g587={'budget_bytes':BUDGET,'CPU_original_budget_preserved':scoring._TORCH_SCORE_BUDGET_BYTES==1<<30,'native_code_object_preserved':npu.__code__ is prior.__code__,'operand_domain':'fixed native indexer per32 UE8M0 FP4 queries and FP4 68byte index-K readback','scope':'P eager native score memory bound only; full model/PD/performance pending'}
    (root/f'score_budget_install.pid{os.getpid()}.json').write_text(json.dumps(scoring._npu_query_budget_g587,indent=2)+'\n')
