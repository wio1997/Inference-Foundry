"""Exclude one concrete BOOL-clear hypothesis without inventing a tensor identity."""
import ast,hashlib
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,read,write

def main():
    assert not (O/'BOOL-prefix-candidate-review-v67.json').exists()
    pins=read(O/'embedding-initialization-operator-contract-v67.json')['source_files']
    model=next(p for p in pins if p['path'].endswith('models/deepseek_v41/model.py'))
    seq=next(p for p in pins if p['path'].endswith('models/common/ops/sequence_parallel.py'))
    for p in [model,seq]:assert hashlib.sha256(Path(p['path']).read_bytes()).hexdigest()==p['sha256']
    tree=ast.parse(Path(model['path']).read_text())
    c=next(x for x in tree.body if isinstance(x,ast.ClassDef)and x.name=='DeepseekV41SharedAttentionState')
    f=next(x for x in c.body if isinstance(x,ast.FunctionDef)and x.name=='reset')
    assert len(f.body)==1 and isinstance(f.body[0],ast.Return)and isinstance(f.body[0].value,ast.Constant)and f.body[0].value.value is None
    write(O/'BOOL-prefix-candidate-review-v67.json',dict(source_files=[model,seq],
        excluded_candidate='shared_attention_state.reset()betweenembeddingandTilecannotclearBOOLbuffer:ASTbodyisonlyreturnNone.',
        compatible_candidate='sp_padding_mask(None,hidden)createsBOOLnew_zeros(N),butcompatibleplacement/shapealoneisnotoperand/fieldproof.',
        retained_unknown='ExactgraphBOOL24bankcouldnotbeboundbyaddresses/argvalues/graphIR;noassignmentrewrite. Engram/forward-context/captureinputguardsneedconcreteproducer-consumerproofifpursued.',
        pending_regimes=2,pending_exec=80,global_ambiguity_unchanged=3136,
        no_remote_or_framework_audit=True,overall_complete=False))
    q=O/'embedding-initialization-operator-contract-v67.json';d=read(q)
    d['evidence_refs'].append('BOOL-prefix-candidate-review-v67.json')
    d['BOOL_clear_candidate_review']='Excludedshared_state.reset(no-op);sp_padding_maskNonecompatiblebutunprovedfield. Preserve2regimes80pending.'
    write(q,d)
    print(dict(shared_state_reset_hypothesis_excluded=True,BOOL_prefix_pending=[2,80],ambiguity=3136))

if __name__=='__main__':main()
