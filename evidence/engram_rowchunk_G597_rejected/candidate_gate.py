import torch
from reference_gate import rotated_gate as reference_gate

CHUNK_ROWS=256

def rotated_gate(x,kv,q_weight,k_weight,eps,clamp_value,rotation_block,input_ids=None,image_token_id=None):
    """Bound only independent token rows; every per-row arithmetic stays native."""
    rows=x.shape[0]
    if rows<=CHUNK_ROWS:
        return reference_gate(x,kv,q_weight,k_weight,eps,clamp_value,rotation_block,input_ids,image_token_id)
    out=torch.empty_like(x)
    for start in range(0,rows,CHUNK_ROWS):
        stop=min(start+CHUNK_ROWS,rows)
        ids=input_ids[start:stop] if input_ids is not None else None
        out[start:stop].copy_(reference_gate(x[start:stop],kv[start:stop],q_weight,k_weight,eps,clamp_value,rotation_block,ids,image_token_id))
    return out
