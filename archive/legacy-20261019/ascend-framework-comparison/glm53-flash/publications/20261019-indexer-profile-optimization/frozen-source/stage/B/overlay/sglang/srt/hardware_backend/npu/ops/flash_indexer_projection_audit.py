# SPDX-License-Identifier: Apache-2.0
"""Bounded rank0 CPU64 actual-model KPool projection/compression oracle.

No values are persisted. Native model/operator arguments and outputs are unchanged.
CPU copies/float64 matmuls are diagnostics and exclude performance qualification.
"""
import pathlib,hashlib,json,os
import torch
import torch.nn.functional as F
_SEEN=set()
_ROOT=pathlib.Path('/run-evidence')


def before(owner,hidden,k_hidden,q_c,projected,k,gate,q_values,weights,cache,tail,metadata,tail_metadata):
    if not (_ROOT/'audit-enabled').exists() or torch.npu.current_device()!=0:return None
    positions=metadata.positions[:k.shape[0]].detach().cpu().long()
    n=len(positions)
    if n==0:return None
    start=int(positions[0]);phase=('fresh' if n>=128 else 'short_fresh') if start==0 else 'decode' if n==1 and start>=2048 else 'long' if n>1 and start>=2048 else None
    if phase is None:return None
    name=owner.k_cache.prefix;key=(name,phase)
    if key in _SEEN:return None
    assert positions.tolist()==list(range(start,start+n))
    assert metadata.query_metadata is None and owner.index_kpool==4
    assert metadata.cum_query_lens.numel()==1,'Bounded oracle requires serial single request'
    rows=sorted(set([0,n//3,2*n//3,n-1]));idx=torch.tensor(rows,dtype=torch.long,device=hidden.device)
    h=k_hidden.index_select(0,idx).detach().cpu().double();w=owner._wk_weight_f32.detach().cpu().double()
    proj=F.linear(h,w);actual=projected.index_select(0,idx).detach().cpu().double()
    torch.testing.assert_close(actual,proj,atol=1e-4,rtol=1e-4)
    norm_weight=owner._norm_weight_f32.detach().cpu().double() if owner._norm_weight_f32 is not None else None
    norm_bias=owner._norm_bias_f32.detach().cpu().double() if owner._norm_bias_f32 is not None else None
    epsilon=getattr(owner.k_norm,'eps',getattr(owner.k_norm,'variance_epsilon',1e-6))
    norm=F.layer_norm(proj[:,:owner.head_dim],(owner.head_dim,),norm_weight,norm_bias,epsilon)
    actual_norm=k.index_select(0,idx).detach().cpu().double();torch.testing.assert_close(actual_norm,norm,atol=1e-4,rtol=1e-4)
    gate_ref=F.linear(h,owner._gate_weight_f32.detach().cpu().double());gate_actual=gate.index_select(0,idx).detach().cpu().double()
    torch.testing.assert_close(gate_actual,gate_ref,atol=1e-4,rtol=1e-4)
    assert not isinstance(q_c,tuple) and q_values is not None
    qw=owner.wq_b.weight.detach().cpu().double();qin=q_c.shape[-1];qout=owner.n_head*owner.head_dim
    assert owner.wq_b.weight.dtype in (torch.bfloat16,torch.float32)
    orientation='out_in'
    if qw.shape!=(qout,qin):
        assert qw.shape==(qin,qout);qw=qw.T;orientation='native_transposed_in_out'
    assert getattr(owner.wq_b,'bias',None) is None
    qref=F.linear(q_c.index_select(0,idx).detach().cpu().double(),qw).reshape(len(rows),owner.n_head,owner.head_dim)
    qa=q_values.index_select(0,idx).detach().cpu().double();torch.testing.assert_close(qa,qref,atol=.02,rtol=.02)
    # The native head-weight branch uses hidden, while K/gates can have a
    # separately materialized k_hidden buffer even with CP disabled.
    wh=hidden.index_select(0,idx).detach().cpu().double()
    wr=F.linear(wh,w[owner.head_dim:]).to(q_values.dtype)*(owner.softmax_scale*owner.n_head**-.5)
    wa=weights.index_select(0,idx).detach().cpu();torch.testing.assert_close(wa.double(),wr.double(),atol=.02,rtol=.02)
    tail_blocks=tail_metadata.block_table.detach().cpu().long();assert tail_blocks.shape[0]==1
    block=int(tail_blocks[0,0]);assert tail.shape[1:]==(2,4,128) and 0<=block<tail.shape[0]
    old_tail=tail[block].detach().cpu().clone();ks=k.detach().cpu().double();gs=gate.detach().cpu().double();ape=owner.index_kpool_compress_ape.detach().cpu().double()
    slots=metadata.slot_mapping[:n].detach().cpu().long();tail_slots=tail_metadata.slot_mapping[:n].detach().cpu().long();expected=[]
    for t,pos in enumerate(positions.tolist()):
        assert int(tail_slots[t])==block*4+pos%4
        if (pos+1)%4:continue
        slot=int(slots[t]);assert 0<=slot<cache.shape[0]*cache.shape[1]
        kval=[];gval=[]
        for p in range(pos-3,pos+1):
            if p>=start:kval.append(ks[p-start]);gval.append(gs[p-start])
            else:assert p>=0;kval.append(old_tail[0,p%4].double());gval.append(old_tail[1,p%4].double())
        coeff=torch.softmax(torch.stack(gval)+ape,dim=0);compressed=(torch.stack(kval)*coeff).sum(0)
        expected.append((slot,compressed))
    tail_ref=old_tail.clone()
    for t in range(max(0,n-4),n):tail_ref[0,int(positions[t])%4].copy_(ks[t].float());tail_ref[1,int(positions[t])%4].copy_(gs[t].float())
    _SEEN.add(key)
    return {'name':name,'phase':phase,'positions':(start,start+n-1),'rows':rows,'q_weight_orientation':orientation,'block':block,'tail_ref':tail_ref,'expected':expected,'projection_errors':{'WK_FP32':float((actual-proj).abs().max()),'norm_FP32':float((actual_norm-norm).abs().max()),'gate_FP32':float((gate_actual-gate_ref).abs().max()),'query_BF16':float((qa-qref).abs().max()),'head_weights_BF16':float((wa.double()-wr.double()).abs().max())}}


def after(check,cache,tail):
    if check is None:return
    max_error=0.
    for slot,reference in check['expected']:
        block,offset=divmod(slot,cache.shape[1]);actual=cache[block,offset].detach().cpu().reshape(-1).double()
        torch.testing.assert_close(actual,reference,atol=.02,rtol=.02);max_error=max(max_error,float((actual-reference).abs().max()))
    actual_tail=tail[check['block']].detach().cpu();assert torch.equal(actual_tail,check['tail_ref'])
    assert hashlib.sha256(actual_tail.numpy().tobytes()).hexdigest()==hashlib.sha256(check['tail_ref'].numpy().tobytes()).hexdigest()
    record={k:v for k,v in check.items() if k not in ['expected','tail_ref']};record.update(kind='indexer_projection_compression_CPU64',device_id=0,pid=os.getpid(),compressed_rows=len(check['expected']),compressed_BF16_max_abs=max_error,FP32_raw_key_gate_tail_exact=True,passed=True,source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
    print('FLASH_INDEXER_PROJECTION_CHECK '+json.dumps(record),flush=True)
