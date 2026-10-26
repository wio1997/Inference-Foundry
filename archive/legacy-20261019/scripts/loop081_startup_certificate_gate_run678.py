"""CPU same-state gate using the actual original/patched refresh method AST.

Run offline via the mechanical executor. Does not install or modify serving code.
Inference tensors have no version counter: immutable serialized ownership remains
an explicit prerequisite, not a mutation-detection claim made by this gate.
"""
from __future__ import annotations
import ast, hashlib, json, socket, sys
from pathlib import Path
from types import SimpleNamespace as NS
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from runtime.startup_slot_certificate import StartupSlotCertificate, _version
import loop081_startup_certificate_patch_run678 as patch

def offline():
    with socket.socket() as s:
        s.settimeout(1)
        if s.connect_ex(('127.0.0.1',8080))==0:
            raise RuntimeError('refuse CPU gate against a ready serving process')

def methods(source):
    tree=ast.parse(source)
    names={'_refresh_draft_context_slots','_flush_slot_refresh_audit'}
    found={n.name:n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in names}
    assert set(found)==names
    mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*found.values()],type_ignores=[])
    ns={'torch':torch};exec(compile(ast.fix_missing_locations(mod),'<actual-refresh-methods>','exec'),ns)
    return ns,found

def bindings(p,bs,*,inference=False):
    rows=[]
    with torch.inference_mode(inference):
        for gid,b in enumerate(bs,2):
            cols=max((x+63)//b for x in p)+4
            table=torch.arange(12*cols,dtype=torch.int64).reshape(12,cols).clone()+1
            mapping=torch.full((96,),-123,dtype=torch.int64)
            rows.append((gid,table,mapping,b))
    return rows

def cert(b,p,**kw):
    args=dict(bindings=b,positions=p,remaining=[1024]*12,cycle_index=0,schedule_mode='off',batch_size=12,width=8)
    args.update(kw);return StartupSlotCertificate.build(**args)

def owner(b,c=None,refresh=True,defer=False):
    return NS(_refresh_draft_slots=refresh,config=NS(batch_size=12),_group_slot_bindings=b,
              _startup_slot_certificate=c,_defer_slot_audit=defer,_pending_slot_refresh_audit=[],slot_refresh_audit=[])

def state(p,cycle):
    return NS(cycle_index=cycle,target_positions=(torch.tensor(p)[:,None]+torch.arange(8)[None,:]).flatten())

def raises(fn,needle=None):
    try:fn()
    except (RuntimeError,IndexError) as e:
        if needle is not None:assert needle in str(e),(needle,str(e))
        return type(e).__name__
    raise AssertionError('expected original guard failure')

def main():
    offline()
    path=ROOT/'evidence/20260929_loop081_bound/run677/candidate/handoff.py'
    raw=path.read_text();edited=patch.handoff(raw)
    manifest=json.loads((ROOT/'evidence/20260929_loop081_bound/run677/candidate_manifest.json').read_text())
    assert hashlib.sha256(raw.encode()).hexdigest()==manifest['sources']['handoff']['candidate_sha256']
    assert (ROOT/'evidence/20260929_loop081_bound/run678/candidate/handoff.py').read_text()==edited
    assert (ROOT/'bootstrap/vllm_dspark_handoff.py').read_text()==edited
    old,oa=methods(raw);new,na=methods(edited)
    oldfn=old['_refresh_draft_context_slots'];newfn=new['_refresh_draft_context_slots']
    # Prove only the conditional predicate changed for the two original guards.
    oldguards=[n for n in ast.walk(oa['_refresh_draft_context_slots']) if isinstance(n,ast.If) and any(isinstance(x,ast.Raise) for x in n.body)]
    newguards=[n for n in ast.walk(na['_refresh_draft_context_slots']) if isinstance(n,ast.If) and any(isinstance(x,ast.Raise) for x in n.body)]
    for label in ('outside reserved range','negative physical block'):
        a=next(n for n in oldguards if label in ast.unparse(n));b=next(n for n in newguards if label in ast.unparse(n))
        assert ast.dump(ast.Module(body=a.body,type_ignores=[]))==ast.dump(ast.Module(body=b.body,type_ignores=[]))
        assert isinstance(a.test,ast.BoolOp) and isinstance(b.test,ast.BoolOp)
        assert ast.dump(a.test.values[1])==ast.dump(b.test.values[1])
        assert ast.unparse(b.test.values[0])=='guard_required'
    scenarios=0
    for bs in ((2,8),(8,32),(1,16)):
      for p in ([0]*12,[32768+i%9 for i in range(12)],[17+i*3 for i in range(12)]):
       for pattern in ('zero','eight','varied'):
        for defer in (False,True):
          a=bindings(p,bs);b=[(g,t.clone(),m.clone(),s) for g,t,m,s in a]
          c=cert(b,p);assert c.record['eligible'],c.record
          x=owner(a,defer=defer);y=owner(b,c,defer=defer);pos=list(p)
          for cycle in range(8):
            st=state(pos,cycle);oldfn(x,st);newfn(y,st)
            assert all(torch.equal(u[2],v[2]) for u,v in zip(a,b))
            delta=[0 if pattern=='zero' else 8 if pattern=='eight' else (cycle+i)%9 for i in range(12)]
            pos=[v+d for v,d in zip(pos,delta)]
          old['_flush_slot_refresh_audit'](x);new['_flush_slot_refresh_audit'](y)
          assert x.slot_refresh_audit==y.slot_refresh_audit and c.record['used_cycles']==8
          scenarios+=1
    p=[64]*12
    # Negative reachable but not yet visited: conservative rejection, unchanged fallback.
    b=bindings(p,(2,8));b[0][1][0,(p[0]+63)//2]=-1
    c=cert(b,p);assert not c.record['eligible'] and c.record['reason']=='negative_reachable_block'
    x=owner([(g,t.clone(),m.clone(),s) for g,t,m,s in b]);y=owner(b,c)
    oldfn(x,state(p,0));newfn(y,state(p,0));assert all(torch.equal(u[2],v[2]) for u,v in zip(x._group_slot_bindings,b))
    # Negative cells outside each row's reachable interval must not over-reject.
    pwide=[64+i*8 for i in range(12)];b=bindings(pwide,(2,8));b[0][1][0,(pwide[-1]+63)//2]=-1
    assert cert(b,pwide).record['eligible']
    for mutation,needle in [('negative','negative physical block'),('outside','outside reserved range')]:
        b=bindings(p,(2,8))
        if mutation=='negative':b[0][1][0,p[0]//2]=-1
        c=cert(b,p,remaining=[64]*12);assert not c.record['eligible']
        st=state(p if mutation=='negative' else [-1]+p[1:],0)
        for fn,obj in ((oldfn,owner(b)),(newfn,owner(b,c))):raises(lambda:fn(obj,st),needle)
    # Range and eligibility gates.
    b=bindings(p,(2,8));tiny=[(g,t[:,:2].contiguous(),m,s) for g,t,m,s in b]
    assert not cert(tiny,p).record['eligible']
    for kwargs in (dict(remaining=[64]*12),dict(cycle_index=1),dict(schedule_mode='next'),dict(batch_size=11),dict(width=7)):
        c=cert(b,p,**kwargs);assert not c.record['eligible'] and not c.permits(0,b) and c.record['used_cycles']==0
    c=cert(b,p);assert c.permits(0,b) and not c.permits(0,b) and c.record['invalidated']
    c=cert(b,p);assert not c.permits(1,b) and c.record['invalidated']
    c=cert(b,p);assert not c.permits(8,b) and c.record['used_cycles']==0
    c=cert(b,p);y=owner(b,c,refresh=False);newfn(y,state(p,0));assert c.record['used_cycles']==0
    # Replacement and table mutation invalidate; fallback retains actual guards.
    for replace in ('table','mapping'):
        b=bindings(p,(2,8));c=cert(b,p);g,t,m,s=b[0]
        b[0]=(g,t.clone() if replace=='table' else t,m.clone() if replace=='mapping' else m,s)
        assert not c.permits(0,b) and c.record['invalidated']
    b=bindings(p,(2,8));c=cert(b,p);b[0][1][0,p[0]//2]=-1
    raises(lambda:newfn(owner(b,c),state(p,0)),'negative physical block');assert c.record['invalidated']
    b=bindings(p,(2,8));g,t,m,s=b[0];b[0]=(g,t,t.flatten()[:96],s)
    assert cert(b,p).record['reason']=='table_mapping_alias'
    spread=[64+i*128 for i in range(12)]
    assert cert(bindings(spread,(2,8)),spread).record['reason']=='certificate_copy_span_too_large'
    # Dtype overflow cannot invalidate the mathematical startup range proof.
    for dtype in (torch.int32,torch.int64):
        limit=torch.iinfo(dtype).max
        assert cert(bindings([0]*12,(2,8)),[limit-63]*12,computed_dtype=dtype).record['reason']=='computed_position_overflow'
    assert cert(bindings(p,(2,8)),p,computed_dtype=torch.float32).record['reason']=='computed_position_overflow'
    # Cross-group alias must be rejected even when each own table/mapping is disjoint.
    b=bindings(p,(2,8));g,t,m,s=b[1];b[1]=(g,t,b[0][1].flatten()[:96],s)
    assert not cert(b,p).record['eligible']
    # In-place mapping metadata change retains id/data_ptr but changes stride.
    b=bindings(p,(2,8));c=cert(b,p);b[0][2].as_strided_((96,),(0,))
    assert not c.permits(0,b) and c.record['invalidated']
    # Inference tensors have no version counter; legitimate immutable use still works.
    b=bindings(p,(2,8),inference=True);assert _version(b[0][1]) is None
    c=cert(b,p);assert c.record['eligible'] and all(not g['version_tracked'] for g in c.record['groups'])
    with torch.inference_mode():
        newfn(owner(b,c),state(p,0))
    assert c.record['used_cycles']==1
    # Original policy only guards cycles <8; do not accidentally broaden it.
    b=bindings(p,(2,8));b[0][1][0,p[0]//2]=-1
    a=[(g,t.clone(),m.clone(),s) for g,t,m,s in b]
    oldfn(owner(a),state(p,8));newfn(owner(b),state(p,8));assert all(torch.equal(u[2],v[2]) for u,v in zip(a,b))
    result=dict(pass_gate=True,cpu_only=True,same_state_scenarios=scenarios,method_source_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                gates=['actual method AST','conditional original guard bodies/predicates retained','first8 slots and audit parity','0..8 advancement','reachable negative fallback','unvisited per-row cells','bounds fallback','early parking','cycle sequence and duplicate','refresh-disabled no skip','binding replacement','table version mutation','own/cross-group alias rejection','mapping stride mutation','computed dtype overflow rejection','copy span bound','inference tensor immutable use','cycle8 original policy'],
                limitations=['No NPU performance claim','Inference-tensor mutation is not detectable by version; serialized immutable ownership is required','No arbitrary state-advance or parking contract claim'])
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
