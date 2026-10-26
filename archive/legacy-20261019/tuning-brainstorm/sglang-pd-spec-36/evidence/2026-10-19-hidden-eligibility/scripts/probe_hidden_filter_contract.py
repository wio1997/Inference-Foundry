#!/usr/bin/env python3
"""Real source filter -> resolve -> NextN input, explicit gather/norm boundaries."""
from __future__ import annotations
import ast,contextlib,copy,hashlib,json,pathlib,types
from unittest.mock import patch
import torch
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L029-R004-filter-hidden';O.mkdir(parents=True,exist_ok=False)
protocol=json.loads((R/'specs/L029-filter-hidden.json').read_text());g={'torch':torch,'ExitStack':contextlib.ExitStack,'_is_npu':True,'_is_cuda':False,'_DEBUG_ASSERT':False};bodies=[];cases=[];checks=0
for name,h in protocol['source_hashes'].items():assert hashlib.sha256((R/'sources/installed'/name).read_bytes()).hexdigest()==h
def check(x,label):
    global checks
    checks+=1;assert x,label
def load(name,owner,method,target):
    p=R/'sources/installed'/name;text=p.read_text();cls=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name==owner);node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==method);m=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),copy.deepcopy(node)],type_ignores=[]);ast.fix_missing_locations(m);exec(compile(m,str(p),'exec'),g);g[target]=g[method];bodies.append({'path':name,'owner':owner,'method':method,'line':node.lineno,'end_line':node.end_lineno,'body_sha256':hashlib.sha256(ast.get_source_segment(text,node).encode()).hexdigest()})
load('python/sglang/srt/speculative/eagle_info.py','EagleDraftInput','filter_batch','filter')
load('python/sglang/srt/managers/overlap_utils.py','FutureMap','_resolve_spec_extras','resolve')
load('python/sglang/srt/models/deepseek_nextn.py','DeepseekModelNextN','forward','nextn')
g['gather_spec_extras']=lambda idx,p,index,bonus,hidden:(p[idx],index[idx],bonus[idx],None if hidden is None else hidden[idx])
g['BumpAllocator']=lambda **kw:types.SimpleNamespace();g['ReplicatedLinear']=type('ReplicatedLinearBoundary',(),{})
class ProjectionBoundary(Exception):pass
with patch.object(torch,'get_device_module',return_value=types.SimpleNamespace(current_stream=lambda:object())),patch.object(torch.Tensor,'record_stream',lambda *a,**kw:None):
    for keep,relay_hidden in [([2,0],True),([2,0],False),([0,2],True),([0,2],False)]:
        raw=torch.arange(3*6144,dtype=torch.float32).view(3,6144)+1
        di=types.SimpleNamespace(future_indices=torch.tensor([1,2,3]),hidden_states=raw,draft_probs=None,future_dsa_topk_indices_available=False)
        g['filter'](di,torch.tensor(keep),keep)
        expected_slots=torch.tensor([1,2,3])[keep]
        check(torch.equal(di.future_indices,expected_slots),'actual future-index filter keeps keys')
        check(di.hidden_states is raw and di.hidden_states.shape[0]==3,'actual future-mode filter delays hidden materialization')
        pool=torch.zeros((8,6144));pool[[1,2,3]]=raw
        f=types.SimpleNamespace(spec_algo=types.SimpleNamespace(is_ngram=lambda:False),device='cpu',need_topk=True,
            need_hidden_states=relay_hidden,topk_p_buf=torch.ones((8,1)),topk_index_buf=torch.ones((8,1),dtype=torch.long),
            output_tokens_buf=torch.ones(8,dtype=torch.long),hidden_states_buf=pool if relay_hidden else None,
            draft_probs_buf=None,dsa_topk_indices_buf=None)
        fb=types.SimpleNamespace(spec_info=di)
        g['resolve'](f,fb)
        check(di.topk_p.shape[0]==2,'resolve topk rows match filtered keys')
        check(di.hidden_states.shape[0]==(2 if relay_hidden else 3),'hidden row alignment requires overwrite')
        if relay_hidden:check(torch.equal(di.hidden_states,raw[keep]),'baseline hidden rows match reordered requests')
        projection=[]
        def stop(x):projection.append(tuple(x.shape));raise ProjectionBoundary()
        model=types.SimpleNamespace(quant_config=object(),rot_weight=None,enorm=torch.nn.RMSNorm(6144,elementwise_affine=False),hnorm=torch.nn.RMSNorm(6144,elementwise_affine=False),eh_proj=stop)
        try:g['nextn'](model,torch.tensor([1,2]),torch.tensor([0,0]),fb,input_embeds=torch.ones((2,6144)))
        except ProjectionBoundary:check(relay_hidden and projection==[(2,12288)],'baseline reaches projection boundary with matching rows')
        except RuntimeError:check(not relay_hidden and not projection,'gate-off raw hidden fails source row concatenation before projection')
        else:raise AssertionError('expected projection boundary or shape rejection')
        cases.append({'keep_indices':keep,'hidden_relay_enabled':relay_hidden,'filtered_rows':2,'hidden_rows':di.hidden_states.shape[0],
            'projection_boundary_reached':bool(projection),'shape_rejection':not relay_hidden,'native_gather_and_model_not_executed':True})
result={'status':'PASS','semantic_checks':checks,'case_count':len(cases),'cases':cases,'source_bodies':bodies,'source_commit':protocol['source_commit'],'source_hashes':protocol['source_hashes'],
    'torch':torch.__version__,'scope':'Complete actual source methods; future filter selects survivor fixture, gather is explicit CPU oracle, norms are CPU RMSNorm, projection stops.',
    'full_scheduler_cancel_executed':False,'native_executed':False,'candidate_accepted':False,'accepted_baseline':None,'performance':'NOT_MEASURED'}
(O/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','semantic_checks','case_count','native_executed']}))
