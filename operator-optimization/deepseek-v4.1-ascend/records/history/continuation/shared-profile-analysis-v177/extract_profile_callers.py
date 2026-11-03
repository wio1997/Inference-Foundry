"""CPU-only resolve recorded public operator/caller hierarchy from intact ET prefix."""
import collections
import hashlib
import json
import time
from pathlib import Path

BASE=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis')
OLD=BASE/'joint176-derived';OUT=BASE/'census177';OUT.mkdir(exist_ok=True)
summary=json.loads((OLD/'SUMMARY.json').read_text());trace=Path(summary['trace']['path'])
def write(n,x):
    p=OUT/n;p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');assert p.stat().st_size<64*1024**2
def nodes(path):
    buf=[]
    with path.open() as f:
        for line in f:
            if line.rstrip()=='      {':
                assert not buf;buf=[line]
            elif buf:
                buf.append(line)
                if line.rstrip() in ['      },','      }']:
                    n=json.loads(''.join(buf).rstrip().removesuffix(','));yield n;buf=[]
        assert not buf,'Individual node truncated'

started=time.time();byid={};counts=collections.Counter();schemas={};tensor_arity=collections.defaultdict(collections.Counter)
for n in nodes(trace):
    assert n['id'] not in byid
    byid[n['id']]=(n['name'],n['ctrl_deps'])
    counts[n['name']]+=1
    if not n['name'].startswith('wio176/'):
        attrs={a['name']:a['value'] for a in n.get('attrs',[])}
        if attrs.get('op_schema'):schemas.setdefault(n['name'],attrs['op_schema'])
        tensor_arity[n['name']][(len(n.get('inputs',{}).get('values',[])),len(n.get('outputs',{}).get('values',[])))]+=1
assert len(byid)==summary['parsed_intact_nodes']
groups=collections.Counter();unresolved=[]
for nid,(name,parent) in byid.items():
    if name.startswith('wio176/'):continue
    chain=[];seen={nid};cur=parent;missing=None
    for _ in range(96):
        if cur in seen:missing='cycle:'+str(cur);break
        seen.add(cur)
        if cur not in byid:
            missing='unemitted_root_or_scope:'+str(cur);break
        pname,cur=byid[cur];chain.append(pname)
    else:missing='depth_limit'
    immediate=chain[0] if chain else None
    public=next((x for x in chain if x.startswith(('aten::','npu::','_C_ascend::','vllm::'))),None)
    scope=next((x for x in chain if x.startswith('wio176/scope/')),None)
    groups[(name,immediate,public,scope,missing)]+=1
    if missing and len(unresolved)<20:unresolved.append({'id':nid,'name':name,'reason':missing,'chain':chain})
write('operator-parent-scope-census.json',{
    'status':'EXACT_RECORDED_ET_PARENT_CHAIN_CENSUS',
    'source_trace':summary['trace'],'intact_nodes':len(byid),
    'operator_names':len([n for n in counts if not n.startswith('wio176/')]),
    'groups':[{'operator':k[0],'immediate_parent':k[1],'nearest_public_operator_parent':k[2],
               'nearest_diagnostic_scope':k[3],'missing_ancestor':k[4],'count':v} for k,v in groups.items()],
    'op_schemas':schemas,
    'unresolved_ancestor_examples':unresolved,
    'note':'Control/call nesting only;not device timing or dataflow/alias/retirement proof. Missing top-level root/scope is retained,never repaired.',
    'new_NPU_work':0,'CPU_seconds':time.time()-started})
mods=json.loads((OLD/'ep0-module-identities.json').read_text());classes=collections.defaultdict(list)
for r in mods:
    classes[r['qualified_class']].append({'root':r['root'],'name':r['name'],'python_id':r['python_id'],
        'scalar_attributes':r['scalar_attributes'],'hooks':r['hook_counts'],
        'quant_method':r['quant_method'],'impl_class':r['impl_class']})
write('loaded-module-instance-census.json',{'classes':[{'qualified_class':k,'instances':v} for k,v in classes.items()],
    'source':str(OLD/'ep0-module-identities.json'),'scope':'Two EP0 catalogues from joint176,no new process imports Torch/NPU.'})
print(json.dumps({'status':'PASS','intact_nodes':len(byid),'parent_scope_groups':len(groups),'classes':len(classes),'schemas':len(schemas),'CPU_seconds':time.time()-started}))
