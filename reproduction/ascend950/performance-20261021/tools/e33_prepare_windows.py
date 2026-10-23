#!/usr/bin/env python3
"""Prepare only the four agreed C32 cohorts, reusing immutable E27 inputs."""
from pathlib import Path
import ast
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]

def replace_assignment(text,name,value):
    tree=ast.parse(text)
    node=next(x for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in x.targets))
    lines=text.splitlines();lines[node.lineno-1:node.end_lineno]=[name+' = '+repr(value)]
    return '\n'.join(lines)+'\n'

def render(tag,phase,kind,namespace):
    if kind=='prefix': template=f'E27-{tag}-L32K-C32-HOT-W3'
    elif kind=='hot':template=f'E27-{tag}-L32K-C32-HOT-R2'
    elif kind=='shape':template=f'E27-{tag}-L32K-C32-SHAPE-W1'
    else:template=f'E27-{tag}-L32K-C32-NORMAL-R1'
    rid=f'E33-{tag}-L32K-C32-{phase}'
    text=(ROOT/'configs/e27'/f'{template}.py').read_text()
    text=replace_assignment(text,'run_id',rid)
    tree=ast.parse(text)
    models=next(x for x in ast.walk(tree) if isinstance(x,ast.Call) and any(k.arg=='cache_namespace' for k in x.keywords))
    old=next(k.value for k in models.keywords if k.arg=='cache_namespace')
    lines=text.splitlines();line=lines[old.lineno-1]
    lines[old.lineno-1]=line[:old.col_offset]+repr(namespace)+line[old.end_col_offset:]
    text='\n'.join(lines)+'\n'
    text=text.replace('root+"/runs/e27-ais/"+run_id','root+"/runs/e33-ais/"+run_id')
    ast.parse(text)
    # Dataset and model limits remain those of the accepted source config.
    assert 'batch_size=32' in text
    assert ('max_out_len=1,' if kind=='prefix' else 'max_out_len=128,') in text
    return rid,text

def main():
    destination=ROOT/'configs/e33';destination.mkdir(exist_ok=True)
    entries=[]
    for tag in ['DI','V']:
        for phase,kind,namespace in [('SHAPE-W1','shape',f'e33-{tag.lower()}-shape'),('HOT-PATH-W1','prefix',f'e33-{tag.lower()}-pathwarm'),('HOT-PATH-W2','hot',f'e33-{tag.lower()}-pathwarm')]:
            rid,text=render(tag,phase,kind,namespace);entries.append((rid,text,kind,namespace))
        for mode in ['NORMAL','HOT']:
            for window in ['U1','P','U2']:
                ns=f'e33-{tag.lower()}-{mode.lower()}-{window.lower()}'
                if mode=='HOT':
                    rid,text=render(tag,f'{mode}-{window}-PREFIX','prefix',ns);entries.append((rid,text,'prefix',ns))
                rid,text=render(tag,f'{mode}-{window}','hot' if mode=='HOT' else 'normal',ns)
                entries.append((rid,text,'hot' if mode=='HOT' else 'normal',ns))
    records=[]
    for rid,text,kind,ns in entries:
        p=destination/(rid+'.py')
        if p.exists() and p.read_text()!=text:raise RuntimeError('Refuse to replace an existing changed config: '+str(p))
        p.write_text(text)
        corpus='di-prefix.jsonl' if kind=='prefix' and '-DI-' in rid else 'v-prefix.jsonl' if kind=='prefix' else 'shape-warm.jsonl' if kind=='shape' else 'corpus.jsonl'
        records.append(dict(run_id=rid,config=str(p.relative_to(ROOT)),config_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),kind=kind,namespace=ns,
                            corpus='data/workloads/E27-L32K-C32/'+corpus,requests=64,concurrency=32,output_tokens=1 if kind=='prefix' else 128,cached_tokens=29440 if kind=='hot' else 0,profiled=rid.endswith('-P')))
    workload={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data/workloads/E27-L32K-C32').glob('*.jsonl')}
    d=dict(scope='four cells only; U1/profiled/U2; warmed paths and fresh completed HTTP prefix prewarm per hot window',formal_or_profile_windows=12,preparation_windows=len(records)-12,window_count=len(records),workloads_sha256=workload,records=records)
    (ROOT/'data/E33-window-manifest.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:d[k] for k in ['formal_or_profile_windows','preparation_windows','window_count']}))

if __name__=='__main__':main()
