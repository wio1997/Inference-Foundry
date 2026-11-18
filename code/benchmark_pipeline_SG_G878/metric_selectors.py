"""Select one reviewed Prometheus series, keeping TP replicas and modes separate."""
import json,math,re

def select_counter(text,name,labels):
    assert labels and all(isinstance(k,str) and isinstance(v,str) for k,v in labels.items())
    pattern=re.compile(r'^'+re.escape(name)+r'(?:\{([^\n]*)\})?\s+([0-9.eE+\-]+)(?:\s+\d+)?$',re.M)
    values=[]
    for match in pattern.finditer(text):
        parsed={k:json.loads('"'+v+'"') for k,v in re.findall(r'(\w+)="((?:\\.|[^"\\])*)"',match[1] or '')}
        if not all(parsed.get(k)==v for k,v in labels.items()):continue
        value=float(match[2]);assert math.isfinite(value);values.append(value)
    assert len(values)==1,('Expected exactly one reviewed metric series',name,labels,len(values))
    return values[0]
