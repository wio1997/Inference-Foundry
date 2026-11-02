# SPDX-License-Identifier: Apache-2.0
"""Device-free prerequisites and bounded synthetic PKI cases for a future probe.

Matching declarations permit a caller's next inspection step; they do not prove
ABI compatibility, exclusive ownership or numerical correctness. This module
does not grant authorization, claim resources, import torch or load libraries.
"""
import hashlib
from pathlib import Path


OPERATORS=('key_pool','pool_key_indexer')
GEOMETRY={'heads':32,'head_dim':128,'pool_size':4,'topk_tokens':2048}


def _int(value,name,minimum=0):
    if type(value) is not int or value<minimum:raise ValueError(f'invalid {name}')
    return value


def _sha(path):
    digest=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def require_owner_authorization(authorized,assignment,identity):
    if authorized is not True:raise ValueError('later explicit user NPU authorization required')
    if set(identity)!= {'task_id','executor','generation'} or any(not isinstance(identity[k],str) or not identity[k] for k in ('task_id','executor')):
        raise ValueError('explicit owner identity required')
    _int(identity['generation'],'generation',1)
    if any(assignment.get(k)!=v for k,v in identity.items()):raise ValueError('current target owner differs')
    if assignment.get('mode')=='preparation_only' or assignment.get('allow_NPU_tests') is not True:
        raise ValueError('current control forbids NPU execution')


def require_probe_prerequisites(*,authorized,assignment,identity,profile,stage,architecture,version,module_origin):
    """Check current task declarations before any backend import.

    The external caller supplies a freshly read owner record and runtime
    metadata resolver. Authorization remains a human decision, not a profile
    property. There is intentionally no built-in supposedly compatible tuple.
    Binary pins cover declared files only, not the complete transitive ABI.
    """
    require_owner_authorization(authorized,assignment,identity)
    if stage not in ('inspect-abi','pki'):raise ValueError('unknown probe stage')
    device=_int(profile.get('device'),'device')
    resource=profile.get('device_resource');claims=assignment.get('NPU_resources_claimed')
    if not isinstance(resource,str) or not resource or not isinstance(claims,list) or resource not in claims:
        raise ValueError('explicit existing device resource claim required')
    if profile.get('architecture')!='aarch64' or architecture!='aarch64':raise ValueError('aarch64 profile required')
    if profile.get('device_type')!='Ascend910C':raise ValueError('explicit Ascend910C profile required')
    packages=profile.get('packages')
    if not isinstance(packages,dict) or not {'torch','torch-npu'}<=packages.keys() or len(packages)<3:
        raise ValueError('exact torch/backend/operator distribution pins required')
    operator_distribution=profile.get('operator_distribution')
    if operator_distribution not in packages or operator_distribution in ('torch','torch-npu'):
        raise ValueError('explicit operator distribution pin required')
    for name,wanted in packages.items():
        if not isinstance(name,str) or not isinstance(wanted,str) or not wanted or version(name)!=wanted:
            raise ValueError(f'package pin differs: {name}')
    files=profile.get('files')
    if not isinstance(files,dict) or not files or len(files)>32:raise ValueError('bounded explicit binary/source file pins required')
    origin=profile.get('module_origin')
    if origin!=module_origin or origin not in files:raise ValueError('operator module origin must be pinned')
    if not any('.so' in Path(p).name for p in files):raise ValueError('native binary file pin required')
    observed={}
    for path,wanted in files.items():
        if not isinstance(path,str) or not Path(path).is_absolute() or not isinstance(wanted,str) or len(wanted)!=64:
            raise ValueError('absolute paths and SHA256 file pins required')
        actual=_sha(path)
        if actual!=wanted:raise ValueError(f'file pin differs: {Path(path).name}')
        observed[path]=actual
    schemas=profile.get('schemas',{})
    if stage=='pki':
        if not isinstance(profile.get('device_name'),str) or not profile['device_name']:raise ValueError('verified device-name declaration required')
        if not isinstance(schemas,dict) or set(schemas)!=set(OPERATORS):raise ValueError('both inspected operator schemas required')
        for name,schema in schemas.items():
            if not isinstance(schema,str) or not schema.startswith(f'cann_ops_transformer::{name}('):raise ValueError('invalid schema declaration')
    return {'stage':stage,'device':device,'device_resource':resource,'packages':dict(packages),
            'files':observed,'prerequisite_declarations_match':True,'ABI_validated':False}


def require_export_schemas(actual,expected):
    """Exact default-overload schema comparison before device allocation."""
    if set(actual)!=set(OPERATORS) or set(expected)!=set(OPERATORS) or actual!=expected:
        raise ValueError('exported operator schemas differ from inspected profile')


def pki_probe_cases():
    """Small T, real GLM geometry, including strict ranking above topk.

    At most one live query per request, at its final position. No CP, speculative
    tokens or graph replay. Pool keys encode ordinal in two BF16-exact channels:
    (1+ordinal//128, (ordinal%128)/128); positive unit queries/head weights make
    scores strictly increasing, avoiding ties even with513 completed pools.
    """
    return (
        {'name':'short_zero_pool','tokens':1,'ends':[1],'lengths':[0],'tails':[3],'positions':[2]},
        {'name':'mixed_zero_query_padding_ranking','tokens':5,'ends':[1,1,2,3],
         'lengths':[0,1,34,513],'tails':[3,0,0,0],'positions':[2,135,2051,-999,-999]},
        {'name':'fully_padded','tokens':3,'ends':[0,0],'lengths':[0,0],'tails':[0,0],'positions':[-999]*3},
        {'name':'zero_tokens','tokens':0,'ends':[],'lengths':[],'tails':[],'positions':[]},
    )


def plan_pki_case(case):
    """Pure integer ownership/table plan; reversed IDs catch slot/namespace bugs."""
    tokens=_int(case['tokens'],'tokens');ends=case['ends'];lengths=case['lengths'];tails=case['tails'];positions=case['positions']
    if tokens>8 or not len(ends)==len(lengths)==len(tails) or len(ends)>8 or len(positions)!=tokens:
        raise ValueError('bounded matching case dimensions required')
    previous=0;used=[]
    for end,count,tail in zip(ends,lengths,tails,strict=True):
        _int(end,'end');_int(count,'compressed length');_int(tail,'tail')
        if not previous<=end<=tokens or end-previous>1 or count>513 or tail>=4:raise ValueError('unsupported case metadata')
        if end>previous:
            if positions[previous]!=count*4+tail-1 or positions[previous]<0:raise ValueError('query position must end at declared history')
        previous=end;used.append(max(1,(count+31)//32))
    blocks=max(1,sum(used));columns=max(used,default=1);next_id=blocks-1;table=[];pool_rows=[]
    for count,number in zip(lengths,used,strict=True):
        ids=list(range(next_id,next_id-number,-1));next_id-=number
        table.append(ids+[-1]*(columns-number))
        pool_rows.append([(ids[j//32],j%32,j) for j in range(count)])
    # Dummy work addresses physical block0. Every physical row is initialized.
    return {'blocks':blocks,'columns':columns,'table':table,'pool_rows':pool_rows,
            'cache_bytes':blocks*32*128*2,'live_tokens':previous}


def expected_pki_tokens(case):
    """Score-order-independent token SET oracle, including raw causal tail."""
    plan_pki_case(case);result=[[] for _ in range(case['tokens'])];previous=0
    for end,count,tail in zip(case['ends'],case['lengths'],case['tails'],strict=True):
        if end>previous:
            first=max(0,count-512)*4
            result[previous]=list(range(first,count*4+tail))
        previous=end
    return result
