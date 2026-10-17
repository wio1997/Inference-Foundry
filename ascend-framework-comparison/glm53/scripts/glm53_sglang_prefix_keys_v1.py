"""Pure CPU reference for the native B103 namespaced, page-chained uint32 hash."""
import hashlib,struct

def namespace_seed(extra_key=None,cache_salt=None):
    cache_salt=cache_salt or None
    if extra_key is None and cache_salt is None:return None
    h=hashlib.sha256(b'sglang-cache-namespace-v1')
    for s in [extra_key,cache_salt]:
        if s is None:h.update(b'\x00')
        else:
            b=s.encode('utf-8');h.update(b'\x01'+len(b).to_bytes(8,'little')+b)
    return h.digest()

def page_hashes(ids,cache_salt,extra_key=None,is_bigram=True,page_size=128,complete_only=True):
    assert type(page_size) is int and page_size>0
    assert isinstance(ids,list) and all(type(x) is int and 0<=x<2**32 for x in ids)
    logical=max(len(ids)-1,0) if is_bigram else len(ids)
    end=logical//page_size*page_size if complete_only else logical
    prior=namespace_seed(extra_key,cache_salt);result=[]
    for start in range(0,end,page_size):
        stop=min(start+page_size,end)
        words=[v for i in range(start,stop) for v in (ids[i],ids[i+1])] if is_bigram else ids[start:stop]
        raw=struct.pack('<'+'I'*len(words),*words)
        prior=hashlib.sha256((prior or b'')+raw).digest();result.append(prior.hex())
    return result

def component_keys(hashes,tag='glm53-sglang-w8a8-v1'):
    return [f'{tag}_{h}__{kind}' for h in hashes for kind in ['k','v','index_k']]
