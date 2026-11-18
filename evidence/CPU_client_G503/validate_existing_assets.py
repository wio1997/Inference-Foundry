"""CPU-only revalidation of immutable G83 dataset bytes; sends no requests."""
from pathlib import Path
import concurrent.futures,datetime,hashlib,json,os,signal
from tokenizers import Tokenizer

ROOT=Path('/task/client_ds41_g503')
TOK=None

def init():
    global TOK
    os.environ['TOKENIZERS_PARALLELISM']='false'
    TOK=Tokenizer.from_file('/models/ds41/tokenizer.json')

def check(entry):
    p=ROOT/'datasets'/Path(entry['path']).name;raw=p.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==entry['sha256']
    rows=[json.loads(line) for line in raw.splitlines() if line.strip()]
    assert len(rows)==entry['requests']
    prefixes=[]
    for row in rows:
        ids=TOK.encode(row['question'],add_special_tokens=False).ids
        assert len(ids)==entry['input_tokens']==row['input_tokens']
        assert row['max_out_len']==1024
        prefixes.append(ids[:128])
    return {'phase':entry['phase'],'length':entry['input_tokens'],'rows':len(rows),'sha256':entry['sha256'],'prefixes':prefixes}

def main():
    os.chdir(ROOT)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Asset revalidation 1800s bound')))
    signal.alarm(1800)
    original=json.loads((ROOT/'original_G83_dataset_manifest.json').read_text())
    assert original['tokenizer_sha256']==hashlib.sha256(Path('/models/ds41/tokenizer.json').read_bytes()).hexdigest()
    assert len(original['profiles'])==24
    (ROOT/'assets_state.json').write_text(json.dumps({'phase':'revalidating','pid':os.getpid(),'at':datetime.datetime.now(datetime.timezone.utc).isoformat()}))
    with concurrent.futures.ProcessPoolExecutor(max_workers=4,initializer=init) as pool:
        results=list(pool.map(check,original['profiles'],chunksize=1))
    all_prefixes=[tuple(p) for e in results for p in e.pop('prefixes')]
    assert len(all_prefixes)==len(set(all_prefixes))==448
    manifest=dict(original)
    manifest['profiles']=[dict(e,path='/task/client_ds41_g503/datasets/'+Path(e['path']).name) for e in original['profiles']]
    manifest['source_original_manifest_sha256']=hashlib.sha256((ROOT/'original_G83_dataset_manifest.json').read_bytes()).hexdigest()
    manifest['revalidated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest['server_usage_validated']=False
    (ROOT/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    audit={'verified':True,'prefix_tokens':128,'prefix_count':448,'formal_first_prefixes_unique':True,'warmup_first_prefixes_unique':True,'no_formal_warmup_prefix_collision':True,'tokenizer_sha256':manifest['tokenizer_sha256'],'datasets':[{'sha256':e['sha256'],'phase':e['phase'],'input_tokens':e['input_tokens']} for e in manifest['profiles']],'no_service_request_submitted':True,'scope':'CPU tokenizer and readonly existing dataset bytes only. No actual server usage or prefix-hit acceptance.'}
    (ROOT/'cold_prefix_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (ROOT/'asset_revalidation.json').write_text(json.dumps({'profiles':results,'passed':True,'prefixes':448,'source_original_manifest_sha256':manifest['source_original_manifest_sha256']},indent=2)+'\n')
    (ROOT/'assets_state.json').write_text(json.dumps({'phase':'complete','pid':os.getpid(),'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'profiles':24,'prefix_count':448,'server_usage_validated':False},indent=2)+'\n')

if __name__=='__main__':main()
