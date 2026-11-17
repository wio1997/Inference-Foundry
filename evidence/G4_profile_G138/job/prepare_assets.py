"""Prepare one fresh matched32K C16 diagnostic, only inside the remote CPU client."""
from pathlib import Path
import hashlib,json,re
from tokenizers import Tokenizer

ROOT=Path('/task');OUT=ROOT/'client_vllm_g42/profile_g138/assets';OUT.mkdir(parents=True,exist_ok=False)
config=ROOT/'client_configs_vllm_g44/vllm_formal_32768.py';text=config.read_text()
matches=re.findall(r"path=['\"](/task/[^'\"]+\.jsonl)['\"]",text);assert len(matches)==1
source=Path(matches[0]);tokenizer=Tokenizer.from_file('/models/glm53/tokenizer.json')
known=set();sources={}
dataset_paths=list(source.parent.glob('*.jsonl'))+list((ROOT/'client_vllm_g42').glob('profile_g*/assets/profile_32768.jsonl'))
for p in sorted(dataset_paths):
    sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    for line in p.open():
        if line.strip():known.add(tuple(tokenizer.encode(json.loads(line)['question'][:8192],add_special_tokens=False).ids[:128]))
rows=[json.loads(s) for s in source.read_text().splitlines() if s.strip()];assert len(rows)==32
records=[];destination=OUT/'profile_32768.jsonl';prefixes=[]
with destination.open('x') as handle:
    for index,row in enumerate(rows[:16]):
        ids=tokenizer.encode(row['question'],add_special_tokens=False).ids;assert len(ids)==32768
        for offset in range(1024,16384,256):
            rotated=ids[offset:]+ids[:offset];candidate=tokenizer.decode(rotated,skip_special_tokens=False)
            for _ in range(5):
                actual=tokenizer.encode(candidate,add_special_tokens=False).ids
                if len(actual)==32768:break
                candidate=tokenizer.decode((actual+rotated[:128])[:32768],skip_special_tokens=False)
            if len(actual)==32768 and tuple(actual[:128]) not in known:break
        else:raise RuntimeError('No exact new prefix within bounded search')
        known.add(tuple(actual[:128]));prefixes.append(hashlib.sha256(json.dumps(actual[:128]).encode()).hexdigest())
        handle.write(json.dumps({'question':candidate,'answer':'','max_out_len':1024,'request_id':f'g138_profile_{index:02d}','input_tokens':32768},ensure_ascii=False)+'\n')
assert len(set(prefixes))==16
updated=text.replace(str(source),str(destination)).replace("abbr='formal_32768_1024'","abbr='profile_G4_g138_32768_1024'")
assert str(destination) in updated and str(source) not in updated
(OUT/'profile.py').write_text(updated)
manifest={'diagnostic_only':True,'benchmark_score':False,'input_tokens':32768,'output_tokens':1024,'requests':16,'client_concurrency':16,
          'tokenizer_sha256':hashlib.sha256(Path('/models/glm53/tokenizer.json').read_bytes()).hexdigest(),
          'profile_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
          'config_sha256':hashlib.sha256((OUT/'profile.py').read_bytes()).hexdigest(),
          'original_config_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),
          'source_dataset_sha256':sources,'first128_unique_vs_all_source_phases':True,'first128_hashes':prefixes,
          'server_usage_validated':False,'same_bytes_required_for_final_SG_profile':True}
(OUT/'assets.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))
