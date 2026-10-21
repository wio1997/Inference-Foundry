import pathlib,urllib.request,json,tarfile,io,concurrent.futures,subprocess
ROOT=pathlib.Path(__file__).resolve().parent
PINS={
 'DeepInfer':('deepseek-ai/DeepInfer','774ce244b87cc0eb8d4b94ab9f5de2ab22049f01'),
 'deepseek-recipe':('deepseek-ai/deepseek-recipe','935817c66f5d6e64838f419692545f06787e485f'),
 'DeepInfer/vendor/DeepEP-Ascend/third-party/deep_jit':('deepseek-ai/DeepJIT','65f513952b8771408a19202f65e36398522db6db'),
 'DeepInfer/vendor/DeepGEMM-Ascend/third-party/deep-jit':('deepseek-ai/DeepJIT','65f513952b8771408a19202f65e36398522db6db'),
 'DeepInfer/vendor/DeepGEMM-Ascend/third-party/magic_enum':('Neargye/magic_enum','b233b96e49d371bad00300f59b5ba581100b8745'),
 'DeepInfer/vendor/DeepSelect':('deepseek-ai/DeepSelect','bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18'),
 'DeepInfer/vendor/FlashMLA':('deepseek-ai/FlashMLA','2e5429fc5653bab6e081f09477126f731882a6a9'),
 'DeepInfer/vendor/DeepEP-Ascend':('deepseek-ai/DeepEP-Ascend','3b25377d04b24fc6154698ded78a2bcb2c59afff'),
 'DeepInfer/vendor/DeepGEMM-Ascend':('deepseek-ai/DeepGEMM-Ascend','3a35326271b069b8b5bf1a754435acd6cc91575b'),
}
def fetch(item):
 path,(repo,sha)=item; dest=ROOT/'src'/path;dest.mkdir(parents=True,exist_ok=True)
 data=subprocess.check_output(['curl','-4','-fLsS','--connect-timeout','10','--max-time','120','--retry','2',f'https://codeload.github.com/{repo}/tar.gz/{sha}'])
 with tarfile.open(fileobj=io.BytesIO(data)) as t:
  for m in t.getmembers():
   parts=pathlib.PurePosixPath(m.name).parts[1:]
   if not parts or '..' in parts:continue
   p=dest.joinpath(*parts)
   if m.isdir():p.mkdir(parents=True,exist_ok=True)
   elif m.isfile():p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(t.extractfile(m).read());p.chmod(m.mode&0o777)
 print(path,sha,len(data),flush=True)
 if (dest/'.gitmodules').exists():print('NESTED',path,(dest/'.gitmodules').read_text(),flush=True)
 return {'path':path,'repo':repo,'sha':sha,'archive_bytes':len(data)}
# Main trees first, then vendor trees, so parent archive cannot overwrite submodules.
records=[]
for group in [[x for x in PINS.items() if x[0].count('/')==i] for i in [0,2,4]]:
 with concurrent.futures.ThreadPoolExecutor(4) as pool:records.extend(pool.map(fetch,group))
(ROOT/'source-pins.json').write_text(json.dumps(records,indent=2))
