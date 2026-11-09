"""Verify the accepted materialized snapshot;no model ordevice dependencies."""
from pathlib import Path
import hashlib,json
def main():
 root=Path(__file__).resolve().parent
 provenance=json.loads((root/'SOURCE.json').read_text())
 manifest=root/provenance['manifest']
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 assert sha(manifest)==provenance['accepted_source_manifest_sha256'],'source manifest identity changed'
 entries=json.loads(manifest.read_text());source=root/provenance['source_directory']
 for name,expected in entries.items():
  path=source/name
  assert path.is_relative_to(source) and sha(path)==expected,name
 files=json.loads((root/'publication-manifest.json').read_text())
 for name,expected in files.items():assert sha(root/name)==expected,name
 joint=json.loads((root/'validation/joint-result.json').read_text())
 assert joint['state']=='JOINT_INVARIANT_PASS' and len(joint['comparisons'])==8 and all(c['equal'] for c in joint['comparisons'])
 assert joint['public_API_control']['state']=='PUBLIC_API_EOS_AND_EXISTING_NPU_BASELINE_PASS'
 print(json.dumps(dict(state='PUBLICATION_SNAPSHOT_PASS',accepted_source=sha(manifest),source_files=len(entries)+1,publication_files=len(files)+1,model_requests=0,model_loads=0)))
if __name__=='__main__':main()
