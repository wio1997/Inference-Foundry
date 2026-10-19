"""Local versioned candidate; actual SDK allocation still required after official scan."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 evidence=ROOT/'results/vllm-all-owned-empty-memory-scan-v1.json';assert sha(evidence)=='03978f827b460a42a80074f5d1470d62f65950e06d86def8d017bcf3f231c252';assert json.loads(evidence.read_text())['pass']
 old=ROOT/'config/vllm-correctness-v9';new=ROOT/'config/vllm-correctness-v10';assert not new.exists();new.mkdir()
 for p in old.iterdir():
  if p.name=='manifest.json':continue
  text=p.read_text().replace('160GB','192GB')
  if p.suffix=='.json':
   c=json.loads(text);c['phase']='SHARED_POOL_AFTER_OFFICIAL_EMPTY_SCAN_NATIVE_RUNTIME_PENDING';write(new/p.name,c)
  else:(new/p.name).write_text(text)
 manifest={'project_id':'glm53','framework':'vllm','phase':'PREPARED_NOT_STAGED_OR_STARTED','parent_files_sha256':{str(p.relative_to(ROOT)):sha(p) for p in old.iterdir() if p.is_file()},'files_sha256':{p.name:sha(p) for p in new.iterdir()},'native_Meta_policy':{'lease_ttl_ms':60000,'evict_high':99,'evict_low':98},'external_pool_bytes':568*1024**3,'provider_GiB':{30:192,31:192,43:128,45:56},'common_max_dram_GiB':192,'official_empty_scan_evidence_sha256':sha(evidence),'memory_policy':'Same568GiB externalpool as SG; placementchanged afterD43/D45 native1GiBscan155/91GiB belowunexecuted160each. SDKactualcapacitypending. SGstaging268.397GiBseparate; totaldatausage notidentical; common1TiBdata ceiling.'}
 write(new/'manifest.json',manifest)
 provider=ROOT/'config/standalone-pool-providers-v4';provider.mkdir(exist_ok=False)
 supervisor=ROOT/'config/vllm-shared-container-v2';supervisor.mkdir(exist_ok=False)
 for n,dram in {30:192,31:192,43:128,45:56}.items():
  cfg=json.loads((ROOT/f'config/standalone-pool-providers-v3/vllm-node-{n}.json').read_text());cfg.update(dram_size=f'{dram}GB',max_dram_size='192GB');write(provider/f'vllm-node-{n}.json',cfg)
  cfg=json.loads((ROOT/f'config/vllm-shared-container-v1/node-{n}.json').read_text());cfg['run_id']='20261017T1754-vllm-shared-pool-v2';cfg['provider_argv'][-1]=sha(provider/f'vllm-node-{n}.json');write(supervisor/f'node-{n}.json',cfg)
 p=ROOT/'scripts/sdk_vllm_pool568_preflight_v3.py';assert not p.exists();p.write_text((ROOT/'scripts/sdk_vllm_pool568_preflight_v2.py').read_text().replace("sizes={30:'136GB',31:'112GB',43:'160GB',45:'160GB'}","sizes={30:'192GB',31:'192GB',43:'128GB',45:'56GB'}").replace('160GB','192GB').replace('and native CLI parse only','only; native CLI requires NPU'))
 p=ROOT/'scripts/vllm_overlay_preflight_payload_v7.py';assert not p.exists();p.write_text((ROOT/'scripts/vllm_overlay_preflight_payload_v6.py').read_text().replace('160GB','192GB'))
 m=json.loads((ROOT/'config/standalone-budget-stage-v8.json').read_text());b=m['bundles'][0];b['remote']='/data/wio/inference-foundry/glm53/overlays/vllm-sdma-v9-20261017'
 for name,item in b['files'].items():
  loc=item['local'].replace('config/vllm-correctness-v9/','config/vllm-correctness-v10/').replace('config/standalone-pool-providers-v3/','config/standalone-pool-providers-v4/').replace('config/vllm-shared-container-v1/','config/vllm-shared-container-v2/').replace('sdk_vllm_pool568_preflight_v2.py','sdk_vllm_pool568_preflight_v3.py').replace('vllm_overlay_preflight_payload_v6.py','vllm_overlay_preflight_payload_v7.py');item.update(local=loc,sha256=sha(ROOT/loc))
 m['official_scan_evidence_sha256']=sha(evidence);write(ROOT/'config/standalone-budget-stage-v9.json',m)
 print(json.dumps({'stage_manifest_sha256':sha(ROOT/'config/standalone-budget-stage-v9.json'),'native_config_manifest_sha256':sha(new/'manifest.json'),'provider_GiB':manifest['provider_GiB'],'resource_actions':0}))
if __name__=='__main__':main()
