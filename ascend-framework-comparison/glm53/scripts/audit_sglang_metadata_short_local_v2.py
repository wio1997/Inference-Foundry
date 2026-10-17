"""Audit all-DP short native logical P/D/count results; no performance claim."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 run=ROOT/'runs/20261017T1965-sglang-metadata-short-v2';receipt=json.loads((run/'receipt.json').read_text());assert receipt['rc']==0 and len(receipt['prechecks'])==4 and all(x==0 for x in receipt['prechecks'].values())
 p=run/'node-34.json';outer=json.loads(p.read_text());assert outer['rc']==0
 text=next(l.split('=',1)[1] for l in outer['stdout'].splitlines() if l.startswith('GLM53_SG_MATRIX_ARTIFACTS='));wire=json.loads(text);assert wire['client_rc']==0
 report=wire['result'];assert report['pass'] and len(report['cases'])==16 and len(report['backend_attempts'])==32 and report['successful_inferences']==32
 assert report['actual_successful_output_tokens']==272 and {r['D_DP'] for r in report['cases']}==set(range(16)) and {r['P_DP'] for r in report['cases']}==set(range(4))
 for i,row in enumerate(report['cases']):
  assert row['pass'] and row['P_DP']==i%4 and row['D_DP']==i
  for role,key,outputs in [('P','prefill',1),('D','decode',16)]:
   x=row[key];d=x['response'];assert x['pass'] and x['status']==200 and d['usage']['prompt_tokens']==512 and d['usage']['completion_tokens']==outputs and d['choices'][0]['finish_reason']=='length'
 cap=json.loads((ROOT/'results/sglang-metadata-pool-capacity-v1.json').read_text())
 result={'project_id':'glm53','framework':'sglang','pass':True,'qualification':'ALL16_D_DP_FUNCTIONAL_ONLY_NATIVE_SDMa_PERIOD_AUDIT_FOLLOWS','logical_pairs':16,'backend_attempts':32,'successful_backend_inferences':32,'successful_output_tokens':272,'formal_requests':0,'runtime_CIDs':{n:x['container_id'] for n,x in cap['providers'].items()},'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 p=ROOT/'results/sglang-metadata-short-v1.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
