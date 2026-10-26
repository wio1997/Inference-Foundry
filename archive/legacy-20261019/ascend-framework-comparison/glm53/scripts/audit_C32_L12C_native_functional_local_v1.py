"""Independent C32 short cohort/count/all-DP audit; actual artifacts only, no resources."""
import collections,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 run=ROOT/'runs/20261019T-L12C-sglang-native-C32-functional-v1'
 receipt=json.loads((run/'receipt.json').read_text());assert receipt['rc']==0 and len(receipt['prechecks'])==4 and all(v==0 for v in receipt['prechecks'].values())
 path=run/'node-34.json';wire=json.loads(path.read_text());assert wire['rc']==0
 result=json.loads(next(line.split('=',1)[1] for line in wire['stdout'].splitlines() if line.startswith('GLM53_SG_MATRIX_ARTIFACTS=')));assert result['client_rc']==0
 report=result['result'];assert report['pass'] and report['offered_concurrency']==32 and len(report['cases'])==32 and len(report['backend_attempts'])==report['model_requests_attempted']==64
 assert report['successful_inferences']==64 and report['actual_successful_output_tokens']==report['successful_output_tokens']==544 and report['input_tokens']==512 and report['input_sha256_le_u32']=='d6e4cf16d46a7e29461cd4a7e19fa45725cb1b42ac3f76fda226260074db107e'
 identifiers=set();DP=collections.Counter()
 for row in report['cases']:
  assert row['pass'];pr=row['prefill'];dr=row['decode'];match=re.search(r'-p([0-9]{2})-functional\.P$',pr['rid']);assert match
  pid=int(match[1]);assert pid not in identifiers and pid in range(32);identifiers.add(pid)
  assert pr['rid'][:-1]==dr['rid'][:-1] and row['P_DP']==pr['global_dp']==pid%4 and row['D_DP']==dr['global_dp']==pid%16;DP[pid%16]+=1
  for response,outputs in [(pr,1),(dr,16)]:
   assert response['pass'] and response['status']==200 and response['terminal_monotonic_ns']>=response['dispatch_monotonic_ns']
   native=response['response'];assert native['usage']['prompt_tokens']==512 and native['usage']['completion_tokens']==outputs and native['choices'][0]['finish_reason']=='length' and native['choices'][0]['text']
   assert hashlib.sha256(native['choices'][0]['text'].encode()).hexdigest()==response['text_sha256']
 assert identifiers==set(range(32)) and DP==collections.Counter({i:2 for i in range(16)})
 capacity=json.loads((ROOT/'results/sglang-C32-L12C-selected-P-pool-capacity-v1.json').read_text());assert capacity['pass']
 proof={'project_id':'glm53','framework':'sglang','pass':True,'qualification':'C32functional/all16D-DP2each/countonly; two80Ksemantic/nativecacheSDMAgatesstillrequired','logical_pairs':32,'backend_attempts':64,'successful_backend_inferences':64,'successful_output_tokens':544,'formal_requests':0,'runtime_CIDs':{n:item['container_id'] for n,item in capacity['providers'].items()},'D_DP_logical_cases':dict(DP),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 output=ROOT/'results/sglang-C32-L12C-native-functional-v1.json';assert not output.exists();output.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
if __name__=='__main__':main()
