"""Decode only collected MC2 communication-group records using installed schema."""
import collections,hashlib,json,re,struct
from pathlib import Path
ROOT=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis')
fmt=struct.Struct('<HHIIIQQ56I');assert fmt.size==256
def one(profile):
 rows=[];pins=[];kfc=[]
 for p in profile.rglob('*'):
  if not p.is_file() or p.name.endswith('.done'):continue
  if 'kfc' in p.name.lower():kfc.append({'path':str(p),'bytes':p.stat().st_size})
  if 'mc2_comm_info.slice_' not in p.name:continue
  b=p.read_bytes();assert len(b)%fmt.size==0
  pins.append({'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
  for at in range(0,len(b),fmt.size):
   x=fmt.unpack_from(b,at);assert x[0]==0x5a5a and x[11]<=8
   rows.append({'record_offset':at,'magic':x[0],'level':x[1],'struct_type':x[2],'thread_id':x[3],
    'data_len':x[4],'raw_timestamp':x[5],'recorded_group_key':str(x[6]),'rank_size':x[7],
    'rank_id':x[8],'usr_rank_id':x[9],'aicpu_kfc_stream_id':x[10],'comm_stream_ids':list(x[12:12+x[11]])})
 return {'profile':profile.name,'raw_files':pins,'MC2_group_records':rows,'KFC_raw_files':kfc}
if __name__=='__main__':
 out={'cohorts':{},'group_key_semantics':'Recorded native group identifier within each cohort;not a globally unique incarnation or a per-operation sequence ID.',
      'schema':'Installed9.2.0 StructFmt.MC2_COMM_INFO_FMT=<HHIIIQQ56I,256B;Mc2CommInfoBean field positions.',
      'limits':['Membership/streams do not pair per-layer/per-iteration exchanges.','No receiver counts or tensor contents in this record.','No cross-cohort key/alias joins.'],'overall_complete':False}
 for name in ['builtin-v164-official-db','npu166-official-db']:
  profiles=sorted((ROOT/name/'profiles').glob('*_ascend_pt'));assert len(profiles)==32
  rows=[one(p) for p in profiles];groups={}
  for p in rows:
   for r in p['MC2_group_records']:
    key=(r['recorded_group_key'],r['rank_size']);g=groups.setdefault(key,{'members':set(),'profiles':set(),'raw_records':0})
    g['members'].add((r['rank_id'],r['usr_rank_id']));g['profiles'].add(p['profile']);g['raw_records']+=1
  group_rows=[{'recorded_group_key':k[0],'rank_size':k[1],'members':[list(x) for x in sorted(v['members'])],
     'profiles':len(v['profiles']),'raw_records':v['raw_records'],'complete_rank_id_set':{x[0] for x in v['members']}==set(range(k[1]))} for k,v in groups.items()]
  out['cohorts'][name]={'profiles':rows,'group_membership_sets':group_rows,
     'raw_MC2_records':sum(len(r['MC2_group_records']) for r in rows),'KFC_raw_files':sum(len(r['KFC_raw_files']) for r in rows)}
 p=ROOT/'npu166-official-db/MC2-GROUP-RECORDS-v167.json';p.write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({name:{'raw_records':d['raw_MC2_records'],'KFC_files':d['KFC_raw_files'],'groups':d['group_membership_sets']} for name,d in out['cohorts'].items()}))
