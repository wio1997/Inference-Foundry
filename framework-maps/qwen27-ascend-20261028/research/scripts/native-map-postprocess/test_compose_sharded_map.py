import gzip, hashlib, importlib.util, json, tempfile, unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('compose_shards',HERE/'compose_sharded_map.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def dump(p,x):
    b=(json.dumps(x,separators=(',',':'),ensure_ascii=False)+'\n').encode();p.write_bytes(b);return b
def gz(p,obj):
    b=(json.dumps(obj,separators=(',',':'),ensure_ascii=False)+'\n').encode()
    with p.open('wb') as f:
        with gzip.GzipFile(fileobj=f,mode='wb',filename='',mtime=0) as z:z.write(b)
    c=p.read_bytes();return {'path':p.name,'json_bytes':len(b),'json_sha256':hashlib.sha256(b).hexdigest(),'gzip_bytes':len(c),'gzip_sha256':hashlib.sha256(c).hexdigest()}
def mkdetail(row,status):
    return {'schema_version':1,'identity':{k:row[k] for k in ('framework','label','rank_basename','rank_key')},'native_rank_audit':{'lineage_summary':{'status':status,'reason':'kept'}},'sections':{s:[] for s in m.SECTIONS}}

class ComposeTests(unittest.TestCase):
    def setup_case(self,root):
        root=Path(root);src=root/'source';src.mkdir();shards=src
        rows=[{'framework':'sglang','label':'waveA','rank_basename':f'rank-{i}','rank_key':f'key-{i}'} for i in range(2)]
        manifest={'schema_version':1,'status':'completed','expected_rank_count':2,'rows':rows};mb=dump(src/'manifest.json',manifest)
        per=[];joined={s:0 for s in m.SECTIONS}
        for i,row in enumerate(rows):
            ls='no_matching_graph_receipts' if i==1 else 'available';fn=f'detail-{i}.json.gz';rec=gz(shards/fn,mkdetail(row,ls))
            per.append({**row,'rank':i,'lineage_status':ls,'lineage_reason':'kept','row_counts':{s:(None if i==1 else 0) for s in m.SECTIONS},'detail':rec})
        unat={s:[] for s in m.SECTIONS};ur=gz(shards/'unattributed-rows.json.gz',{'schema_version':1,'kind':'unattributed_native_rows','sections':unat})
        sections={s:{'source_row_count':0,'rank_joined_row_count':0,'unattributed_row_count':0} for s in m.SECTIONS}
        index={'schema_version':1,'status':'sharded_native_evidence_pending_root_acceptance','identity_count':2,'bundle_errors':0,'inputs':{'native_bundle':{'path':'bundle.json','bytes':123,'sha256':'a'*64},'completion_manifest':{'path':'manifest.json','bytes':len(mb),'sha256':hashlib.sha256(mb).hexdigest()}},'sections':sections,'per_rank':per,'unattributed_shard':ur,'limits':{'max_rank_detail_json_bytes':m.MAX_RANK,'max_unattributed_json_bytes':m.MAX_UNATTR}}
        ip=src/'index.json';dump(ip,index)
        combined={'metadata':{'status':'PREVIEW_NOT_FINAL','validity':'existing unresolved quality','preview_not_final':True,'quality_scope':'unresolved'},'new_measurements':{'profile_rows':{'type':'profile_rows','rows':[{'canary':'keep'}]},'ablation_pairs':[{'formal':'keep'}],'cpu_lifecycle_map':{'status':'ready','wave_count':41,'waves':[{'label':'kept'}],'native_bundle':{'status':'pending_input_not_supplied','note':'pending'}},'supplement':{'schema_version':1,'layer_rows':[{'canary':'layer'}],'native_rank_audits':[{'canary':'audit'}]}}}
        cp=src/'combined-map.json';dump(cp,combined)
        return cp,ip,src/'manifest.json',shards,combined
    def test_compose_preserves_canary_and_formal_arrays_and_unknown(self):
        with tempfile.TemporaryDirectory() as t:
            cp,ip,mp,shards,src=self.setup_case(t);out=Path(t)/'out'
            r=m.build(cp,ip,mp,out,expected=2);new=json.loads((out/'combined-map.json').read_text())
            self.assertEqual(new['new_measurements']['supplement'],src['new_measurements']['supplement'])
            self.assertEqual(new['new_measurements']['ablation_pairs'],src['new_measurements']['ablation_pairs'])
            self.assertEqual(new['new_measurements']['profile_rows'],src['new_measurements']['profile_rows'])
            self.assertEqual(new['new_measurements']['cpu_lifecycle_map']['waves'],src['new_measurements']['cpu_lifecycle_map']['waves'])
            self.assertEqual(new['metadata']['status'],'REVIEWABLE_NATIVE_EVIDENCE_PENDING_ROOT_ACCEPTANCE')
            self.assertTrue(new['metadata']['preview_not_final'])
            self.assertEqual(new['new_measurements']['native_sharded_map']['detail_base_href'],'native-rank-details/')
            self.assertIsNone(new['new_measurements']['native_sharded_map']['index']['per_rank'][1]['row_counts']['layer_rows'])
            self.assertEqual(new['new_measurements']['native_sharded_map']['index']['per_rank'][1]['lineage_status'],'no_matching_graph_receipts')
            self.assertEqual(r['verified_identity_count'],2)
            self.assertTrue((out/'native-rank-details'/'detail-0.json.gz').exists())
    def test_duplicate_identity_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as t:
            cp,ip,mp,_,_=self.setup_case(t);ix=json.loads(ip.read_text());ix['per_rank'][1]['rank_key']=ix['per_rank'][0]['rank_key'];ip.write_text(json.dumps(ix))
            out=Path(t)/'badout'
            with self.assertRaises(m.ComposeError):m.build(cp,ip,mp,out,expected=2)
            self.assertFalse(out.exists())
    def test_corrupt_shard_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            cp,ip,mp,shards,_=self.setup_case(t);p=shards/'detail-0.json.gz';p.write_bytes(p.read_bytes()+b'bad')
            out=Path(t)/'badout'
            with self.assertRaises(m.ComposeError):m.build(cp,ip,mp,out,expected=2)
            self.assertFalse(out.exists())
    def test_unattributed_shard_must_be_valid_and_conserved(self):
        with tempfile.TemporaryDirectory() as t:
            cp,ip,mp,shards,_=self.setup_case(t);p=shards/'unattributed-rows.json.gz';p.write_bytes(p.read_bytes()[:-5])
            out=Path(t)/'badout'
            with self.assertRaises(m.ComposeError):m.build(cp,ip,mp,out,expected=2)
            self.assertFalse(out.exists())

if __name__=='__main__':unittest.main()
