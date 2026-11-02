# SPDX-License-Identifier: Apache-2.0
"""Device-free guards and scalar ranking fixtures; no native library execution."""
import copy,hashlib,importlib.abc,json,os,runpy,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from deepinfer.hugging_face.glm5_next_cann_probe import require_probe_prerequisites,require_export_schemas,pki_probe_cases,plan_pki_case,expected_pki_tokens


class NativeImportTrap(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','torch_npu','cann_ops_transformer','vllm','triton'}:
            raise AssertionError('FORBIDDEN_NATIVE_IMPORT:'+fullname)


class CANNProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        # Byte-only fixture: this is never dlopened and is not an ABI profile.
        origin=self.root/'__init__.py';binary=self.root/'fixture.so';origin.write_bytes(b'fixture source');binary.write_bytes(b'not a library')
        self.identity={'task_id':'fixture-task','executor':'fixture-executor','generation':2}
        self.owner={**self.identity,'mode':'fixture_future_native','allow_NPU_tests':True,'NPU_resources_claimed':['fixture-host/card0']}
        self.schemas={n:f'cann_ops_transformer::{n}(Tensor x) -> Tensor' for n in ('key_pool','pool_key_indexer')}
        self.profile={'architecture':'aarch64','device_type':'Ascend910C','device_name':'fixture910C','device':0,'device_resource':'fixture-host/card0','operator_distribution':'fixture-operator',
            'packages':{'torch':'fixture-torch','torch-npu':'fixture-backend','fixture-operator':'fixture-version'},
            'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (origin,binary)},'module_origin':str(origin),'schemas':self.schemas}
        self.args=dict(authorized=True,assignment=self.owner,identity=self.identity,profile=self.profile,stage='pki',architecture='aarch64',
            version=lambda n:self.profile['packages'][n],module_origin=str(origin))

    def test_matching_fixture_is_only_a_declaration(self):
        result=require_probe_prerequisites(**self.args)
        self.assertTrue(result['prerequisite_declarations_match']);self.assertFalse(result['ABI_validated'])

    def test_revoked_or_wrong_owner_rejects_before_metadata(self):
        for kind in ('unauthorized','preparation','forbidden','executor','generation','claim'):
            args=self.args.copy();owner=self.owner.copy();args['assignment']=owner
            if kind=='unauthorized':args['authorized']=False
            elif kind=='preparation':owner['mode']='preparation_only'
            elif kind=='forbidden':owner['allow_NPU_tests']=False
            elif kind=='executor':owner['executor']='other'
            elif kind=='generation':owner['generation']=3
            else:owner['NPU_resources_claimed']=[]
            args['version']=lambda n:(_ for _ in ()).throw(AssertionError('metadata reached'))
            with self.subTest(kind=kind),self.assertRaises(ValueError):require_probe_prerequisites(**args)

    def test_package_architecture_device_origin_and_binary_drift(self):
        for kind in ('version','architecture','device','origin','binary','no_binary','packages','schema','stage'):
            args=self.args.copy();profile=copy.deepcopy(self.profile);args['profile']=profile
            if kind=='version':args['version']=lambda n:'different'
            elif kind=='architecture':args['architecture']='x86_64'
            elif kind=='device':profile['device_type']='Ascend910B'
            elif kind=='origin':args['module_origin']='other'
            elif kind=='binary':profile['files'][str(self.root/'fixture.so')]='0'*64
            elif kind=='no_binary':profile['files'].pop(str(self.root/'fixture.so'))
            elif kind=='packages':profile['packages']={'torch':'fixture-torch'}
            elif kind=='schema':profile['schemas'].pop('key_pool')
            else:args['stage']='unknown'
            with self.subTest(kind=kind),self.assertRaises(ValueError):require_probe_prerequisites(**args)

    def test_inspection_needs_pins_but_not_an_existing_schema(self):
        args=self.args.copy();args['stage']='inspect-abi';args['profile']={k:v for k,v in self.profile.items() if k!='schemas'}
        self.assertEqual(require_probe_prerequisites(**args)['stage'],'inspect-abi')

    def test_schema_alias_return_and_extra_overload_drift_reject(self):
        require_export_schemas(self.schemas,self.schemas.copy())
        for kind in ('missing','return','mutation','extra'):
            actual=self.schemas.copy()
            if kind=='missing':actual.pop('key_pool')
            elif kind=='return':actual['key_pool']=actual['key_pool'].replace('-> Tensor','-> Tensor[]')
            elif kind=='mutation':actual['key_pool']=actual['key_pool'].replace('Tensor x','Tensor(a!) x')
            else:actual['other']='schema'
            with self.assertRaises(ValueError):require_export_schemas(actual,self.schemas)

    def test_scalar_oracle_strict_score_and_reversed_block_namespace(self):
        cases=pki_probe_cases();mixed=cases[1];plan=plan_pki_case(mixed);expected=expected_pki_tokens(mixed)
        self.assertEqual(expected[0],[0,1,2]);self.assertEqual(expected[1],list(range(136)))
        self.assertEqual(expected[2],list(range(4,2052)));self.assertEqual(expected[3:],[[],[]])
        self.assertEqual(plan['blocks'],21);self.assertLessEqual(plan['cache_bytes'],172032)
        used=[block*32+row for rows in plan['pool_rows'] for block,row,j in rows];self.assertEqual(len(used),len(set(used)))
        scores=[1+j//128+(j%128)/128 for j in range(513)]
        self.assertTrue(all(a<b for a,b in zip(scores,scores[1:])));self.assertEqual(plan['table'][-1][0],16)

    def test_bad_case_positions_counts_or_multiple_queries_reject(self):
        for field,value in [('tokens',99),('lengths',[514]),('positions',[3]),('ends',[2])]:
            case=copy.deepcopy(pki_probe_cases()[0]);case[field]=value
            with self.assertRaises(ValueError):plan_pki_case(case)

    def test_cli_unauthorized_and_current_policy_never_import_native(self):
        script=Path(__file__).with_name('probe_glm5_next_cann_indexer_npu.py')
        owner=self.root/'owner.json';profile=self.root/'profile.json';owner.write_text(json.dumps({**self.owner,'mode':'preparation_only','allow_NPU_tests':False}));profile.write_text('{}')
        base=[str(script),'--assignment',str(owner),'--profile',str(profile),'--checkpoint',str(self.root),
              '--output-dir',str(self.root/'no-output'),'--task-id',self.identity['task_id'],'--executor',self.identity['executor'],'--generation','2']
        with patch.object(sys,'meta_path',[NativeImportTrap(),*sys.meta_path]):
            with patch.object(sys,'argv',base),self.assertRaises(SystemExit) as e:runpy.run_path(str(script),run_name='__main__')
            self.assertEqual(e.exception.code,2)
            with patch.object(sys,'argv',base+['--authorized-npu']),self.assertRaisesRegex(ValueError,'current control forbids'):runpy.run_path(str(script),run_name='__main__')
        self.assertFalse((self.root/'no-output').exists())


if __name__=='__main__':unittest.main(verbosity=2)
