# SPDX-License-Identifier: Apache-2.0
"""Small file-byte tests and metadata/symmetric-channel guards;no INT8 math."""
from dataclasses import replace
from pathlib import Path
import hashlib
import os
import tempfile
import unittest
from unittest.mock import patch
import torch
from test_glm5_next_sfa_weights import config
from test_glm5_next_sfa_inputs import fixture
from deepinfer.model_executor.models.glm5_next.sfa_inputs import SFAHeadPartition
from deepinfer.model_executor.models.glm5_next.oproj_payload import read_i8_columns_cpu,o_proj_column_manifest,load_o_proj_columns_cpu


class Reader:
    def __init__(self):
        self.quant={};self.desc={};self.reads=[];self.max_read_bytes=8*1024**2
        for layer in (3,43):
            p=f'model.language_model.layers.{layer}.self_attn.o_proj'
            for suffix,dtype,shape,size in (('weight','I8',[4096,16384],4096*16384),('weight_scale','F32',[4096,1],4096*4),('weight_offset','F32',[4096,1],4096*4)):
                self.quant[p+'.'+suffix]='W8A8_DYNAMIC';self.desc[p+'.'+suffix]={'dtype':dtype,'shape':shape,'data_offsets':[0,size]}
        self.scale=torch.full((4096,1),.1234);self.offset=torch.zeros(4096,1)
    def descriptor(self,name):return Path('inert-INT8-shard'),32,self.desc[name]
    def tensor(self,name):
        self.reads.append(name)
        if name.endswith('.weight_scale'):return self.scale,'F32'
        if name.endswith('.weight_offset'):return self.offset,'F32'
        raise AssertionError('full weight tensor read')
    def matrix(self,*a,**k):raise AssertionError('matrix/dequantize path')
    materialize=load_linear=linear_payload=matrix


def setup(rank=0,layer=3,count=3):
    c=config();c.hidden_size=4096;p,*_=fixture(start=3,count=count,rank=rank);p=replace(p,layer_id=layer);return Reader(),c,p


def fake_weight(*a,**k):return torch.full((4096,1024),-7,dtype=torch.int8)


class OProjPayloadTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_real_small_file_columns_preserve_signed_bytes_and_bound_each_read(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'],prefix='oproj-columns-') as root:
            path=Path(root)/'bytes';data=bytes((i*7+128)%256 for i in range(8*17));path.write_bytes(b'HEADER'+data)
            class FileReader:
                max_read_bytes=256
                def descriptor(self,name):return path,6,{'dtype':'I8','shape':[8,17],'data_offsets':[0,136]}
            r=FileReader();calls=[];original=Path.open
            class Tracked:
                def __init__(self,f):self.f=f
                def __enter__(self):return self
                def __exit__(self,*a):self.f.close()
                def seek(self,p):return self.f.seek(p)
                def read(self,n):calls.append(n);return self.f.read(n)
            with patch.object(Path,'open',lambda p,*a,**k:Tracked(original(p,*a,**k))):value=read_i8_columns_cpu(r,'w',13,17)
            expected=b''.join(data[row*17+13:row*17+17] for row in range(8))
            self.assertEqual(value.dtype,torch.int8);self.assertEqual(value.shape,(8,4));self.assertEqual(value.view(torch.uint8).numpy().tobytes(),expected);self.assertEqual(calls,[4]*8);self.assertTrue(bool((value<0).any()))

    def test_column_budget_range_dtype_and_EOF_guards(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'],prefix='oproj-guards-') as root:
            path=Path(root)/'short';path.write_bytes(bytes(5))
            class R:
                max_read_bytes=32
                def descriptor(self,name):return path,0,{'dtype':'I8','shape':[2,8],'data_offsets':[0,16]}
            r=R()
            for first,stop in ((-1,2),(1,9),(2,2),(True,3)):
                with self.assertRaises(ValueError):read_i8_columns_cpu(r,'w',first,stop)
            with patch.object(Path,'open',side_effect=AssertionError('open before budget')):
                with self.assertRaisesRegex(ValueError,'byte budget'):read_i8_columns_cpu(r,'w',0,4,max_bytes=1)
            with self.assertRaises(EOFError):read_i8_columns_cpu(r,'w',0,4)
            class Wrong(R):
                def descriptor(self,name):return path,0,{'dtype':'BF16','shape':[2,8],'data_offsets':[0,32]}
            with self.assertRaises(ValueError):read_i8_columns_cpu(Wrong(),'w',0,4)

    def test_manifest_is_input_column_shard_with_all_output_channels(self):
        for rank,layer,columns in ((0,3,(0,1024)),(15,43,(15360,16384))):
            r,c,p=setup(rank,layer);m=o_proj_column_manifest(r,c,layer,p.heads,p.contract)
            self.assertEqual((m.first_column,m.stop_column),columns);self.assertEqual(m.full_shape,(4096,16384));self.assertEqual(m.weight_bytes,4194304)
            self.assertEqual(m.checkpoint_bytes,4194304+32768);self.assertEqual(m.runtime_bytes,4194304+8192);self.assertEqual(r.reads,[])

    def test_INT8_payload_and_full_BF16_scale_owned_without_compute(self):
        r,c,p=setup()
        with patch('deepinfer.model_executor.models.glm5_next.oproj_payload.read_i8_columns_cpu',side_effect=fake_weight) as read,patch('torch.matmul',side_effect=AssertionError('INT8 compute forbidden')):
            q=load_o_proj_columns_cpu(r,c,3,p.heads,p.contract)
        self.assertEqual(read.call_args.args[2:],(0,1024));self.assertEqual(q.weight_nk.shape,(4096,1024));self.assertEqual(q.weight_nk.dtype,torch.int8);self.assertTrue(bool((q.weight_nk==-7).all()))
        self.assertTrue(torch.equal(q.weight_scale,r.scale.flatten().bfloat16()));self.assertEqual(q.weight_scale.shape,(4096,));self.assertEqual(len(r.reads),2)
        self.assertEqual(q.weight_sha256,hashlib.sha256(q.weight_nk.view(torch.uint8).numpy().tobytes()).hexdigest())
        with torch.inference_mode():r.scale.zero_()
        self.assertTrue(bool((q.weight_scale>0).all()))

    def test_bad_descriptor_recipe_unsharded_and_budgets_before_payload(self):
        for kind in ('recipe','shape','dtype','range','heads','checkpoint','runtime','reader'):
            r,c,p=setup();kw={}
            if kind=='recipe':r.quant[next(iter(r.quant))]='FLOAT'
            elif kind=='shape':r.desc[prefix(r,1)]['shape']=[256,1]
            elif kind=='dtype':r.desc[prefix(r,0)]['dtype']='BF16'
            elif kind=='range':r.desc[prefix(r,0)]['data_offsets'][1]-=1
            elif kind=='heads':p=replace(p,heads=SFAHeadPartition(64,1,0,tuple(range(64))))
            elif kind=='checkpoint':kw['max_checkpoint_bytes']=1
            elif kind=='runtime':kw['max_runtime_bytes']=1
            else:r.max_read_bytes=1
            with self.subTest(kind=kind),patch('deepinfer.model_executor.models.glm5_next.oproj_payload.read_i8_columns_cpu',side_effect=AssertionError('weight read before guard')):
                with self.assertRaises(ValueError):load_o_proj_columns_cpu(r,c,3,p.heads,p.contract,**kw)
            self.assertEqual(r.reads,[])

    def test_asymmetric_nonfinite_and_bad_rounded_scale_before_weight_read(self):
        for kind in ('zero','negative','nan','offset','offsetnan','overflow','underflow'):
            r,c,p=setup()
            if kind=='offset':r.offset[0]=1
            elif kind=='offsetnan':r.offset[0]=float('nan')
            else:r.scale[0]={'zero':0,'negative':-1,'nan':float('nan'),'overflow':3.4e38,'underflow':1e-45}[kind]
            with self.subTest(kind=kind),patch('deepinfer.model_executor.models.glm5_next.oproj_payload.read_i8_columns_cpu',side_effect=AssertionError('read weight before channels')):
                with self.assertRaises(ValueError):load_o_proj_columns_cpu(r,c,3,p.heads,p.contract)

    def test_metadata_drift_or_wrong_weight_payload_rejects(self):
        for kind in ('drift','wrong'):
            r,c,p=setup()
            def bad(*args,**kwargs):
                if kind=='drift':r.desc[prefix(r,0)]['data_offsets']=[16,16+4096*16384];return fake_weight()
                return torch.zeros(1,1,dtype=torch.uint8)
            with patch('deepinfer.model_executor.models.glm5_next.oproj_payload.read_i8_columns_cpu',side_effect=bad),self.assertRaises(ValueError):load_o_proj_columns_cpu(r,c,3,p.heads,p.contract)

    def test_plan_matched_input_flatten_is_copy_with_empty_and_error_guards(self):
        r,c,p=setup()
        with patch('deepinfer.model_executor.models.glm5_next.oproj_payload.read_i8_columns_cpu',side_effect=fake_weight):q=load_o_proj_columns_cpu(r,c,3,p.heads,p.contract)
        values=torch.arange(3*4*256).reshape(3,4,256).bfloat16();result=q.prepare_input_cpu(values,plan=p)
        self.assertTrue(torch.equal(result,values.reshape(3,1024)))
        with torch.inference_mode():result.zero_()
        self.assertFalse(bool((values==0).all()))
        wrong=replace(p,heads=SFAHeadPartition(64,16,15,(60,61,62,63)))
        for bad in (wrong,replace(p,layer_id=7)):
            with self.assertRaises(ValueError):q.prepare_input_cpu(values,plan=bad)
        with self.assertRaises(ValueError):q.prepare_input_cpu(values,plan=p,max_input_bytes=1)
        with self.assertRaises(ValueError):q.prepare_input_cpu(values.float(),plan=p)
        with self.assertRaises(ValueError):q.prepare_input_cpu(torch.full_like(values,float('nan')),plan=p)
        empty,*_=fixture(start=3,count=0);self.assertEqual(q.prepare_input_cpu(torch.empty(0,4,256,dtype=torch.bfloat16),plan=empty).shape,(0,1024))


def prefix(r,index):return list(r.desc)[index]
if __name__=='__main__':unittest.main(verbosity=2)
