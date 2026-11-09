"""Bounded checkpoint row reads must preserve ordering and reject unsafe ranges."""
import json
import struct

import pytest
import torch

from deepinfer.npu_ops.checkpoint910c import ByteCheckpoint


@pytest.fixture
def table(tmp_path):
    value=torch.arange(35).reshape(7,5).bfloat16()
    raw=value.view(torch.uint8).numpy().tobytes()
    header=json.dumps({'embed.weight':{'dtype':'BF16','shape':[7,5],'data_offsets':[0,len(raw)]}}).encode()
    (tmp_path/'shard.safetensors').write_bytes(struct.pack('<Q',len(header))+header+raw)
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'embed.weight':'shard.safetensors'}}))
    return tmp_path,value


def test_rows_duplicates_order_and_shape(table):
    root,value=table;reader=ByteCheckpoint(root,max_read_bytes=40)
    indices=torch.tensor([[6,1],[1,0]])
    torch.testing.assert_close(reader.rows('embed.weight',indices),value[indices],rtol=0,atol=0)
    with pytest.raises(ValueError,match='large full-tensor'):
        reader.tensor('embed.weight')


@pytest.mark.parametrize('indices',[[-1],[7],[1.5]])
def test_invalid_rows(table,indices):
    root,_=table
    with pytest.raises(ValueError):ByteCheckpoint(root).rows('embed.weight',indices)


def test_rows_read_limit_and_empty(table):
    root,_=table;reader=ByteCheckpoint(root,max_read_bytes=9)
    with pytest.raises(ValueError,match='large row'):
        reader.rows('embed.weight',[0])
    assert reader.rows('embed.weight',torch.empty(0,dtype=torch.int64)).shape==(0,5)


def test_quantized_table_row_geometry(tmp_path):
    entries={};payload=bytearray()
    for name,value,dtype in [('layers.1.engram.embed.weight',torch.full((3,32),0x38,dtype=torch.uint8),'F8_E4M3'),('layers.1.engram.embed.scale',torch.tensor([[127],[128],[126]],dtype=torch.uint8),'F8_E8M0')]:
        raw=value.numpy().tobytes();entries[name]={'dtype':dtype,'shape':list(value.shape),'data_offsets':[len(payload),len(payload)+len(raw)]};payload.extend(raw)
    header=json.dumps(entries).encode();(tmp_path/'data').write_bytes(struct.pack('<Q',len(header))+header+payload)
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':dict.fromkeys(entries,'data')}))
    reader=ByteCheckpoint(tmp_path,max_read_bytes=64);indices=torch.tensor([2,1])
    result=reader.table_rows('layers.1.engram.embed.weight',indices,'cpu')
    torch.testing.assert_close(result,torch.tensor([.5,2.]).bfloat16()[:,None].expand(2,32),rtol=0,atol=0)
    with pytest.raises(ValueError,match='unquantized'):
        reader.rows('layers.1.engram.embed.weight',indices)
