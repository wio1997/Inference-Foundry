"""Restricted stdlib reader for the saved routing TensorList format; never imports torch."""
import io,itertools,pickle,pickletools,struct,zipfile
from collections import OrderedDict

STORAGE_TYPES={'IntStorage':('i',4),'LongStorage':('q',8),'BoolStorage':('?',1),
    'ByteStorage':('B',1),'CharStorage':('b',1),'ShortStorage':('h',2),
    'FloatStorage':('f',4),'DoubleStorage':('d',8)}

class Storage:
    def __init__(self,kind,raw,numel):self.kind=kind;self.raw=raw;self.numel=numel

class Tensor:
    def __init__(self,storage,offset,size,stride):
        assert isinstance(storage,Storage)and isinstance(offset,int)and offset>=0
        assert isinstance(size,tuple)and isinstance(stride,tuple)and len(size)==len(stride)<=4
        assert all(isinstance(x,int)and 0<=x<=65536 for x in size+stride)
        self.storage=storage;self.offset=offset;self.size=size;self.stride=stride
        count=1
        for dim in size:count*=dim
        assert count<=1000000
        if count:assert offset+sum((d-1)*s for d,s in zip(size,stride))<storage.numel
        self.numel=count

    def values(self):
        fmt,width=self.storage.kind
        if not self.numel:return []
        step=1;contig=True
        for dim,stride in zip(reversed(self.size),reversed(self.stride)):
            if dim>1 and stride!=step:contig=False
            step*=dim
        if contig:
            start=self.offset*width;end=start+self.numel*width
            return list(struct.unpack('<'+str(self.numel)+fmt,self.storage.raw[start:end]))
        return [struct.unpack_from('<'+fmt,self.storage.raw,(self.offset+sum(i*s for i,s in zip(coord,self.stride)))*width)[0]
            for coord in itertools.product(*(range(d)for d in self.size))]

def rebuild(storage,offset,size,stride,requires_grad,hooks,metadata=None):
    assert requires_grad is False and not hooks and metadata in [None,{}]
    return Tensor(storage,offset,size,stride)

class RestrictedReader(pickle.Unpickler):
    def __init__(self,raw,archive,prefix):super().__init__(io.BytesIO(raw));self.archive=archive;self.prefix=prefix
    def find_class(self,module,name):
        if(module,name)==('torch._utils','_rebuild_tensor_v2'):return rebuild
        if(module,name)==('collections','OrderedDict'):return OrderedDict
        if module=='torch'and name in STORAGE_TYPES:return STORAGE_TYPES[name]
        raise ValueError('Unsupported saved-global '+module+'.'+name)
    def persistent_load(self,token):
        assert isinstance(token,tuple)and len(token)==5 and token[0]=='storage'
        _,kind,key,location,n=token
        assert kind in STORAGE_TYPES.values()and isinstance(key,str)and key.isdigit()and len(key)<=8
        assert location=='cpu'and isinstance(n,int)and 0<=n<=1000000
        data=self.archive.read(self.prefix+'data/'+key);assert len(data)==n*kind[1]
        return Storage(kind,data,n)

def load_saved_tensor_list(blob):
    assert isinstance(blob,bytes)and len(blob)<=2*1024*1024
    with zipfile.ZipFile(io.BytesIO(blob))as z:
        infos=z.infolist();assert len(infos)<=64 and sum(i.file_size for i in infos)<=2*1024*1024
        paths=[i.filename for i in infos if i.filename.endswith('/data.pkl')];assert len(paths)==1
        prefix=paths[0][:-len('data.pkl')];assert z.read(prefix+'byteorder')==b'little'
        pkl=z.read(paths[0]);assert len(pkl)<=65536
        for op,arg,pos in pickletools.genops(pkl):
            assert op.name not in ['BUILD','EXT1','EXT2','EXT4','NEWOBJ','NEWOBJ_EX','INST','OBJ','PERSID']
        tensors=RestrictedReader(pkl,z,prefix).load()
        assert isinstance(tensors,list)and len(tensors)<=16 and all(isinstance(t,Tensor)for t in tensors)
        return tensors
