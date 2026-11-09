"""Opt-in asynchronous observations of original target/draft handoffs.

Capture records retain graph tensor addresses. A diagnostic marker enables
owned device snapshots; serialization runs in the original post-sync phase.
No CPU model arithmetic, synchronization, or forced tokens are introduced.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib,json,os
from pathlib import Path
import torch

_capture=ContextVar('deepinfer_draft_capture_observer',default=None)
_tape=ContextVar('deepinfer_decode_tape',default=None)
_steps={}


def root():
    value=os.environ.get('DEEPINFER_910C_INVARIANT_TRACE_DIR')
    return Path(value) if value else None


@contextmanager
def capture(module,count,decode_input):
    if root() is None:
        yield None
        return
    record=dict(inputs=decode_input,values={},samplers=[])
    token=_capture.set(record)
    try:
        yield record
    finally:
        _capture.reset(token)
        if not hasattr(module,'_decode_capture_records'):module._decode_capture_records={}
        module._decode_capture_records[count]=record


def sampler_before(**values):
    captured=_capture.get()
    if captured is not None:
        captured['samplers'].append(dict(values))
        return captured['samplers'][-1]
    tape=_tape.get()
    if tape is not None:
        previous=values.pop('previous',None)
        previous_tokens=values.pop('previous_tokens',None)
        rows=values.get('read_rows')
        if previous is not None and rows is not None:
            values['previous']=previous.index_select(0,rows.long())
            values['previous_tokens']=previous_tokens.index_select(0,rows.long())
        return tape.copy('target_sampler'+str(tape.sampler_count),values)
    return None


def sampler_after(record,**values):
    if record is None:return
    captured=_capture.get()
    if captured is not None:
        record.update(values)
    else:
        tape=_tape.get()
        tape.copy('target_sampler'+str(tape.sampler_count)+'_output',values)
        tape.sampler_count+=1


def main_update(owner,features,hidden,positions,requests,keys):
    if root() is None or owner is None:return
    owner._decode_main_update=dict(features=tuple(features),projected=hidden,
        positions=positions,requests=requests,keys=tuple(keys))


def tensors(value,prefix=''):
    if isinstance(value,torch.Tensor):
        yield prefix,value
    elif isinstance(value,dict):
        for name,item in value.items():yield from tensors(item,prefix+'/'+str(name))
    elif isinstance(value,(tuple,list)):
        for index,item in enumerate(value):yield from tensors(item,prefix+'/'+str(index))


class Tape:
    def __init__(self,path,config,number,executor,inputs,sampler_inputs):
        self.path,self.config,self.number=path,config,number
        self.executor,self.inputs=executor,inputs
        self.values={};self.device_sources=[];self.sampler_count=0;self.bytes=0
        self.copy('main_metadata',{name:getattr(inputs,name,None) for name in
            ('input_ids','positions','state_block_idx','valid_token_mask')})
        owner=getattr(inputs,'_decode_observer_owner',executor.model)
        self.copy('main_update',getattr(owner,'_decode_main_update',{}))
        data=sampler_inputs.driver
        if data is not None:
            self.copy('request_sampling',dict(seeds=data.kernel.seq_seeds,
                positions=data.kernel.sample_positions))
        self.metadata=dict(rank=int(os.environ.get('RANK','0')),phase=config['phase'],
            step=number,is_prompt=bool(inputs.is_prompt),
            rows=[dict(seq_id=row.seq_id,state_id=row.state_id,seqlen_q=row.seqlen_q,
                       seqlen_k=row.seqlen_k) for row in inputs.original_packet.rows])

    def copy(self,name,value):
        for field,tensor in tensors(value):
            if tensor.dtype==torch.uint64:tensor=tensor.view(torch.int64)
            size=tensor.numel()*tensor.element_size()
            if self.bytes+size>self.config.get('max_bytes_per_step',128*1024**2):continue
            # Enqueue D2H before the original sample completion event. The
            # post callback must not append a new NPU copy after a later
            # target forward has already been launched by original ZOS.
            host=torch.empty_like(tensor,device='cpu',pin_memory=True,
                                  memory_format=torch.contiguous_format)
            host.copy_(tensor,non_blocking=True)
            self.values[name+field]=host
            self.device_sources.append(tensor)
            self.bytes+=size
        return True

    def draft(self,module,request_count):
        runner=module.decode_graph_runner
        count,_=runner.runner.select(request_count)
        record=getattr(module,'_decode_capture_records',{}).get(count)
        if record is None:return
        self.copy('draft_values',record['values'])
        self.copy('draft_sampler',record['samplers'])
        current=record['inputs'];driver=current.driver
        fields=dict(first=current.first_token_positions)
        if driver is not None:
            fields.update(offset=driver.last_valid_offset,golden=driver.golden_tokens,
                seeds=driver.sampling.kernel_inputs.seq_seeds,
                positions=driver.sampling.kernel_inputs.sample_positions,
                write_slots=driver.sampling.mtp_info_write_slots)
        self.copy('draft_metadata',fields)
        slots=[self.executor.cache_storage.state_slots[row.state_id] for row in self.inputs.original_packet.rows]
        self.metadata['context_slots']=slots
        for index,plane in enumerate(module.draft.planes):
            indices=torch.tensor([row for slot in slots for row in
                range(slot*plane.resident_pages,(slot+1)*plane.resident_pages)],
                dtype=torch.int64,device=plane.cache.device)
            self.copy('draft_context'+str(index),plane.cache.index_select(0,indices))

    def write(self):
        rank=self.metadata['rank'];phase=self.metadata['phase']
        directory=self.path/'decode'/phase/f'rank{rank}'/f'step{self.number}'
        directory.mkdir(parents=True,exist_ok=False)
        records=[]
        for index,(name,tensor) in enumerate(self.values.items()):
            bits=tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
            filename=str(index)+'.bin';(directory/filename).write_bytes(bits)
            records.append(dict(name=name,shape=list(tensor.shape),dtype=str(tensor.dtype),
                file=filename,sha256=hashlib.sha256(bits).hexdigest()))
        (directory/'MANIFEST.json').write_text(json.dumps(dict(**self.metadata,tensors=records),indent=2)+'\n')


@contextmanager
def observe(executor,inputs,sampler_inputs):
    path=root();marker=path/'decode-enable' if path else None
    if marker is None or not marker.exists():
        yield None
        return
    if not inputs.original_packet.rows:
        yield None
        return
    config=json.loads(marker.read_text());phase=config['phase']
    if not isinstance(phase,str) or not phase.isidentifier():raise ValueError('invalid decode diagnostic phase')
    key=(int(os.environ.get('RANK','0')),phase)
    number=_steps.get(key,0);_steps[key]=number+1
    if number>=config.get('max_steps',3):
        yield None
        return
    tape=Tape(path,config,number,executor,inputs,sampler_inputs)
    token=_tape.set(tape)
    try:yield tape
    finally:_tape.reset(token)


def observe_draft(module,request_count):
    tape=_tape.get()
    if tape is not None:tape.draft(module,request_count)
