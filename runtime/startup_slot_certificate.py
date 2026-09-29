"""Conservative first-eight-cycle slot-range certificate for frozen 12x8 serving.

Only valid inside one serialized worker RPC with the standard fixed state
advance (0..8 tokens/cycle), immutable bound block tables and scheduling OFF.
A failed or invalidated certificate falls back to the original device guards.
"""
import torch

def _version(tensor):
    try:return tensor._version
    except RuntimeError as exc:
        if 'Inference tensors do not track version counter' not in str(exc):raise
        # Source-pinned worker serialization/read-only table ownership is the
        # primary invariant. Version counters add detection where available.
        return None

def _identity(binding):
    gid,table,mapping,block_size=binding
    return (gid,id(table),table.data_ptr(),tuple(table.shape),tuple(table.stride()),
            table.dtype,table.device,table.storage_offset(),_version(table),
            id(mapping),mapping.data_ptr(),tuple(mapping.shape),tuple(mapping.stride()),mapping.dtype,
            mapping.device,mapping.storage_offset(),block_size)

class StartupSlotCertificate:
    def __init__(self):
        self.record={'eligible':False,'reason':'not_checked','invalidated':False,'used_cycles':0,'groups':[]}
        self._identities=[]
        self._next_cycle=0

    @classmethod
    def build(cls,*,bindings,positions,remaining,cycle_index,schedule_mode,batch_size,width,computed_dtype=torch.int32):
        obj=cls()
        def reject(reason):obj.record['reason']=reason;return obj
        if cycle_index!=0 or schedule_mode!='off' or batch_size!=12 or width!=8:
            return reject('outside_frozen_startup_contract')
        if len(positions)!=12 or len(remaining)!=12 or any(type(x) is not int or x<0 for x in positions):
            return reject('invalid_initial_positions')
        if computed_dtype not in (torch.int32,torch.int64) or max(positions)+64>torch.iinfo(computed_dtype).max:
            return reject('computed_position_overflow')
        if any(type(x) is not int or x<=64 for x in remaining):return reject('possible_early_parking')
        if not bindings or len({b[0] for b in bindings})!=len(bindings):return reject('invalid_bindings')
        # Check cross-group aliases too: an earlier group's mapping copy can
        # otherwise overwrite a later group's certified table in the same cycle.
        for _,table,_,_ in bindings:
            for _,_,mapping,_ in bindings:
                if not table.is_contiguous() or not mapping.is_contiguous():
                    return reject('unsupported_table_layout')
                t0=table.data_ptr();t1=t0+table.numel()*table.element_size()
                m0=mapping.data_ptr();m1=m0+mapping.numel()*mapping.element_size()
                if table.device==mapping.device and max(t0,m0)<min(t1,m1):
                    return reject('table_mapping_alias')
        identities=[];groups=[]
        for binding in bindings:
            gid,table,mapping,bs=binding
            if type(bs) is not int or bs<=0 or table.ndim!=2 or table.shape[0]!=12:
                return reject('invalid_table_shape')
            if table.dtype not in (torch.int32,torch.int64) or not table.is_contiguous() or not mapping.is_contiguous():
                return reject('unsupported_table_layout')
            # A mapping write must not mutate the certified table bytes.
            t0=table.data_ptr();t1=t0+table.numel()*table.element_size()
            m0=mapping.data_ptr();m1=m0+mapping.numel()*mapping.element_size()
            if table.device==mapping.device and max(t0,m0)<min(t1,m1):return reject('table_mapping_alias')
            low=[p//bs for p in positions];high=[(p+63)//bs for p in positions]
            if max(high)>=table.shape[1]:return reject('reachable_range_outside_table')
            left,right=min(low),max(high)+1
            if right-left>256:return reject('certificate_copy_span_too_large')
            before=_identity(binding)
            # One compact read per group, before any Runtime target is issued.
            snapshot=table[:,left:right].cpu()
            for row,(lo,hi) in enumerate(zip(low,high)):
                if bool((snapshot[row,lo-left:hi-left+1]<0).any()):
                    return reject('negative_reachable_block')
            if _identity(binding)!=before:return reject('table_changed_during_certificate')
            identities.append(before)
            groups.append({'gid':gid,'block_size':bs,'left':left,'right_exclusive':right,'version_tracked':before[8] is not None})
        obj._identities=identities
        obj.record.update(eligible=True,reason='certified_p0_through_p0_plus63',groups=groups)
        return obj

    def permits(self,cycle,bindings):
        if not self.record['eligible'] or self.record['invalidated'] or not 0<=cycle<8:return False
        if cycle!=self._next_cycle or [_identity(b) for b in bindings]!=self._identities:
            self.record.update(invalidated=True,reason='cycle_or_binding_generation_changed')
            return False
        self._next_cycle+=1;self.record['used_cycles']+=1
        return True
