"""Bounded observation of actual SFA cache-address and length tensors."""


def install(ns):
    import torch
    fields=('slot_mapping','seq_lens','seq_lens_cpu','cum_query_lens','block_table','query_start_loc')
    scalars=('num_actual_tokens','num_input_tokens','num_decodes','num_decode_tokens',
             'num_prefills','block_size','max_query_len','max_seq_len','attn_state')

    def prepare(metadata,tensors):
        report={'groups':[],'refused':[],'scope':'Existing SFA mapping/length tensors at target return, before native sampling/drafting; not cache content equality.'}
        if not isinstance(metadata,dict):
            report['refused'].append('attention metadata is not a layer mapping');return report
        seen={}
        for name,meta in metadata.items():
            if not type(meta).__name__.startswith('AscendSFA'):continue
            identity=tuple(id(getattr(meta,field,None)) for field in ('slot_mapping','seq_lens','cum_query_lens','block_table'))
            if identity in seen:
                seen[identity]['layers'].append(name);continue
            if len(report['groups'])>=2:
                report['refused'].append('more than two distinct SFA mapping groups');break
            index=len(report['groups']);group={'class':type(meta).__name__,'layers':[name],'scalars':{},'cpu_values':{},'tensor_keys':{},'source_layouts':{}}
            for field in scalars:
                value=getattr(meta,field,None)
                group['scalars'][field]=value if value is None or type(value) in (int,float,bool,str) else str(value)
            for field in fields:
                value=getattr(meta,field,None)
                if value is None:continue
                if not isinstance(value,torch.Tensor) or value.numel()>16384:
                    report['refused'].append(type(meta).__name__+'.'+field+': unsupported type/extent');continue
                group['source_layouts'][field]={'shape':list(value.shape),'stride':list(value.stride()),'dtype':str(value.dtype),'device_type':value.device.type}
                if value.device.type=='cpu':group['cpu_values'][field]=value.tolist();continue
                if value.device.type not in ('npu','privateuseone') or not (value.is_contiguous() or value.ndim==2 and value.stride(-1)==1):
                    report['refused'].append(type(meta).__name__+'.'+field+': unsupported device/strides');continue
                key='attention%d__%s'%(index,field);tensors[key]=value;group['tensor_keys'][field]=key
            seen[identity]=group;report['groups'].append(group)
        if not report['groups']:report['refused'].append('no SFA mapping group found')
        return report
    ns['_glm53_prepare_attention_snapshot']=prepare
