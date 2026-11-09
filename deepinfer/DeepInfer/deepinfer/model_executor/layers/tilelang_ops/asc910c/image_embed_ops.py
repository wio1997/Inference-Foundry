"""NPU implementations of the original image metadata/embedding factories."""
import math
import torch

from deepinfer.npu_ops.vision910c import _npu


def replace_input_embeds_tl(hidden):
    def run(input_embeds, input_ids, images_embeds, image_token_id,
            image_token_offset, image_pad_token_ids, image_pad_embeds):
        _npu(input_embeds, input_ids, images_embeds, image_token_offset,
             image_pad_token_ids, image_pad_embeds)
        if (input_embeds.ndim != 2 or input_embeds.shape[-1] != hidden
                or images_embeds.shape[1:] != (hidden,)
                or input_ids.shape != input_embeds.shape[:1]
                or image_token_offset.shape != input_ids.shape
                or image_pad_embeds.shape != (image_pad_token_ids.numel(),hidden)):
            raise ValueError('invalid original image embedding layout')
        if images_embeds.shape[0]:
            rows = (image_token_offset.long()-1).clamp(0,images_embeds.shape[0]-1)
            image_values = images_embeds.index_select(0,rows)
            input_embeds.copy_(torch.where((input_ids==image_token_id)[:,None],image_values,input_embeds))
        for index in range(image_pad_token_ids.numel()):
            input_embeds.copy_(torch.where((input_ids==image_pad_token_ids[index])[:,None],
                                           image_pad_embeds[index],input_embeds))
    return run


def rgb_pad_to_patches_tl(patch_size):
    if type(patch_size) is not int or patch_size < 1:
        raise ValueError('patch size must be positive')
    def run(raw, cu_bytes, sizes, crops, cu_patches, mean, std, pad_color, output):
        _npu(raw,cu_bytes,sizes,crops,cu_patches,output)
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0 or not 0 <= int(pad_color) <= 255:
            raise ValueError('invalid image normalization configuration')
        if (raw.dtype != torch.uint8 or raw.ndim != 1
                or output.dtype != torch.bfloat16 or output.shape[1:] != (3,patch_size,patch_size)
                or sizes.ndim != 2 or sizes.shape[1] != 2 or crops.shape != sizes.shape
                or cu_bytes.shape != (sizes.shape[0]+1,) or cu_patches.shape != cu_bytes.shape):
            raise ValueError('invalid original RGB patch layout')
        if output.shape[0] == 0:
            return
        patch = torch.arange(output.shape[0],dtype=torch.int32,device=raw.device)
        image = torch.searchsorted(cu_patches[1:].contiguous(),patch,right=True).long()
        local = patch-cu_patches.index_select(0,image)
        grid = crops.index_select(0,image).long()
        real = sizes.index_select(0,image).long()
        gap = grid*patch_size-real
        # Python/PIL round(gap*0.5), including odd ties rounding to even.
        border = torch.div(gap,2,rounding_mode='floor')+(gap.remainder(4)==3)
        offset = torch.arange(patch_size,device=raw.device,dtype=torch.int64)
        y = (torch.div(local,grid[:,0],rounding_mode='floor')*patch_size-border[:,1])[:,None]+offset
        x = (local.remainder(grid[:,0])*patch_size-border[:,0])[:,None]+offset
        valid = (y[:,:,None]>=0)&(y[:,:,None]<real[:,1,None,None])&(x[:,None,:]>=0)&(x[:,None,:]<real[:,0,None,None])
        position = (y[:,:,None]*real[:,0,None,None]+x[:,None,:])*3
        channel = torch.arange(3,device=raw.device,dtype=torch.int64)
        indices = cu_bytes.index_select(0,image)[:,None,None,None]+position[:,None]+channel[None,:,None,None]
        values = raw.index_select(0,indices.flatten().clamp(0,raw.numel()-1)).view_as(output).float()
        values = torch.where(valid[:,None],values,float(int(pad_color)))
        output.copy_(((values/255.0-mean)/std).to(output.dtype))
    return run


def batch_downsample_tl(ratio, dim, do_downsample, dtype):
    if type(ratio) is not int or ratio < 1 or type(dim) is not int or dim < 1:
        raise ValueError('invalid spatial downsample dimensions')
    def run(features, images_id, crops, cu_downsample, cu_patch, embeds, pooled):
        target = pooled if do_downsample else embeds
        _npu(features,images_id,crops,cu_downsample,cu_patch,target)
        if features.shape[1:] != (dim,) or features.dtype != dtype or target.dtype != dtype:
            raise ValueError('invalid original downsample feature layout')
        if target.shape[0] == 0:
            return
        image = images_id.long()
        grid = crops.index_select(0,image).long()
        local = torch.arange(target.shape[0],device=features.device)-cu_downsample.index_select(0,image)
        width = torch.div(grid[:,0]+ratio-1,ratio,rounding_mode='floor')
        delta = torch.arange(ratio,device=features.device)
        y = (torch.div(local,width,rounding_mode='floor')*ratio)[:,None]+delta
        x = (local.remainder(width)*ratio)[:,None]+delta
        valid = (y[:,:,None]<grid[:,1,None,None])&(x[:,None,:]<grid[:,0,None,None])
        rows = cu_patch.index_select(0,image)[:,None,None]+y[:,:,None]*grid[:,0,None,None]+x[:,None,:]
        block = features.index_select(0,rows.flatten().clamp(0,features.shape[0]-1).long()).view(-1,ratio,ratio,dim)
        block = torch.where(valid[:,:,:,None],block,0.)
        if do_downsample:
            inverse = 1.0/valid.sum((1,2)).float()
            result = block.float().sum((1,2))*inverse[:,None]
            target.copy_(result.to(dtype))
        else:
            target.copy_(block.permute(0,3,1,2))
    return run
