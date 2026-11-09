"""Native copies for the original fixed-stride DFlash graph inputs."""
import torch


def copy_dflash_decode_input_tl(has_driver):
    def run(*args):
        if len(args) != 27:
            raise ValueError('original DFlash copy requires three nine-field inputs')
        source, padding, target = args[:9], args[9:18], args[18:]
        count, capacity = source[0].numel(), target[0].numel()
        if not 0 <= count <= capacity or padding[0].numel() < capacity - count:
            raise ValueError('DFlash graph input or padding exceeds its fixed capacity')
        for index, (src, pad, dst) in enumerate(zip(source, padding, target, strict=True)):
            if index and not has_driver:
                if any(value is not None for value in (src, pad, dst)):
                    raise ValueError('follower DFlash inputs cannot own driver fields')
                continue
            if any(not isinstance(value, torch.Tensor) or value.ndim != 1
                   or value.device != target[0].device or value.dtype != dst.dtype
                   for value in (src, pad, dst)):
                raise ValueError('DFlash graph copy fields must share dtype and device')
            if src.numel() != count or dst.numel() != capacity or pad.numel() < capacity - count:
                raise ValueError('DFlash graph input field has a different request count')
        # Equal-width signed seed views preserve every uint64 bit, including
        # strided source fields. Other fields use ordinary strided copies.
        # Zero padding write-slots retain original pool semantics.
        for src, pad, dst in zip(source, padding, target, strict=True):
            if dst is not None:
                if dst.dtype == torch.uint64:
                    src, pad, dst = (value.view(torch.int64) for value in (src, pad, dst))
                dst[:count].copy_(src)
                if capacity != count:
                    dst[count:].copy_(pad[:capacity - count])
    return run
