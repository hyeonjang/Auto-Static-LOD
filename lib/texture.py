# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# from nvdiffmodeling

import numpy as np
import torch

def bilinear_downsample(x : torch.tensor, spp) -> torch.Tensor:
    w = torch.tensor([[1, 3, 3, 1], [3, 9, 9, 3], [3, 9, 9, 3], [1, 3, 3, 1]], dtype=torch.float32, device=x.device) / 64.0
    g = x.shape[-1]
    w = w.expand(g, 1, 4, 4) 
    x = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    steps = int(np.log2(spp))
    for _ in range(steps):
        xp = torch.nn.functional.pad(x, (1,1,1,1), mode='replicate')
        x = torch.nn.functional.conv2d(xp, w, padding=0, stride=2, groups=g)
    return x.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

def scale_img_hwc(x : torch.Tensor, size, mag='bilinear', min='area') -> torch.Tensor:
    return scale_img_nhwc(x[None, ...], size, mag, min)[0]

def scale_img_nhwc(x  : torch.Tensor, size, mag='bilinear', min='area') -> torch.Tensor:
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (x.shape[1] < size[0] and x.shape[2] < size[1]), "Trying to magnify image in one dimension and minify in the other"
    y = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]: # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else: # Magnification
        if mag == 'bilinear' or mag == 'bicubic':
            y = torch.nn.functional.interpolate(y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

def avg_pool_nhwc(x  : torch.Tensor, size) -> torch.Tensor:
    y = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    y = torch.nn.functional.avg_pool2d(y, size)
    return y.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

# trainable texture creation
def create(init, res, auto_mipmaps):
    with torch.no_grad():
        if isinstance(init, np.ndarray):
            init = torch.tensor(init, dtype=torch.float32, device='cuda')

        # Pad to NHWC if needed
        if len(init.shape) == 1: # Extend constant to NHWC tensor
            init = init[None, None, None, :]
        elif len(init.shape) == 3:
            init = init[None, ...]

        # Scale input to desired resolution.
        init = scale_img_nhwc(init, res)

        # Genreate custom mipchain
        if not auto_mipmaps:
            mip_chain = [init.clone().detach().requires_grad_(True)]
            while mip_chain[-1].shape[1] > 1 or mip_chain[-1].shape[2] > 1:
                new_size = [max(mip_chain[-1].shape[1] // 2, 1), max(mip_chain[-1].shape[2] // 2, 1)]
                init = scale_img_nhwc(mip_chain[-1], new_size)
                mip_chain += [init.clone().detach().requires_grad_(True)]
            return mip_chain
        else:
            return init.clone().detach().requires_grad_(True)

########################################################################################################
# Utility functions for loading / storing a texture
########################################################################################################
import imageio

def save_image(fn, x : np.ndarray) -> np.ndarray:
    imageio.imwrite(fn, np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8))

def load_image(fn) -> np.ndarray:
    img = imageio.imread(fn)
    # img = np.flipud(img)
    if img.dtype == np.float32: # HDR image
        return img
    else: # LDR image
        return img.astype(np.float32) / 255
