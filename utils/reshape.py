from mmseg.ops import  resize
import  torch
def reshape(out,img_shape):
    out = resize(
        input=out,
        size=img_shape[2:],
        mode='bilinear',
        align_corners=False,
    )
    return out