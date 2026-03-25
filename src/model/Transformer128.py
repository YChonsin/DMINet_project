# get an object of Transformer
from src.model.TIR import TransformerIR

def Transformer128():
    model = TransformerIR(
                 img_size=(128, 128),
                 patch_size=1,
                 in_chans=1,
                 embed_dim=96,
                 depths=[3, 3],
                 num_heads=[2, 2],
                 mlp_ratio=2.,
                 upsampler='None',
                 resi_connection='1conv',
                 ape=True,
                    )
    return model