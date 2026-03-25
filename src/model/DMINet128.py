from src.model.DMINet import DMINet


def DMINet128():
    model = DMINet(img_size=(128, 128),
                 patch_size=1,
                 in_chans=1,
                 embed_dim=96,
                 depths=[4, 4],
                 drop_rate=0.,
                 d_state = 16,
                 mlp_ratio=2.,
                 drop_path_rate=0.1,
                 patch_norm=True,
                 img_range=1.,
                 resi_connection='1conv',
    )
    return model
