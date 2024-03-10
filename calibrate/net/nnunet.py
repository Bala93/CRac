
from torch import nn
from monai.networks.nets import DynUNet

class NNUNet(nn.Module):
    
    def __init__(self, input_channels=1, num_classes=2):
        
        super().__init__()
    
        kernel_size = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
        strides = [[1, 1],[2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]

        self.net = DynUNet(
            spatial_dims=2,
            in_channels=input_channels,
            out_channels=num_classes,
            kernel_size=kernel_size,
            strides=strides,
            upsample_kernel_size=strides[1:],
            norm_name="instance",
            res_block=True
            # deep_supervision=True,
            # deep_supr_num=3,
        )
        
    def forward(self,x):
        
        return self.net(x)