import torch
from torch import nn
import torch.nn.functional as F

class DynamicPENeckSOFT(nn.Module):
    """
    PEMASKNeck. Code is modified based on the codebase of GEDepth.
    """
    def __init__(self, in_channels:list = [512, 256, 128, 64, 32, 16], output_channel=11, kernel_size=3, padding=1,stride=1):
        """
        in_channels: 每个卷积层的输入通道数, 对应backbone提取的多尺度特征
        output_channel: 最终输出的通道数
        """
        super(DynamicPENeckSOFT, self).__init__()
        in_channels.sort(reverse=True)
        self.in_channels = in_channels
        self.convfinal = nn.Conv2d(in_channels[-1], output_channel, kernel_size=kernel_size, padding=padding, stride=stride)
        for i, c in enumerate(in_channels):
            setattr(self, f"conv{i}", nn.Conv2d(c, in_channels[-1], kernel_size=kernel_size, padding=padding, stride=stride))

        

    # init weight
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # xavier_init(m, distribution='uniform')
                nn.init.xavier_uniform_(m.weight)
        
            
    def forward(self, inputs):
        scale_feats = inputs[::-1]
        x = torch.zeros_like(scale_feats[-1])
        for i in range(len(self.in_channels)):
            """
            每个尺度先过卷积层, 然后上采样到最大尺度的尺寸
            """
            conv_layer = getattr(self, f"conv{i}")
            tmp = conv_layer(scale_feats[i])
            if i < len(self.in_channels) - 1:
                tmp = F.interpolate(tmp, size=[scale_feats[-1].shape[2], scale_feats[-1].shape[3]], mode='bilinear', align_corners=True)
            x += tmp
        return self.convfinal(x)
    


def build_dynamic_pe_neck(cfg):
    return DynamicPENeckSOFT(
        in_channels=cfg["in_channels"],
        output_channel=cfg["output_channel"],
        kernel_size=cfg["kernel_size"],
        padding=cfg["padding"],
        stride=cfg["stride"]
    )


if __name__ == "__main__":
    neck = DynamicPENeckSOFT()
    feat = [torch.randn((1,16,448,448)),torch.randn((1,32,224,224)),torch.randn((1,64,112,112)), torch.randn((1,128,56,56)), torch.randn((1,256,28,28)), torch.randn((1,512,14,14))] 
    res = neck(feat)
    print(res.shape)
    print(list(reversed([16, 32, 64, 128, 256, 512])))