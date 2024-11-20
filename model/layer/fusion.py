import torch
from torch import nn
from torch.nn import functional as F
from utils.wdm3d_utils import FUSION_METHOD



class FusionLayer(nn.Module):
    def __init__(self, in_channel, out_channel, fusion_method="plus"):
        """

        :param in_channel:
        :param out_channel:
        :param fusion_method: 表示融合时使用相加操作还是使用相乘操作
        """
        super(FusionLayer, self).__init__()
        self.fusion_method = FUSION_METHOD[fusion_method]
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)

    def forward(self, x, to_fusion):
        h, w = x.shape[2], x.shape[3]
        t = F.interpolate(to_fusion, size=(h, w))
        return F.relu(self.conv(self.fusion_method(x, t)), inplace=False)


class FusionModule(nn.Module):

    def __init__(self, channels=[64, 128, 256, 512], mode="down", fusion_method="plus"):
        """

        :param channels: 输入特征的通道数依次
        :param mode: 表示从上到下(down)融合还是从下到上(up)融合, "上"指靠近原图或通道数少或尺寸大的一端, "下"指远离原图或通道数多或尺寸小的一端, 当mode=="up"时, 等价于FPN
        :param fusion_method: 表示融合时使用相加操作还是使用相乘操作
        """
        super(FusionModule, self).__init__()
        assert fusion_method in FUSION_METHOD
        self.channels = channels
        self.mode = mode
        self.layers = nn.ModuleList()
        channels_tmp = channels if mode == "down" else channels[::-1]
        for i, v in enumerate(channels_tmp):
            self.layers.append(
                nn.Conv2d(v, channels_tmp[i + 1], kernel_size=1) if i == 0 else FusionLayer(v, channels_tmp[
                    i + 1] if i != len(channels) - 1 else v, fusion_method=fusion_method)
            )

    def forward(self, features):
        fused_feats = []

        features = features if self.mode == "down" else features[::-1]

        for i, (x, layer) in enumerate(zip(features, self.layers)):
            if i == 0:
                fx = layer(x)
            else:
                fx = layer(x, fused_feats[-1])
            fused_feats.append(fx)

        return fused_feats


if __name__ == '__main__':
    channels = [64, 128, 256, 512]
    b = 1
    feats = []
    #
    for i, v in enumerate(channels):
        feats.append(torch.randn((b, v, 512 // (2 ** i), 512 // (2 ** i))))

    fusion = FusionModule(channels, mode="up", fusion_method="plus")
    res = fusion(feats)
    # print(fusion.layers)
    print([x.shape for x in res])