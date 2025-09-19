import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from nets.ds_deconv import ds_basic_deconv
from nets.ds_conv import ds_basic_conv


class de_mobile_backbone_stage1(nn.Module):
    def __init__(self, in_channels,time_emb_dim):
        super(de_mobile_backbone_stage1, self).__init__()
        self.in_channels = in_channels

        # ------------------------------------- 3分支->中间分支 ------------------------------------- #
        self.ds_basic_deconv = ds_basic_conv(in_channels=in_channels, out_channels=in_channels, time_emb_dim=time_emb_dim)
        self.batchnorm_center = nn.BatchNorm2d(in_channels)
        # ------------------------------------------------------------------------------------------- #

        # ------------------------------------- 3分支->左侧分支 ------------------------------------- #
        self.conv_left = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1,
                                   padding=0)                                                                             # remove
        self.batchnorm_left = nn.BatchNorm2d(in_channels)
        self.shortcut = nn.Identity()
        # ------------------------------------------------------------------------------------------- #

        # ------------------------------------- 3分支->右侧分支 ------------------------------------- #
        self.batchnorm_right = nn.BatchNorm2d(in_channels)                                                                # remove
        # ------------------------------------------------------------------------------------------- #

        self.activation = nn.ReLU()

    def forward(self, x, t):
        x_res = x
        x_left = x                                                                             # remove
        x_right = x                                                                            # remove
        x_center = x

        # --------------------------------- 中间分支 --------------------------------- #
        output_center = self.ds_basic_deconv(x_center, t)
        output_center = self.batchnorm_center(output_center)
        # print("中间分支:", output_center.shape)
        # ---------------------------------------------------------------------------- #

        # # --------------------------------- 左侧分支 --------------------------------- #
        output_left = self.conv_left(x_left)                                                    # remove
        output_left = self.batchnorm_left(output_left)                                          # remove
        # # print("左侧分支:", output_left.shape)                                                  # remove
        # # ---------------------------------------------------------------------------- #

        # # --------------------------------- 右侧分支 --------------------------------- #
        output_right = self.batchnorm_right(x_right)                                            # remove
        # # print("右侧分支:", output_right.shape)                                                 # remove
        # # ---------------------------------------------------------------------------- #
        output = output_center + output_left + output_right                                     # remove
        # output = output_center + self.batchnorm_left(self.shortcut(x))

        output = self.activation(output)

        return output, x_res


class de_mobile_backbone_stage2(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super(de_mobile_backbone_stage2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # ------------------------------------- 2分支->左侧分支 ------------------------------------- #
        # self.conv_left = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2,
        #                                     padding=1)
        self.conv_left = ds_basic_deconv(in_channels=in_channels, out_channels=out_channels, time_emb_dim=time_emb_dim)
        self.batchnorm_left = nn.BatchNorm2d(out_channels)
        # ------------------------------------------------------------------------------------------- #

        # ------------------------------------- 2分支->右侧分支 ------------------------------------- #
        self.conv_right = ds_basic_deconv(in_channels=in_channels, out_channels=out_channels, time_emb_dim=time_emb_dim)
        self.batchnorm_right = nn.BatchNorm2d(out_channels)
        # ------------------------------------------------------------------------------------------- #

        self.activation = nn.ReLU()

    def forward(self, x, t):
        x_left = x
        x_right = x

        output_left = self.conv_left(x_left, t)
        output_left = self.batchnorm_left(output_left)
        # print("deconv 左侧分支:", output_left.shape)

        output_right = self.conv_right(x_right, t)
        output_right = self.batchnorm_right(output_right)
        # print("deconv 右侧分支:", output_right.shape)

        output = output_left + output_right

        output = self.activation(output)

        return output


class de_mobile_backbone_two_stage(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super(de_mobile_backbone_two_stage, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.de_conv_stage1 = de_mobile_backbone_stage1(in_channels=in_channels,time_emb_dim=time_emb_dim)
        self.de_conv_stage2 = de_mobile_backbone_stage2(in_channels=in_channels, out_channels=out_channels,time_emb_dim=time_emb_dim)
        self.long_res = ds_basic_deconv(in_channels=in_channels, out_channels=out_channels, time_emb_dim=time_emb_dim)

    def forward(self, x, t):
        output, output_res = self.de_conv_stage1(x, t)
        output_long_res = self.long_res(output_res, t)
        output = self.de_conv_stage2(output, t)
        output = output + output_long_res
        return output


if __name__ == '__main__':
    input = torch.randn(8, 16, 64, 32)
    model = de_mobile_backbone_two_stage(in_channels=16, out_channels=8)
    print(model(input).shape)