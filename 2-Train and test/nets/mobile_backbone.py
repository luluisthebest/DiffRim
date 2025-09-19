import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from nets.ds_conv import ds_basic_conv


class mobile_backbone_stage1(nn.Module):
    def __init__(self, in_channels, time_emb_dim):
        super(mobile_backbone_stage1, self).__init__()
        self.in_channels = in_channels

        # ------------------------------------- 3分支->中间分支 ------------------------------------- #
        self.ds_basic_conv = ds_basic_conv(in_channels=in_channels, out_channels=in_channels, time_emb_dim=time_emb_dim)
        self.batchnorm_center = nn.BatchNorm2d(in_channels)
        # ------------------------------------------------------------------------------------------- #

        # # ------------------------------------- 3分支->左侧分支 ------------------------------------- #            # remove
        self.conv_left = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1,    # remove
                                   padding=0)                                                                     # remove
        self.batchnorm_left = nn.BatchNorm2d(in_channels)                                  
        # # ------------------------------------------------------------------------------------------- #           # remove

        # # ------------------------------------- 3分支->右侧分支 ------------------------------------- #            # remove
        self.batchnorm_right = nn.BatchNorm2d(in_channels)                                                        # remove
        # # ------------------------------------------------------------------------------------------- #    
        out_channels = in_channels                                                                                  # add
        self.shortcut = nn.Identity() if in_channels == out_channels else \
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)                                 # add

        self.activation = nn.ReLU()

    def forward(self, x, scale_shift):
        x_res = x
        x_left = x                                                                                   # remove 
        x_right = x                                                                                  # remove 
        x_center = x

        # --------------------------------- 中间分支 --------------------------------- #
        output_center = self.ds_basic_conv(x_center, scale_shift)
        output_center = self.batchnorm_center(output_center)
        # print("中间分支:", output_center.shape)
        # ---------------------------------------------------------------------------- #

        # # --------------------------------- 左侧分支 --------------------------------- #
        output_left = self.conv_left(x_left)                                                          # remove 
        output_left = self.batchnorm_left(output_left)                                                # remove 
            
        # # print("左侧分支:", output_left.shape)
        # # ---------------------------------------------------------------------------- #

        # # --------------------------------- 右侧分支 --------------------------------- #
        output_right = self.batchnorm_right(x_right)                                                  # remove 
        # # print("右侧分支:", output_right.shape)                                                       # remove 
        # # ---------------------------------------------------------------------------- #
        output = output_center + output_left + output_right                                           # remove 

        # output = output_center + self.batchnorm_left(self.shortcut(x))                                                       # add

        output = self.activation(output)

        return output, x_res


class mobile_backbone_stage2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, time_emb_dim=None):
        super(mobile_backbone_stage2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # ------------------------------------- 2分支->左侧分支 ------------------------------------- #
        self.conv_left = ds_basic_conv(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                       stride=stride, padding=1, time_emb_dim=time_emb_dim)
        self.batchnorm_left = nn.BatchNorm2d(out_channels)
        # ------------------------------------------------------------------------------------------- #

        # ------------------------------------- 2分支->右侧分支 ------------------------------------- #
        # self.conv_right = ds_basic_conv(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
        #                                 stride=stride, padding=1, time_emb_dim=time_emb_dim)
        self.conv_right = ds_basic_conv(in_channels=in_channels, out_channels=out_channels, kernel_size=5,
                                        stride=stride, padding=2, time_emb_dim=time_emb_dim)
        # self.conv_right = ds_basic_conv(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
        #                                 stride=1, padding=0, time_emb_dim=time_emb_dim)
        self.batchnorm_right = nn.BatchNorm2d(out_channels)
        # ------------------------------------------------------------------------------------------- #

        self.activation = nn.ReLU()

    def forward(self, x, scale_shift):
        x_left = x
        x_right = x

        output_left = self.conv_left(x_left,scale_shift)
        output_left = self.batchnorm_left(output_left)
        # print("mobile backbone stage2 left shape:", output_left.shape)

        output_right = self.conv_right(x_right,scale_shift)
        output_right = self.batchnorm_right(output_right)
        # print("mobile backbone stage2 right shape:", output_right.shape)

        output = output_left + output_right

        output = self.activation(output)

        return output


class mobile_backbone_two_stage(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, time_emb_dim=None):
        super(mobile_backbone_two_stage, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_stage1 = mobile_backbone_stage1(in_channels=in_channels,time_emb_dim=time_emb_dim)
        self.conv_stage2 = mobile_backbone_stage2(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, time_emb_dim=time_emb_dim)
        self.long_res = ds_basic_conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding,time_emb_dim=time_emb_dim)

    def forward(self, x, scale_shift):
        output, output_res = self.conv_stage1(x, scale_shift)
        output_long_res = self.long_res(output_res, scale_shift)
        output = self.conv_stage2(output, scale_shift)
        output = output + output_long_res
        return output
