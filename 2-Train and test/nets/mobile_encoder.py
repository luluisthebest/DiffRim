import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torchsummary import summary
from einops import rearrange,reduce

from nets.mobile_backbone import mobile_backbone_two_stage

def exists(x):
    return x is not None

# class eca_block(nn.Module):
#     def __init__(self, channel, H, b=1, gamma=2, s=3):
#         super(eca_block, self).__init__()
#         kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
#         kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
#         self.channel = channel
#         self.H = H
#         self.s = s
#         self.eps = 1e-6

#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()

#         self.to_q = nn.Conv2d(channel, channel, 1)
#         self.to_k = nn.Conv2d(channel, channel, 1)
#         self.to_v = nn.Conv2d(channel, channel, 1)
#         self.pos_bias = nn.Parameter(torch.Tensor(H, H))
#         self.proj = nn.Conv2d(channel, channel, kernel_size=3, padding=1)

#         nn.init.xavier_uniform_(self.pos_bias)

#     def forward(self, x):
#         # temporal/channel attention
#         output = self.avg_pool(x)
#         # print("average pool shape:", output.shape)
#         output = self.conv(output.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
#         output = self.sigmoid(output)

#         # spatio attention
#         for i in range(self.H):
#             for j in range(self.H):
#                 if math.fabs(i-j) > self.s:
#                     self.pos_bias.data[i][j] = 0 
#         # max_pos_bias = self.pos_bias.max()
#         Q = torch.sigmoid(self.to_q(x))
#         K = self.to_k(x)
#         V= self.to_v(x)

#         temp = torch.exp(self.pos_bias) @ torch.mul(torch.exp(K-K.max()), V)
#         weighted = temp / (torch.exp(self.pos_bias) @ torch.exp(K-K.max()) + self.eps)
#         # numerator = torch.log(torch.einsum('ij,bchj->bchi', stable_exp_bias, weighted_V) + self.eps)
#         # denominator = torch.log(torch.einsum('ij,bchj->bchi', stable_exp_bias, exp_K) + self.eps)
#         Yt = torch.mul(weighted, Q)
#         Yt = self.proj(Yt)

#         return Yt * output

class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.avg_pool(x)
        # print("average pool shape:", output.shape)
        output = self.conv(output.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        output = self.sigmoid(output)

        return x * output.expand_as(x)


class mobile_encoder(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super(mobile_encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv_init = nn.Conv2d(in_channels=in_channels, out_channels=in_channels*3, kernel_size=1, padding=0,
                                   stride=1)                                                                             # *3→*2
        # self.conv_init_oneframe = nn.Conv2d(in_channels=in_channels, out_channels=in_channels*9, kernel_size=1, 
        #                                     padding=0, stride=1)
        self.dropout_init = nn.Dropout(p=0.01)

        self.encoder_first = mobile_backbone_two_stage(in_channels=in_channels*9, out_channels=32, kernel_size=3,
                                                       stride=2, padding=1, time_emb_dim=time_emb_dim)                   # *9→*6, 32→16
        self.eca_first = eca_block(channel=32)                                                                     # 32→16
        self.dropout1 = nn.Dropout(p=0.01)
        
        self.encoder_second = mobile_backbone_two_stage(in_channels=32, out_channels=64, kernel_size=3, stride=2,    
                                                        padding=1, time_emb_dim=time_emb_dim)                            # 32→16, 64→32
        self.eca_second = eca_block(channel=64)                                                                    # 64→32
        self.dropout2 = nn.Dropout(p=0.01)

        self.encoder_third = mobile_backbone_two_stage(in_channels=64, out_channels=out_channels, kernel_size=3,
                                                       stride=2, padding=1, time_emb_dim=time_emb_dim)                   # 64→32 
        self.eca_third = eca_block(channel=out_channels)                                                            # 64
        self.dropout3 = nn.Dropout(p=0.01)                                                                               # 

    def forward(self, x, time_emb=None):

        # x_t,x_t_1,x_t_2 = x
        if len(x.shape) == 5:
            x_t = x[0,...]
            x_t_1 = x[1,...]
            x_t_2 = x[2,...]
        else:
            x_t = torch.cat((torch.unsqueeze(x[:,0,:,:], 1), torch.unsqueeze(x[:,3,:,:],1)), dim=1)
            x_t_1 = torch.cat((torch.unsqueeze(x[:,1,:,:], 1), torch.unsqueeze(x[:,4,:,:], 1)), dim=1)
            x_t_2 = torch.cat((torch.unsqueeze(x[:,2,:,:], 1), torch.unsqueeze(x[:,5,:,:], 1)), dim=1)

        x_t = self.conv_init(x_t)
        x_t_1 = self.conv_init(x_t_1)
        x_t_2 = self.conv_init(x_t_2)

        x_total = torch.cat([x_t, x_t_1, x_t_2], dim=1)
        # x_total = self.conv_init_oneframe(x)
        x_total = self.dropout_init(x_total)

        x_total = self.encoder_first(x_total, time_emb)
        x_total = self.eca_first(x_total)
        x_total = self.dropout1(x_total)
        # x_total = self.maxpool_first(x_total)
        x_first_out = x_total
        # print("x_first_out.shape:", x_first_out.shape)
        #x_save = x_total
        #np.save('results/before_encoder.npy', x_save.cpu())
        x_total = self.encoder_second(x_total, time_emb)
        #x_save = x_total
        #np.save('results/after_encoder.npy', x_save.cpu())
        x_total = self.eca_second(x_total)
        x_total = self.dropout2(x_total)
        # x_total = self.maxpool_second(x_total)
        x_second_out = x_total
        # print("x_second_out.shape:", x_second_out.shape)

        x_total = self.encoder_third(x_total, time_emb)                                                                  # remove 
        x_total = self.eca_third(x_total)                                                                                # remove 
        x_total = self.dropout3(x_total)                                                                                 # remove 
        # x_total = self.maxpool_third(x_total)
        # print("x_total.shape:", x_total.shape)

        return x_total, x_first_out, x_second_out
        # return x_first_out, x_second_out                                                                                   # remove 







if __name__ == '__main__':
    input = torch.randn(4, 1, 128, 64), torch.randn(4, 1, 128, 64), torch.randn(4, 1, 128, 64)
    model = mobile_encoder(in_channels=1, out_channels=128)
    output = model(input)
    param_num = sum([param.nelement() for param in model.parameters()])
    print(param_num)


