import math
import os
import random
import sys
from collections import namedtuple
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from einops import rearrange, reduce
from ema_pytorch import EMA
from PIL import Image
from torch import einsum, nn
from torch.optim import Adam
from tqdm.auto import tqdm
from nets.de_mobile_backbone import de_mobile_backbone_two_stage
from nets.mobile_encoder import eca_block
from nets.mobile_encoder import mobile_encoder
from utils.dataloader import get_dataloader
from utils.dataloader import get_test_real_dataloader
# from utils.dataloader_1_frame import get_dataloader
# from utils.dataloader_test_sinr import get_dataloader
# from utils.dataloader_COSMOS import get_dataloader 
# from utils.dataloader_rock import get_dataloader
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

ModelResPrediction = namedtuple(
    'ModelResPrediction', ['pred_res', 'pred_noise', 'pred_x_start'])
# helpers functions


def set_seed(SEED):
    # initialize random seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def normalize_to_neg_one_to_one(img):
    if isinstance(img, list):
        return [img[k] * 2 - 1 for k in range(len(img))]
    else:
        return img * 2 - 1


def unnormalize_to_zero_to_one(img):
    if isinstance(img, list):
        return [(img[k] + 1) * 0.5 for k in range(len(img))]
    else:
        return (img + 1) * 0.5

def plot_real_world_test(gt_map, corrupted_map, denoised_rd_map):
    gt_map = gt_map.cpu()
    corrupted_map = corrupted_map.squeeze(0).cpu()
    denoised_rd_map = denoised_rd_map.squeeze(0).cpu()
    fig, ax = plt.subplots(1, 3, figsize=(15, 7))
    ax[0].imshow(gt_map)
    ax[0].set_title("Ground truth")
    ax[1].imshow(corrupted_map)
    ax[1].set_title("Interfered map")
    ax[2].imshow(denoised_rd_map)
    ax[2].set_title("Output")
    plt.savefig("real_world_test.png")
    plt.close(fig)
    

def plot_denoised_rd_maps(i, image_path, folder, sinr_level, gt_map, corrupted_map, denoised_rd_map):
        if image_path is not None:
            image_dir = image_path[i]
            image = Image.open(image_dir)
            image_name = image_dir.split("/")
            results_name = f"{image_name[-2]}_{image_name[-1].split('.')[0]}"
            gt_map = gt_map.squeeze(0).squeeze(0).cpu()        
            corrupted_map = corrupted_map.squeeze(0).squeeze(0).cpu()
            # noised_map = noised_map.cpu()
            denoised_rd_map = denoised_rd_map.squeeze(0).squeeze(0).cpu()

            fig, ax = plt.subplots(1, 4, figsize=(15, 7))
            ax[0].imshow(image)
            ax[0].set_title("Image t")
            ax[1].imshow(gt_map)
            ax[1].set_title("Frame t gt")
            ax[2].imshow(corrupted_map)
            ax[2].set_title("Frame t interference sinr=%d" % sinr_level)
            # ax[3].imshow(noised_map)
            # ax[3].set_title("with noise")
            ax[3].imshow(denoised_rd_map)
            ax[3].set_title("Frame t denoised")

            # plt.imshow(denoised_rd_map)
            plt.savefig(f"{folder}/{results_name}_{i}.png")
            #plt.show()        
            plt.close(fig)
        else:
            corrupted_map = corrupted_map.squeeze(0).squeeze(0).cpu()
            denoised_rd_map = denoised_rd_map.squeeze(0).squeeze(0).cpu()
            fig, ax = plt.subplots(1,2,figsize=(15,7))
            ax[0].imshow(corrupted_map)
            ax[0].set_title("Interference")
            ax[1].imshow(denoised_rd_map)
            ax[1].set_title("Denoised")

            plt.savefig(f"/home/liululu/dataset/realworld_interference/denoised_results/{i:05d}.png")
            plt.close(fig)
  

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(
            half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

# building block modules

class LWAE(nn.Module):
    def __init__(self, dim, encoder_in_channels, encoder_out_channels, decoder_out_channels, channels=1,
                 learned_sinusoidal_cond=False, random_fourier_features=False, out_dim=1,
                 learned_sinusoidal_dim=16, self_condition=False):
        super(LWAE, self).__init__()
        self.encoder_in_channels = encoder_in_channels
        self.encoder_out_channels = encoder_out_channels
        self.decoder_out_channels = decoder_out_channels
        self.self_condition = self_condition
        self.channels = channels
        self.out_dim = out_dim
        
        time_dim = dim * 4
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )                   

        # ----------------------- Encoder ------------------------- #
        self.LWAE_encoder = mobile_encoder(in_channels=encoder_in_channels, out_channels=encoder_out_channels, 
                                               time_emb_dim=time_dim)
        # --------------------------------------------------------- #

        # ----------------------- Decoder ------------------------- #
        self.deconv1 = de_mobile_backbone_two_stage(in_channels=encoder_out_channels, out_channels=64, time_emb_dim=time_dim)   # 64→16
        self.de_eca_first = eca_block(channel=64)                                                                               # 64→16
        self.dropout1 = nn.Dropout(p=0.01)                                                                                    # remove

        self.deconv2 = de_mobile_backbone_two_stage(in_channels=64, out_channels=32, time_emb_dim=time_dim)   # 32→16                 # 64→16， # 16→decoder_out_channels
        self.de_eca_second = eca_block(channel=32)                                                                            # 32→16 remove
        self.dropout2 = nn.Dropout(p=0.01)                                                                                    # remove

        self.deconv3 = de_mobile_backbone_two_stage(in_channels=32, out_channels=decoder_out_channels, time_emb_dim=time_dim)   # remove

        self.activation = nn.ReLU()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d or nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, time, x_self_cond=None):

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)
        
        #r = x.clone()        
        t = self.time_mlp(time)

        encoder_out, encoder_first_out, encoder_second_out = self.LWAE_encoder(x, t)                    

        output = self.deconv1(encoder_out, t)
        output = output + encoder_second_out                                                           
        output = self.de_eca_first(output)
        output = self.dropout1(output)                                                  

        output = self.deconv2(output, t)
        output = output + encoder_first_out                                                                         
        output = self.de_eca_second(output)                                                               
        output = self.dropout2(output)                                                                 

        output = self.deconv3(output, t)                                                   
        output = self.activation(output)

        return output



class DiffRim(nn.Module):
    def __init__(
        self,
        dim,
        encoder_out_channels,
        decoder_out_channels,
        out_dim=1,
        channels=1,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        self_condition=False,
        condition=False,
        num_unet=1,
        input_condition=False,
        objective='pred_res_noise',
        test_res_or_noise="res_noise",
    ):
        super().__init__()
        self.channels = channels
        self.input_condition = input_condition
        self.out_dim = out_dim
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        self.self_condition = self_condition
        self.num_unet = num_unet
        self.objective = objective
        self.test_res_or_noise = test_res_or_noise

        input_channels = channels + channels*(1 if condition else 0)
        # determine dimensions
        
        if self.num_unet == 2:
            self.LWAE0 = LWAE(dim,input_channels,encoder_out_channels,decoder_out_channels,
                              learned_sinusoidal_cond=learned_sinusoidal_cond,
                              random_fourier_features=random_fourier_features,
                              learned_sinusoidal_dim=learned_sinusoidal_dim,
                              condition=condition, input_condition=input_condition)
            self.LWAE1 = LWAE(dim,input_channels,encoder_out_channels,decoder_out_channels,
                              learned_sinusoidal_cond=learned_sinusoidal_cond,
                              random_fourier_features=random_fourier_features,
                              learned_sinusoidal_dim=learned_sinusoidal_dim,
                              condition=condition, input_condition=input_condition)
            self.LWAE0._initialize_weights()
            self.LWAE1._initialize_weights()
        elif self.num_unet == 1:
            self.LWAE0 = LWAE(dim,input_channels,encoder_out_channels,decoder_out_channels,
                              learned_sinusoidal_cond=learned_sinusoidal_cond,
                              random_fourier_features=random_fourier_features,
                              learned_sinusoidal_dim=learned_sinusoidal_dim)
            self.LWAE0._initialize_weights()

    def forward(self, x, time, x_self_cond=None):
        
        if self.num_unet == 2:
            return self.LWAE0(x, time, x_self_cond=x_self_cond), self.LWAE1(x, time, x_self_cond=x_self_cond)
        elif self.num_unet == 1:
            if self.objective == "pred_noise":
                time = time[1]
            elif self.objective == "pred_res" or self.objective == "auto_res_noise":
                pass
            return [self.LWAE0(x, time, x_self_cond=x_self_cond)]

# gaussian diffusion trainer class


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    #print(out.min(), out.max())
    out = out.reshape(b, *((1,) * (len(x_shape) - 1)))
    return out


def gen_coefficients(timesteps, schedule="increased", sum_scale=1):
    if schedule == "increased":
        x = torch.linspace(1, timesteps, timesteps, dtype=torch.float64)
        scale = 0.5*timesteps*(timesteps+1)
        alphas = x/scale
    elif schedule == "decreased":
        x = torch.linspace(1, timesteps, timesteps, dtype=torch.float64)
        x = torch.flip(x, dims=[0])
        scale = 0.5*timesteps*(timesteps+1)
        alphas = x/scale
    elif schedule == "average":
        alphas = torch.full([timesteps], 1/timesteps, dtype=torch.float64)
    else:
        alphas = torch.full([timesteps], 1/timesteps, dtype=torch.float64)
    assert alphas.sum()-torch.tensor(1) < torch.tensor(1e-10)

    return alphas*sum_scale


class ResidualDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        sampling_timesteps=None,
        loss_type='l1',
        objective='pred_res_noise',
        ddim_sampling_eta=0.,
        condition=False,
        sum_scale=None,
        input_condition=False,
        input_condition_mask=False,
        test_res_or_noise=None,
        alpha_res_to_0_or_1=None
    ):
        super().__init__()
        assert not (
            type(self) == ResidualDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.image_size = image_size
        self.objective = objective
        self.condition = condition
        self.input_condition = input_condition
        self.input_condition_mask = input_condition_mask
        self.test_res_or_noise = test_res_or_noise

        if self.condition:
            self.sum_scale = sum_scale if sum_scale else 0.01
            ddim_sampling_eta = 0.
        else:
            self.sum_scale = sum_scale if sum_scale else 1.
         
        convert_to_ddim = True
        if convert_to_ddim:
            beta_schedule = "linear"
            beta_start = 0.0001
            beta_end = 0.02
            if beta_schedule == "linear":
                betas = torch.linspace(
                    beta_start, beta_end, timesteps, dtype=torch.float32)
            elif beta_schedule == "scaled_linear":
                # this schedule is very specific to the latent diffusion model.
                betas = (
                    torch.linspace(beta_start**0.5, beta_end**0.5,
                                timesteps, dtype=torch.float32) ** 2
                )
            elif beta_schedule == "squaredcos_cap_v2":
                # Glide cosine schedule
                betas = betas_for_alpha_bar(timesteps)
            else:
                raise NotImplementedError(
                    f"{beta_schedule} does is not implemented for {self.__class__}")

            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            alphas_cumsum = 1-alphas_cumprod ** 0.5
            betas2_cumsum = 1-alphas_cumprod

            alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
            betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)
            alphas = alphas_cumsum-alphas_cumsum_prev
            alphas[0] = 0
            betas2 = betas2_cumsum-betas2_cumsum_prev
            betas2[0] = 0
        else:
            alphas = gen_coefficients(timesteps, schedule="decreased")
            betas2 = gen_coefficients(timesteps, schedule="increased", sum_scale=self.sum_scale)
            alphas_cumsum = alphas.cumsum(dim=0).clip(0, 1)
            betas2_cumsum = betas2.cumsum(dim=0).clip(0, 1)
            alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
            betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)            
        
        betas_cumsum = torch.sqrt(betas2_cumsum)
        
        posterior_variance = betas2*betas2_cumsum_prev/betas2_cumsum
        posterior_variance[0] = 0

        timesteps, = alphas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters
        # default num sampling timesteps to number of timesteps at training
        self.sampling_timesteps = default(sampling_timesteps, timesteps)

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))

        register_buffer('alphas', alphas)
        register_buffer('alphas_cumsum', alphas_cumsum)
        register_buffer('one_minus_alphas_cumsum', 1-alphas_cumsum)
        register_buffer('betas2', betas2)
        register_buffer('betas', torch.sqrt(betas2))
        register_buffer('betas2_cumsum', betas2_cumsum)
        register_buffer('betas_cumsum', betas_cumsum)
        register_buffer('posterior_mean_coef1',
                        betas2_cumsum_prev/betas2_cumsum)
        register_buffer('posterior_mean_coef2', (betas2 *
                        alphas_cumsum_prev-betas2_cumsum_prev*alphas)/betas2_cumsum)
        register_buffer('posterior_mean_coef3', betas2/betas2_cumsum)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))

        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2[0] = 0
        self.posterior_mean_coef3[0] = 1
        self.one_minus_alphas_cumsum[-1] = 1e-6

        if objective == "auto_res_noise" or objective == "auto_intf_noise":
            if alpha_res_to_0_or_1:
                self.alpha_res_to_0_or_1 = alpha_res_to_0_or_1
                self.alpha_res = torch.nn.Parameter(alpha_res_to_0_or_1*torch.ones((1)), requires_grad=True)
            else:
                self.alpha_res = torch.nn.Parameter(0.5*torch.ones((1)), requires_grad=True)
                self.alpha_res_to_0_or_1 = None

    def init(self):
        timesteps = 1000

        convert_to_ddim = True
        if convert_to_ddim:
            beta_schedule = "linear"
            beta_start = 0.0001
            beta_end = 0.02
            if beta_schedule == "linear":
                betas = torch.linspace(
                    beta_start, beta_end, timesteps, dtype=torch.float32)
            elif beta_schedule == "scaled_linear":
                # this schedule is very specific to the latent diffusion model.
                betas = (
                    torch.linspace(beta_start**0.5, beta_end**0.5,
                                   timesteps, dtype=torch.float32) ** 2
                )
            elif beta_schedule == "squaredcos_cap_v2":
                # Glide cosine schedule
                betas = betas_for_alpha_bar(timesteps)
            else:
                raise NotImplementedError(
                    f"{beta_schedule} does is not implemented for {self.__class__}")

            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            alphas_cumsum = 1-alphas_cumprod ** 0.5
            betas2_cumsum = 1-alphas_cumprod

            alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
            betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)
            alphas = alphas_cumsum-alphas_cumsum_prev
            alphas[0] = alphas[1]
            betas2 = betas2_cumsum-betas2_cumsum_prev
            betas2[0] = betas2[1]
        else:
            alphas = gen_coefficients(timesteps, schedule="average", ratio=1)
            betas2 = gen_coefficients(
                timesteps, schedule="increased", sum_scale=self.sum_scale, ratio=3)

            alphas_cumsum = alphas.cumsum(dim=0).clip(0, 1)
            betas2_cumsum = betas2.cumsum(dim=0).clip(0, 1)

            alphas_cumsum_prev = F.pad(
                alphas_cumsum[:-1], (1, 0), value=alphas_cumsum[1])
            betas2_cumsum_prev = F.pad(
                betas2_cumsum[:-1], (1, 0), value=betas2_cumsum[1])

        betas_cumsum = torch.sqrt(betas2_cumsum)
        posterior_variance = betas2*betas2_cumsum_prev/betas2_cumsum
        posterior_variance[0] = 0

        timesteps, = alphas.shape
        self.num_timesteps = int(timesteps)

        self.alphas = alphas
        self.alphas_cumsum = alphas_cumsum
        self.one_minus_alphas_cumsum = 1-alphas_cumsum
        self.betas2 = betas2
        self.betas = torch.sqrt(betas2)
        self.betas2_cumsum = betas2_cumsum
        self.betas_cumsum = betas_cumsum
        self.posterior_mean_coef1 = betas2_cumsum_prev/betas2_cumsum
        self.posterior_mean_coef2 = (
            betas2 * alphas_cumsum_prev-betas2_cumsum_prev*alphas)/betas2_cumsum
        self.posterior_mean_coef3 = betas2/betas2_cumsum
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = torch.log(
            posterior_variance.clamp(min=1e-20))

        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2[0] = 0
        self.posterior_mean_coef3[0] = 1
        self.one_minus_alphas_cumsum[-1] = 1e-6

    def predict_noise_from_res(self, x_t, t, x_input, pred_res):
        return (
            (x_t[0,]-x_input[0,]-(extract(self.alphas_cumsum, t, x_t.shape[1:])-1)
             * pred_res)/extract(self.betas_cumsum, t, x_t.shape[1:])
        )

    def predict_start_from_xinput_noise(self, x_t, t, x_input, noise):      
        return (
            (x_t[0,]-extract(self.alphas_cumsum, t, x_t.shape[1:])*x_input[0,] -
             extract(self.betas_cumsum, t, x_t.shape[1:]) * noise)/extract(self.one_minus_alphas_cumsum, t, x_t.shape[1:])
        )

    def predict_start_from_res_noise(self, x_t, t, x_res, noise):
        return (
            x_t[0,] - extract(self.alphas_cumsum, t, x_t[0,].shape) * x_res -
            extract(self.betas_cumsum, t, x_t[0,].shape) * noise
        )

    def q_posterior_from_res_noise(self, x_res, noise, x_t, t):
        return (x_t-extract(self.alphas, t, x_t.shape) * x_res -
                (extract(self.betas2, t, x_t.shape)/extract(self.betas_cumsum, t, x_t.shape)) * noise)

    def q_posterior(self, pred_res, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_t +
            extract(self.posterior_mean_coef2, t, x_t.shape) * pred_res +
            extract(self.posterior_mean_coef3, t, x_t.shape) * x_start
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x_input, x, t, x_input_condition=0, x_self_cond=None, clip_denoised=True, log_file=None):
        if not self.condition:
            x_in = x
        else:
            if self.input_condition:
                x_in = torch.cat((x, x_input, x_input_condition), dim=1)
            else:
                x_in = torch.cat((x, x_input), dim=2)
        model_output = self.model(x_in,
                                  t,
                                  x_self_cond)
        # out_save = model_output[0]
        #np.save('results/model_out_25187.npy', out_save.cpu())
        maybe_clip = partial(torch.clamp, min=-1.,
                             max=1.) if clip_denoised else identity

        if self.objective == 'pred_res_noise':
            pred_res = model_output[0]
            pred_noise = model_output[1]
            pred_res = maybe_clip(pred_res)
            x_start = self.predict_start_from_res_noise(
                x, t, pred_res, pred_noise)
            x_start = maybe_clip(x_start)
        elif self.objective == 'pred_res_add_noise':
            pred_res = model_output[0]
            pred_noise = model_output[1] - model_output[0]
            pred_res = maybe_clip(pred_res)
            x_start = self.predict_start_from_res_noise(
                x, t, pred_res, pred_noise)
            x_start = maybe_clip(x_start)
        elif self.objective == 'pred_x0_noise':
            pred_res = x_input-model_output[0]
            pred_noise = model_output[1]
            pred_res = maybe_clip(pred_res)
            x_start = maybe_clip(model_output[0])
        elif self.objective == 'pred_x0_add_noise':
            x_start = model_output[0]
            pred_noise = model_output[1] - model_output[0]
            pred_res = x_input-x_start
            pred_res = maybe_clip(pred_res)
            x_start = maybe_clip(model_output[0])
        elif self.objective == "pred_noise":
            pred_noise = model_output[0]
            x_start = self.predict_start_from_xinput_noise(
                x, t, x_input, pred_noise)
            x_start = maybe_clip(x_start)
            pred_res = x_input - x_start
            pred_res = maybe_clip(pred_res)
        elif self.objective == "pred_res":
            pred_res = model_output[0]
            pred_res = maybe_clip(pred_res)
            pred_noise = self.predict_noise_from_res(x, t, x_input, pred_res)
            x_start = x_input - pred_res
            x_start = maybe_clip(x_start)
        elif self.objective == "auto_res_noise" or self.objective == "auto_intf_noise":
            # print(self.alpha_res)
            if log_file:
                log_file.write(f'alpha_res: {self.alpha_res.item():.8f}\n')
            if self.alpha_res > 0.5:
                pred_res = model_output[0]
                pred_res = maybe_clip(pred_res)
                pred_noise = self.predict_noise_from_res(x, t, x_input, pred_res)
                x_start = x_input[0,] - pred_res
                x_start = maybe_clip(x_start)
            else:
                pred_noise = model_output[0]
                x_start = self.predict_start_from_xinput_noise(
                    x, t, x_input, pred_noise)
                x_start = maybe_clip(x_start)
                pred_res = x_input[0,] - x_start
                pred_res = maybe_clip(pred_res)

        return ModelResPrediction(pred_res, pred_noise, x_start)

    def p_mean_variance(self, x_input, x, t, x_input_condition=0, x_self_cond=None, log_file=None):
        preds = self.model_predictions(
            x_input, x, t, x_input_condition, x_self_cond, log_file=log_file)
        pred_res = preds.pred_res
        x_start = preds.pred_x_start

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            pred_res=pred_res, x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x_input, x, t: int, x_input_condition=0, x_self_cond=None, log_file=None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full(
            (x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x_input, x=x, t=batched_times, x_input_condition=x_input_condition, x_self_cond=x_self_cond, log_file=log_file)
        noise = 0.01*torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, x_input, shape, last=True, log_file=None):
        if self.input_condition:
            x_input_condition = x_input[1]
        else:
            x_input_condition = 0
        x_input = x_input[0]

        batch, device = shape[0], self.betas.device

        if self.condition:
            img = x_input+math.sqrt(self.sum_scale) * \
                0.01*torch.randn(shape, device=device)
            input_add_noise = img
        else:
            img = 0.01*torch.randn(shape, device=device)

        x_start = None

        if not last:
            img_list = []

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(
                x_input, img, t, x_input_condition, self_cond, log_file)

            if not last:
                img_list.append(img)

        if self.condition:
            if not last:
                img_list = [input_add_noise]+img_list
            else:
                img_list = [input_add_noise, img]
            return unnormalize_to_zero_to_one(img_list)
        else:
            if not last:
                img_list = img_list
            else:
                img_list = [img]
            return unnormalize_to_zero_to_one(img_list)

    @torch.no_grad()
    def ddim_sample(self, x_input, shape, last=True, log_file=None):
        if self.input_condition:
            x_input_condition = x_input[1]
        else:
            x_input_condition = 0

        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
            1], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs = list(zip(times[:-1], times[1:]))

        if self.condition:
            img = x_input+math.sqrt(self.sum_scale) * \
                torch.randn(shape, device=device)
            input_add_noise = img
        else:
            img = torch.randn(shape, device=device)

        x_start = None
        type = "use_pred_noise"

        if not last:
            img_list = []

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full(
                (batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            preds = self.model_predictions(
                x_input, img, time_cond, x_input_condition, self_cond, log_file=log_file)

            pred_res = preds.pred_res
            pred_noise = preds.pred_noise
            x_start = preds.pred_x_start

            if time_next < 0:
                img = x_start
                if not last:
                    img_list.append(img)
                continue

            alpha_cumsum = self.alphas_cumsum[time]
            alpha_cumsum_next = self.alphas_cumsum[time_next]
            alpha = alpha_cumsum-alpha_cumsum_next

            betas2_cumsum = self.betas2_cumsum[time]
            betas2_cumsum_next = self.betas2_cumsum[time_next]
            betas2 = betas2_cumsum-betas2_cumsum_next
            betas = betas2.sqrt()
            betas_cumsum = self.betas_cumsum[time]
            betas_cumsum_next = self.betas_cumsum[time_next]
            sigma2 = eta * (betas2*betas2_cumsum_next/betas2_cumsum)
            sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum = (
                betas2_cumsum_next-sigma2).sqrt()/betas_cumsum

            if eta == 0:
                noise = 0
            else:
                noise = torch.randn_like(img)

            if type == "use_pred_noise":
                img[0,] = img[0,] - alpha*pred_res - \
                    (betas_cumsum-(betas2_cumsum_next-sigma2).sqrt()) * \
                    pred_noise + sigma2.sqrt()*noise
            elif type == "use_x_start":
                img = sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum*img + \
                    (1-sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum)*x_start + \
                    (alpha_cumsum_next-alpha_cumsum*sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum)*pred_res + \
                    sigma2.sqrt()*noise
            elif type == "special_eta_0":
                img = img - alpha*pred_res - \
                    (betas_cumsum-betas_cumsum_next)*pred_noise
            elif type == "special_eta_1":
                img = img - alpha*pred_res - betas2/betas_cumsum*pred_noise + \
                    betas*betas2_cumsum_next.sqrt()/betas_cumsum*noise

            if not last:
                img_list.append(img)

        if self.condition:
            if not last:
                img_list = [input_add_noise]+img_list
            else:
                img_list = [input_add_noise, img] # img is predicted start
            return unnormalize_to_zero_to_one(img_list)
      
        else:
            if not last:
                img_list = img_list
            else:
                img_list = [img]
            return unnormalize_to_zero_to_one(img_list)

    @torch.no_grad()
    def sample(self, x_input=0, batch_size=16, last=True, log_file=None):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        if self.condition:
            if self.input_condition and self.input_condition_mask:
                x_input[0] = normalize_to_neg_one_to_one(x_input[0])
            else:
                x_input = normalize_to_neg_one_to_one(x_input)
                pass
            temp, batch_size, channels, h, w = x_input.shape
            size = (temp, batch_size, channels, h, w)
        else:
            size = (batch_size, channels, image_size, image_size)
        return sample_fn(x_input, size, last=last, log_file=log_file)

    def q_sample(self, x_start, x_input, x_res, t, shape, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_input))

        alpha = extract(self.alphas_cumsum, t, shape)
        beta = extract(self.betas_cumsum, t, shape)
        c,h,w = x_input.shape[2:]
        return (
            x_start.reshape(-1, c, h, w)+ \
                alpha * x_res.reshape(-1, c, h, w) + beta * noise.reshape(-1, c, h, w)
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, imgs, t, log_file=None, noise=None):
        if isinstance(imgs, list):  # Condition
            if self.input_condition:
                x_input_condition = imgs[2]
            else:
                x_input_condition = 0
            x_input = imgs[0]
            x_start = imgs[1]  # gt = imgs[1], input = imgs[0]
        else:  # Generation
            x_input = 0
            x_start = imgs

        #noise = default(noise, lambda: torch.randn_like(x_input))
        noise = torch.randn_like(x_input)
        x_res = x_input - x_start

        temp, b, c, h, w = x_input.shape

        # noise sample
        x = self.q_sample(x_start, x_input, x_res, t, x_start.shape[1:], noise=noise).reshape(temp,b,c,h,w)
        
        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly
        x_self_cond = None
        if self.self_condition and random.random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(
                    x_input, x, t, x_input_condition if self.input_condition else 0, log_file=log_file).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step
        if not self.condition:
            x_in = x
        else:
            if self.input_condition:
                x_in = torch.cat((x, x_input, x_input_condition), dim=1)
            else:
                x_in = torch.cat((x, x_input), dim=2)

        model_out = self.model(x_in,
                               t.reshape(b,temp)[:,0],
                               x_self_cond)

        target = []
        if self.objective == 'pred_res_noise':
            target.append(x_res[0].squeeze(0))
            target.append(noise[0].squeeze(0))

            pred_res = model_out[0]
            pred_noise = model_out[1]
        elif self.objective == 'pred_res_add_noise':
            target.append(x_res)
            target.append(x_res+noise)

            pred_res = model_out[0]
            pred_noise = model_out[1]-model_out[0]
        elif self.objective == 'pred_x0_noise':
            target.append(x_start)
            target.append(noise)

            pred_res = x_input-model_out[0]
            pred_noise = model_out[1]
        elif self.objective == 'pred_x0_add_noise':
            target.append(x_start)
            target.append(x_start+noise)

            pred_res = x_input-model_out[0]
            pred_noise = model_out[1] - model_out[0]
        elif self.objective == "pred_noise":
            target.append(noise)
            pred_noise = model_out[0]

        elif self.objective == "pred_res":
            target.append(x_res[0,...])

            pred_res = model_out[0]
        elif self.objective == 'auto_res_noise':
            loss_list = []
            clip_denoised = True
            maybe_clip = partial(torch.clamp, min=-1.,
                                max=1.) if clip_denoised else identity
            
            if self.alpha_res_to_0_or_1 == None:
                alpha_res_value = self.alpha_res.item()
            else:
                alpha_res_value = self.alpha_res_to_0_or_1
            #print(alpha_res_value)
            if log_file:
                log_file.write(f'alpha_res_value: {alpha_res_value:.8f}\n')
            sys.stdout.flush()
            if np.abs(alpha_res_value - 0.5)<1e-2:
                target.append(x_res)
                target.append(noise)

                # if is res
                x_start = self.predict_start_from_xinput_noise(x, t.reshape(temp,b)[0,], x_input, model_out[0])
                x_start = maybe_clip(x_start)
                #print(x_start.mean().item(), x_start.std().item())
                pred_res = x_input[0,] - x_start # from noise
                #print(model_out[0].shape)
                pred_res = self.alpha_res*model_out[0] + (1-self.alpha_res)*pred_res

                # if is noise
                pred_noise = self.predict_noise_from_res(x, t.reshape(temp,b)[0,], x_input, model_out[0]) # from res
                pred_noise = self.alpha_res*pred_noise+(1-self.alpha_res)*model_out[0]
                #print(pred_res.mean().item(), pred_res.std().item(), target[0][0,].mean().item(), target[0][0,].std().item())
                loss = self.loss_fn(pred_res, target[0][0,], reduction='none')
                loss = reduce(loss, 'b ... -> b (...)', 'mean').mean()
                loss_list.append(loss)

                loss = self.loss_fn(pred_noise, target[1][0,], reduction='none')
                loss = reduce(loss, 'b ... -> b (...)', 'mean').mean()
                loss_list.append(loss)
                #print("res+noise")
                if log_file:
                    log_file.write(f'res+noise\n')
            elif alpha_res_value>0.5:
                target.append(x_res)

                pred_res = model_out[0]

                loss = self.loss_fn(model_out[0], target[0][0,], reduction='none')
                loss = reduce(loss, 'b ... -> b (...)', 'mean').mean()
                loss_list.append(loss)

                loss_list.append(0)
                #print("res")
                if log_file:
                    log_file.write(f'res\n')
                self.alpha_res_to_0_or_1 = 1
            elif alpha_res_value<0.5:
                target.append(noise)

                pred_noise = model_out[0]

                loss_list.append(0)

                loss = self.loss_fn(model_out[0], target[0][0,], reduction='none')
                loss = reduce(loss, 'b ... -> b (...)', 'mean').mean()
                loss_list.append(loss)
                #print("noise")
                if log_file:
                    log_file.write(f'noise\n')
                self.alpha_res_to_0_or_1 = 0
            sys.stdout.flush()
            return loss_list
        
        else:
            raise ValueError(f'unknown objective {self.objective}')
            
        u_loss = False
        if u_loss:
            x_u = self.q_posterior_from_res_noise(pred_res, pred_noise, x, t)
            u_gt = self.q_posterior_from_res_noise(x_res, noise, x, t)
            loss = 10000*self.loss_fn(x_u, u_gt, reduction='none')
            return loss
        else:
            loss_list = []
            for i in range(len(model_out)):
                assert model_out[i].shape == target[i].shape
                loss = self.loss_fn(model_out[i], target[i], reduction='none')
                loss = reduce(loss, 'b ... -> b (...)', 'mean')
                loss_list.append(loss.mean())
            return loss_list
        
    def forward(self, img, log_file, *args, **kwargs):
        if isinstance(img, list):
            temp, b, c, h, w, device, img_size, = *img[0].shape, img[0].device, self.image_size
        else:
            temp, b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (temp * b,), device=device).long()

        if self.input_condition and self.input_condition_mask:
            img[0] = normalize_to_neg_one_to_one(img[0])
            img[1] = normalize_to_neg_one_to_one(img[1])
        else:
            img = normalize_to_neg_one_to_one(img)
            pass

        return self.p_losses(img, t, log_file, *args, **kwargs)

# trainer class


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        original_ddim_ddpm,
        data_path,
        *,
        train_ratio = 0.92,
        train_batch_size=16,
        gradient_accumulate_every=1,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=1000,
        num_samples=25,
        results_folder='',
        amp=False,
        fp16=False,
        split_batches=True,
        condition=False,
        sub_dir=False,
        crop_patch=False,
        data=None,
        num_unet=1,
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no'
        )
        self.sub_dir = sub_dir
        self.crop_patch = crop_patch
        self.original_ddim_ddpm = original_ddim_ddpm
        self.accelerator.native_amp = amp
        self.results_folder = results_folder

        self.model = diffusion_model

        assert has_int_squareroot(
            num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size
        self.condition = condition
        self.num_unet = num_unet
        if data == 'rock':
            self.min_data = 0.0
            self.max_data = 1.0
        elif data == 'cosmos':
            self.min_data = 0.0
            self.max_data = 1.0
        else:
            self.min_data = -0.9596880733445612
            self.max_data = 7.157625200063627

        self.condition_type = 2         

        if data == 'realworld':
            testloader, validloader, trainloader = get_test_real_dataloader(data_path)
        else:
            trainloader, testloader, validloader, self.image_path = get_dataloader(data_path, batch_size=train_batch_size)
        
        if validloader is not None:
            self.sample_loader = cycle(self.accelerator.prepare(validloader))  # cpu_count()
        else:
            self.sample_loader = cycle(self.accelerator.prepare(testloader))
        
        if trainloader is not None:
            self.dl = cycle(self.accelerator.prepare(trainloader))
        self.test_loader = testloader
        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay,
                           update_every=ema_update_every)

            self.set_results_folder(results_folder)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        device = self.accelerator.device
        self.device = device


    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        path = Path(self.results_folder / f'model-{milestone}.pt')

        if path.exists():
            data = torch.load(
                str(path), map_location=self.device)

            model = self.accelerator.unwrap_model(self.model)
            model.load_state_dict(data['model'])

            self.step = data['step']
            self.opt.load_state_dict(data['opt'])
            self.ema.load_state_dict(data['ema'])

            if exists(self.accelerator.scaler) and exists(data['scaler']):
                self.accelerator.scaler.load_state_dict(data['scaler'])

            print("load model - "+str(path))

        self.ema.to(self.device)

    def train(self, ADPT=False):
        accelerator = self.accelerator
        current_path = __file__
        parent_path = os.path.dirname(current_path)
        grand_parent_path = os.path.dirname(parent_path)
        log_path = os.path.join(grand_parent_path, self.results_folder, 'results.log')
        writer = SummaryWriter(grand_parent_path / self.results_folder)
        
        with open(log_path, 'a') as log_file:
            with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

                while self.step < self.train_num_steps:

                    total_loss = [0.]
                    for _ in range(self.gradient_accumulate_every):
                        data = next(self.dl)
                        data = [item.to(self.device) for item in data[:2]]
                        data = [(item-self.min_data) / (self.max_data-self.min_data) for item in data]

                        loss = self.model(data,log_file)
                        if self.model.objective == "auto_res_noise" or self.model.objective == "auto_intf_noise":
                            if self.model.alpha_res_to_0_or_1 == None:
                                alpha_res_value=self.model.alpha_res
                            else:
                                alpha_res_value=self.model.alpha_res_to_0_or_1  # for auto select res or noise
                                # pass
                            loss[0] = alpha_res_value*loss[0] / self.gradient_accumulate_every
                            loss[1] = (1-alpha_res_value)*loss[1] / self.gradient_accumulate_every
                            loss[0] = loss[0] + loss[1]
                            total_loss[0] = loss[0].item()
                        else:
                            for i in range(self.num_unet):
                                loss[i] = loss[i] / self.gradient_accumulate_every
                                total_loss[i] = total_loss[i] + loss[i].item()
                    
                        for i in range(self.num_unet):
                            self.accelerator.backward(loss[i])
                    if ADPT and (self.model.alpha_res_to_0_or_1==0 or self.model.alpha_res_to_0_or_1==1):
                        return self.model.alpha_res_to_0_or_1
                    writer.add_scalar('Loss/train', total_loss[0], self.step)
                    accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                    accelerator.wait_for_everyone()

                    self.opt.step()
                    self.opt.zero_grad()

                    accelerator.wait_for_everyone()

                    self.step += 1
                    if accelerator.is_main_process:
                        self.ema.to(self.device)
                        self.ema.update()

                        if self.step != 0 and self.step % self.save_and_sample_every == 0:
                            milestone = self.step // self.save_and_sample_every
                            _, gt_val, _, out_val = self.sample(milestone,log_file=log_file)
                            loss_val = self.model.loss_fn(gt_val, out_val[1][0])
                            
                            writer.add_scalar('Loss/val', loss_val, global_step=self.step)
                            self.save(milestone)
                    if log_file:
                        log_file.write(f'step: {self.step}, loss: {total_loss[0]:.8f}\n')
                    pbar.set_description(f'loss: {total_loss[0]:.8f}')
                    pbar.update(1)
            
        accelerator.print('training complete')
        # return milestone

    def sample(self, milestone, items=None, last=True, FID=False, ddpm_test=False, log_file=None):
        self.ema.ema_model.eval()

        with torch.no_grad():
            batches = self.num_samples
            if self.condition_type == 0:
                x_input_sample = [0]
                show_x_input_sample = []
            elif self.condition_type == 1:
                x_input_sample = [next(self.sample_loader).to(self.device)]
                show_x_input_sample = x_input_sample
            elif self.condition_type == 2:
                if not ddpm_test:
                    items = next(self.sample_loader)
                 
                x_input_sample = [item.to(self.device)
                                  for item in items[:2]]
                x_input_sample = [(item-self.min_data) / (self.max_data-self.min_data) for item in x_input_sample]
                
                gt = x_input_sample[1][0,]
                x_input_sample = x_input_sample[0]  #[1] for ground truth
            elif self.condition_type == 3:
                x_input_sample = next(self.sample_loader)
                x_input_sample = [item.to(self.device)
                                  for item in x_input_sample]
                
                x_input_sample = x_input_sample[0]

            output = self.ema.ema_model.sample(
                    x_input_sample, batch_size=batches, last=last, log_file=log_file) # output[0] is noise_add_img, output[1] is predicted start
            if not self.original_ddim_ddpm:
                output[0] = output[0][0,]
            
        return milestone, gt, x_input_sample[0,], output # x_input_sample[0] is interfered img without noise

    def test(self, model, data, only_test=False, sample=False, timesteps=1, last=True, FID=False):
        # if self.model.objective == 'auto_res_noise':
        self.ema.ema_model.init()
        self.ema.to(self.device)
        print("test start")
        if self.condition:
            self.ema.ema_model.eval()
            loader = self.test_loader 
            i = 0
            final_results = []
            gts = []
            mapts = []
            add_noise = []
            #================sinr level test==================#
            if data == 'synt':
                data_by_level = {i: {'gt': [], 'map': [], 'result': []} for i in range(7)}
                        
            with tqdm(initial=i, total=len(loader)) as pbar:
                for items in loader:
                    #file_name = f'{i}.png'
                    
                    with torch.no_grad():
                        batches = self.num_samples

                        if self.condition_type == 0:
                            x_input_sample = [0]
                            show_x_input_sample = []
                        elif self.condition_type == 1:
                            x_input_sample = items.to(self.device)
                            show_x_input_sample = x_input_sample
                        elif self.condition_type == 2:
                            x_input_sample = [item.to(self.device)
                                            for item in items[:2]]
                            x_input_sample = [(item-self.min_data) /(self.max_data-self.min_data) for item in x_input_sample]
                            show_x_input_sample = x_input_sample[0][0,], x_input_sample[1][0,]
                            x_input_sample = x_input_sample[0]
                            gts.append(show_x_input_sample[1].squeeze(0))
                            mapts.append(show_x_input_sample[0].squeeze(0))                        
                        elif self.condition_type == 3:
                            x_input_sample = [item.to(self.device)
                                            for item in items]
                            show_x_input_sample = x_input_sample
                            x_input_sample = x_input_sample[1:]

                        if sample:
                            all_images_list = list(show_x_input_sample) + \
                                list(self.ema.ema_model.sample(
                                    x_input_sample, batch_size=batches))
                        else:
                            all_images_list = list(self.ema.ema_model.sample(
                                x_input_sample, batch_size=batches, last=last))
                            add_noise.append(all_images_list[0][0,].squeeze(0))
                            
                                         
                    # train main()
                    if len(all_images_list[1].shape) == 4:
                        final_results.append(all_images_list[1][0,])
                    elif len(all_images_list[1].shape) == 5:
                        final_results.append(all_images_list[1][0,].squeeze(0))                       

                    level_value = items[2][0]                                                # for 7 level synthetic dataset
                    plot_denoised_rd_maps(i,None,self.results_folder,items[2][0],show_x_input_sample[1],
                                          show_x_input_sample[0],all_images_list[1][0,])

                    if 0 <= level_value <= 6:
                        data_by_level[level_value]['gt'].append(show_x_input_sample[1].squeeze(0))     # for 7 level synthetic dataset
                        data_by_level[level_value]['map'].append(show_x_input_sample[0].squeeze(0))    # for 7 level synthetic dataset
                        data_by_level[level_value]['result'].append(all_images_list[1][0,].squeeze(0)) # for 7 level synthetic dataset
                  

                    i += 1
                    # if i > 70: break   # for debug
            # final_results = np.array(torch.cat(final_results, dim=0).cpu())      # only for real-world data
            # final_results = final_results * (121.01468+8.503986) + (-8.503986)   # only for real-world data
            # np.save(os.path.join(self.results_folder, 'realworld_denoised.npy'), final_results)   # only for real-world data

            #==============test sinr level=============#
            if data == 'synt':
                for i, data_dict in data_by_level.items():
                    gt_array = np.array(torch.cat(data_dict['gt'], dim=0).cpu())
                    map_array = np.array(torch.cat(data_dict['map'], dim=0).cpu())
                    result_array = np.array(torch.cat(data_dict['result'], dim=0).cpu())
            
                    np.save(os.path.join(self.results_folder, f"result_{i}.npy"), result_array)
                    np.save(os.path.join(self.results_folder, f"gt_{i}.npy"), gt_array)
                    np.save(os.path.join(self.results_folder, f"map_{i}.npy"), map_array)
            
            #====================test sinr level end=====================#

            gts = np.array(torch.cat(gts, dim=0).cpu())
            mapts = np.array(torch.cat(mapts, dim=0).cpu())
            final_results = np.array(torch.cat(final_results, dim=0).cpu())
            add_noise = np.array(torch.cat(add_noise, dim=0).cpu())
            print(final_results.shape)
            print(gts.shape)
            print(mapts.shape)
            print(add_noise.shape)
            os.makedirs(os.path.join(self.results_folder, "test_timestep_"+str(timesteps)+"-model_"+str(model)), exist_ok=True)
            np.save(os.path.join(self.results_folder, "test_timestep_"+str(timesteps)+"-model_"+str(model), "denoised.npy"), final_results)
            np.save(os.path.join(self.results_folder, "test_timestep_"+str(timesteps)+"-model_"+str(model), "gt.npy"), gts)
            np.save(os.path.join(self.results_folder, "test_timestep_"+str(timesteps)+"-model_"+str(model), "map.npy"), mapts)
            np.save(os.path.join(self.results_folder, "test_timestep_"+str(timesteps)+"-model_"+str(model), "add_noise.npy"), add_noise)
            
        print("test end")

    def set_results_folder(self, path):
        self.results_folder = Path(path)
        if not self.results_folder.exists():
            os.makedirs(self.results_folder)

    