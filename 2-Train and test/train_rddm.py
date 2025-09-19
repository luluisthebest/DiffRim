# Based on RDDM https://link.zhihu.com/?target=https%3A//github.com/nachifur/RDDM
import os
import sys
sys.path.append('/home/liululu/code/rd_map_temporal_spatial_denoising_autoencoder')
#from rddm_three_frames import GaussianDiffusion
from src.residual_denoising_diffusion_pytorch import (ResidualDiffusion,
                                                      Trainer, DiffRim, 
                                                      set_seed)
# from src.rddm_train_oneframe import (ResidualDiffusion, Trainer, DiffRim, set_seed)
# from src.rddm_three_frames import (ResidualDiffusion, Trainer, DiffRim, set_seed)

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in [5])
sys.stdout.flush()
set_seed(10)
only_test = False
debug = False
adpt = False         # always False in the project
res_noise = False    # predict residual and noise if True
data = 'synt'       # choose real-world 'COSMOS' dataset/simulated 'rock' dataset/synthetic dataset 'synt'
results_folder = 'results/53'

def get_trainer(predict_res_or_noise=None, results_folder='', hparams=None):    
    
    condition = True
    objective = 'pred_res'
    num_unet = 2 if res_noise else 1
    test_res_or_noise = "res_noise"

    if data == 'cosmos':
        folder = "/home/liululu/dataset/COSMOS/"
    elif data == 'rock':
        folder = '/home/liululu/dataset/Rock/sim_200x1+25x8+25x8_1-3i/'
    else:
        folder = "/home/liululu/dataset/radical/data_split/"

    model = DiffRim(
            dim=18,
            encoder_out_channels= 128,  # 128→64
            #decoder_out_channels=3,
            decoder_out_channels=1,
            condition=condition,
            objective=objective,
            num_unet=num_unet,
        )
    diffusion = ResidualDiffusion(
        model,
        image_size=hparams['image_size'],
        timesteps=1000,           # number of steps
        # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        sampling_timesteps=hparams['sampling_timesteps'],
        loss_type='l1',            # L1 or L2
        condition=condition,
        sum_scale = hparams['sum_scale'],
        input_condition=False,
        input_condition_mask=False,
        objective=objective,
        test_res_or_noise = test_res_or_noise,
        alpha_res_to_0_or_1=predict_res_or_noise,
    ).cuda()
    
    total_params = sum(p.numel() for p in diffusion.parameters())
    trainable_params = sum(p.numel() for p in diffusion.parameters() if p.requires_grad)
    print(f"Total params: {total_params}")
    print(f"Trainable params: {trainable_params}")
    print(f"Non-trainable params: {total_params - trainable_params}")
    trainer = Trainer(
        diffusion,
        original_ddim_ddpm=False,
        data_path=folder,
        train_batch_size=hparams['train_batch_size'],
        num_samples=1,
        train_lr=8e-5,
        train_num_steps=hparams['train_num_steps'],         # total training steps
        gradient_accumulate_every=2,    # gradient accumulation steps
        ema_decay=0.995,                # exponential moving average decay
        results_folder=results_folder,
        amp=False,                        # turn on mixed precision
        condition=condition,
        save_and_sample_every=hparams['save_and_sample_every'],
        crop_patch=False,
        data=data,
        num_unet=num_unet
    )

    return trainer, hparams['train_num_steps'] // hparams['save_and_sample_every'], hparams['sampling_timesteps']

def main():
    print('debug mode: ', debug, 'only test mode: ', only_test)
    print(results_folder)
    os.makedirs(results_folder, exist_ok=True)

    hparams = {
        'debug': debug,
        'adpt': adpt,
        'res_noise': res_noise,
        'data': data,
        'original_ddim_ddpm': False,
        'image_size': 128,
        'sum_scale': 1,
        'train_lr': 8e-5,
        'train_batch_size':512,                    #128
        'sampling_timesteps': 1 if not debug else 2,
        'gradient_accumulate_every': 2,
        'train_num_steps' : 1 if debug else 40000,   #150000
        'save_and_sample_every': 1 if debug else 10000,
    }

    # train
    if not only_test:
        
        trainer,model_id,sampling_timesteps = get_trainer(predict_res_or_noise=None, results_folder=results_folder, hparams=hparams)
        trainer.train()

    # test
    if only_test:
        trainer,model_id,sampling_timesteps = get_trainer(predict_res_or_noise=None, results_folder=results_folder, hparams=hparams)
    if not trainer.accelerator.is_local_main_process:
        pass
    else:
        # trainer.load(model_id)
        trainer.load(milestone=model_id)
        trainer.set_results_folder(os.path.join(results_folder, 'test_timestep_'+str(sampling_timesteps)))
        # trainer.set_results_folder(os.path.join(results_folder, 'test_timestep_'+str(1501)))
        trainer.test(model=model_id, only_test=only_test, data=data, timesteps=sampling_timesteps, last=True)


if __name__ == '__main__':
    main()
