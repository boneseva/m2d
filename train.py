import json
import os
import datetime
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
import argparse
from argparse import Namespace

from matplotlib.image import imsave
from timm.optim import optim_factory
from timm.utils import NativeScaler
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

import raw_audio_dataset
from m2d import models_mae
from m2d.engine_pretrain_m2d import train_one_epoch_m2dx
from m2d.runtime_audio import RuntimeM2D
from util import misc

import subprocess
import common

def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Namespace(**config_dict)

def validate_config(cfg):
    required_fields = [
        'root_dir',
        'sample_rate',
        'input_size'
    ]
    # add here checking if the required fields are in the config and adding default values

def ema_decay_sched(step, total_steps, ema_decay_init, ema_decay):
    interp = step / (total_steps - 1)
    tau = ema_decay_init + (ema_decay - ema_decay_init) * interp
    return tau


def get_optim(args, param_groups):
    if args.optim == 'adamw':
        return torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    elif args.optim == 'sgd':
        return torch.optim.SGD(param_groups, args.lr, momentum=0.9, weight_decay=0)
    assert False, f'Unsupported optimizer {args.optim}'


def load_model(args, model_without_ddp, optimizer, loss_scaler, delta_epoch=1, strict=True):
    if args.training['resume']:
        checkpoint = torch.load(args.training['resume'], map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=strict)
        print("Resume checkpoint %s" % args.resume)
        if strict == True and 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + delta_epoch
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")

def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_train = raw_audio_dataset.build_dataset(args, mode='train')
    dataset_val = raw_audio_dataset.build_dataset(args, mode='val')


    teacher_model = RuntimeM2D(weight_file=r"C:\Users\evabo\Documents\Repos\m2d\m2d_vit_base-80x608p16x16-221006-mr7\m2d_vit_base-80x608p16x16-221006-mr7\checkpoint-300.pth")
    models_mae.set_requires_grad(teacher_model, False)
    off_emb_dim = teacher_model.cfg.feature_d
    teacher_model.to(device)
    teacher_model.eval()

    model = models_mae.__dict__[args.model['name']](
        img_size=args.model['input_size'],
        patch_size=args.training['patch_size'],  # Assuming patch_size is in training section
        decoder_depth=args.model['decoder_depth'],
        norm_pix_loss=args.training['norm_pix_loss'],
        loss_type=args.training['loss_fn'],
        target_layers=args.model['target_layers'],
        loss_m2d=args.training['loss_m2d'],
        loss_off=args.training['loss_off'],
        off_emb_dim=off_emb_dim,
        norm_stats=args.preprocessing['norm_stats']
    )

    if args.logging['log_dir'] is not None:
        os.makedirs(args.logging['log_dir'], exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.logging['log_dir'])
    else:
        log_writer = None
    common.PrintLogger(f'{args.logging['log_dir']}/console.txt')
    print(args)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.training['batch_size'],
        num_workers=args.training['num_workers'],
        pin_memory=args.training['pin_mem'],
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.training['batch_size'],
        num_workers=args.training['num_workers'],
        pin_memory=args.training['pin_mem'],
        drop_last=False
    )

    model.set_random_1d_mask()

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    org_args_lr = args.training['learning_rate']
    if args.training['learning_rate'] is None:  # only base_lr is specified
        args.training['learning_rate'] = args.training['learning_rate'] * args.training['batch_size'] / 256

    print("base lr: %.2e" % (args.training['learning_rate'] * 256 / args.training['batch_size']) if org_args_lr is None else 'base lr: not effective')
    print("actual lr: %.2e" % args.training['learning_rate'])

    print("accumulate grad iterations: %d" % args.training['accum_iter'])
    print("effective batch size: %d" % args.training['learning_rate'])

    def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
        decay = []
        no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # skip frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}
        ]

    # Then use it for your optimizer setup:
    param_groups = add_weight_decay(model_without_ddp, weight_decay=args.training['weight_decay'])
    optimizer = torch.optim.AdamW(param_groups, lr=args.training['learning_rate'], betas=args.training['betas'])

    print(optimizer)
    loss_scaler = NativeScaler()

    load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler,
               delta_epoch=0, strict=False)

    print(f"Start training for {args.training['epochs']} epochs")
    start_time = time.time()
    last_subprocess = None
    for epoch in range(args.training['start_epoch'], args.training['epochs']):
        epoch1 = epoch + 1
        train_stats = train_one_epoch_m2dx(
            model, teacher_model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            partial(ema_decay_sched, total_steps=len(data_loader_train) * args.training['epochs'],
                    ema_decay_init=args.training['ema_decay_init']),
            val_loader=data_loader_val,
            do_analysis=(epoch1 % args.training['feature_eval_freq'] == 0),
            autocast_args=dict(dtype=torch.bfloat16) if args.training['bf16'] else {},
            args=args
        )

        if args.output_dir and (epoch1 % args.training['save_freq'] == 0 or epoch1 == args.training['epochs']):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch1)
            # run the external evaluator
            if args.eval_after <= epoch1 and epoch1 < args.training['epochs'] and misc.is_main_process():
                abspath = Path(f'{args.output_dir}/checkpoint-{epoch1}.pth').absolute()
                print('quick_eval', abspath)
                last_subprocess = subprocess.Popen(['/bin/bash', './quick_eval.sh', abspath])

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.stop_at > 0 and epoch1 >= args.stop_at:
            if last_subprocess is not None:
                last_subprocess.wait()
            print(f'Stop training by reaching args.stop_at epoch: {args.stop_at}')
            exit(0)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    del model_without_ddp, model, data_loader_train, optimizer, loss_scaler
    if misc.is_main_process():
        abspath = Path(f'{args.output_dir}/checkpoint-{epoch1}.pth').absolute()
        subprocess.call(['/bin/bash', './all_eval.sh', abspath])
    return epoch1






if __name__ == "__main__":
    # get the arguments for config path
    parser = argparse.ArgumentParser()
    config_path = parser.add_argument(
        '--config', type=str, default='./config.yaml',
        help='Path to the config file'
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    validate_config(cfg)
    main(cfg)
