# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Training Generative Adversarial Networks with Limited Data"."""

import os
import argparse
import json
import re
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib

from training import training_loop
from training import dataset
from metrics import metric_defaults

#----------------------------------------------------------------------------

class UserError(Exception):
    pass

#----------------------------------------------------------------------------

def setup_training_options(
    # General options (not included in desc).
    gpus       = None, # Number of GPUs: <int>, default = 1 gpu
    snap       = None, # Snapshot interval: <int>, default = 50 ticks

    # Training dataset.
    data       = None, # Training dataset (required): <path>
    res        = None, # Override dataset resolution: <int>, default = highest available
    mirror     = None, # Augment dataset with x-flips: <bool>, default = False

    # Metrics (not included in desc).
    metrics    = None, # List of metric names: [], ['fid50k_full'] (default), ...
    metricdata = None, # Metric dataset (optional): <path>

    # Base config.
    cfg        = None, # Base config: 'auto' (default), 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar', 'cifarbaseline'
    gamma      = None, # Override R1 gamma: <float>, default = depends on cfg
    kimg       = None, # Override training duration: <int>, default = depends on cfg

    # Discriminator augmentation.
    aug        = None, # Augmentation mode: 'ada' (default), 'noaug', 'fixed', 'adarv'
    p          = None, # Specify p for 'fixed' (required): <float>
    target     = None, # Override ADA target for 'ada' and 'adarv': <float>, default = depends on aug
    augpipe    = None, # Augmentation pipeline: 'blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc' (default), ..., 'bgcfnc'

    # Comparison methods.
    cmethod    = None, # Comparison method: 'nocmethod' (default), 'bcr', 'zcr', 'pagan', 'wgangp', 'auxrot', 'spectralnorm', 'shallowmap', 'adropout'
    dcap       = None, # Multiplier for discriminator capacity: <float>, default = 1

    # Transfer learning.
    resume     = None, # Load previous network: 'noresume' (default), 'ffhq256', 'ffhq512', 'ffhq1024', 'celebahq256', 'lsundog256', <file>, <url>
    freezed    = None, # Freeze-D: <int>, default = 0 discriminator layers
):
    # Initialize dicts.
    args = dnnlib.EasyDict()
    args.G_args = dnnlib.EasyDict(func_name='training.networks.G_main')
    args.D_args = dnnlib.EasyDict(func_name='training.networks.D_main')
    args.G_opt_args = dnnlib.EasyDict(beta1=0.0, beta2=0.99)
    args.D_opt_args = dnnlib.EasyDict(beta1=0.0, beta2=0.99)
    args.loss_args = dnnlib.EasyDict(func_name='training.loss.stylegan2')
    args.augment_args = dnnlib.EasyDict(class_name='training.augment.AdaptiveAugment')

    # ---------------------------
    # General options: gpus, snap
    # ---------------------------

    if gpus is None:
        gpus = 1
    assert isinstance(gpus, int)
    if not (gpus >= 1 and gpus & (gpus - 1) == 0):
        raise UserError('--gpus must be a power of two')
    args.num_gpus = gpus

    if snap is None:
        snap = 50
    assert isinstance(snap, int)
    if snap < 1:
        raise UserError('--snap must be at least 1')
    args.image_snapshot_ticks = snap
    args.network_snapshot_ticks = snap

    # -----------------------------------
    # Training dataset: data, res, mirror
    # -----------------------------------

    assert data is not None
    assert isinstance(data, str)
    data_name = os.path.basename(os.path.abspath(data))
    if not os.path.isdir(data) or len(data_name) == 0:
        raise UserError('--data must point to a directory containing *.tfrecords')
    desc = data_name

    with tf.Graph().as_default(), tflib.create_session().as_default(): # pylint: disable=not-context-manager
        args.train_dataset_args = dnnlib.EasyDict(path=data, max_label_size='full')
        dataset_obj = dataset.load_dataset(**args.train_dataset_args) # try to load the data and see what comes out
        args.train_dataset_args.resolution = dataset_obj.shape[-1] # be explicit about resolution
        args.train_dataset_args.max_label_size = dataset_obj.label_size # be explicit about label size
        validation_set_available = dataset_obj.has_validation_set
        dataset_obj.close()
        dataset_obj = None

    if res is None:
        res = args.train_dataset_args.resolution
    else:
        assert isinstance(res, int)
        if not (res >= 4 and res & (res - 1) == 0):
            raise UserError('--res must be a power of two and at least 4')
        if res > args.train_dataset_args.resolution:
            raise UserError(f'--res cannot exceed maximum available resolution in the dataset ({args.train_dataset_args.resolution})')
        desc += f'-res{res:d}'
    args.train_dataset_args.resolution = res

    if mirror is None:
        mirror = False
    else:
        assert isinstance(mirror, bool)
        if mirror:
            desc += '-mirror'
    args.train_dataset_args.mirror_augment = mirror

    # ----------------------------
    # Metrics: metrics, metricdata
    # ----------------------------

    if metrics is None:
        metrics = ['fid50k_full']
    assert isinstance(metrics, list)
    assert all(isinstance(metric, str) for metric in metrics)

    args.metric_arg_list = []
    for metric in metrics:
        if metric not in metric_defaults.metric_defaults:
            raise UserError('\n'.join(['--metrics can only contain the following values:', 'none'] + list(metric_defaults.metric_defaults.keys())))
        args.metric_arg_list.append(metric_defaults.metric_defaults[metric])

    args.metric_dataset_args = dnnlib.EasyDict(args.train_dataset_args)
    if metricdata is not None:
        assert isinstance(metricdata, str)
        if not os.path.isdir(metricdata):
            raise UserError('--metricdata must point to a directory containing *.tfrecords')
        args.metric_dataset_args.path = metricdata

    # -----------------------------
    # Base config: cfg, gamma, kimg
    # -----------------------------

    if cfg is None:
        cfg = 'auto'
    assert isinstance(cfg, str)
    desc += f'-{cfg}'

    cfg_specs = {
        'auto':          dict(ref_gpus=-1, kimg=25000,  mb=-1, mbstd=-1, fmaps=-1,  lrate=-1,     gamma=-1,   ema=-1,  ramp=0.05, map=2), # populated dynamically based on 'gpus' and 'res'
        'stylegan2':     dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=10,   ema=10,  ramp=None, map=8), # uses mixed-precision, unlike original StyleGAN2
        'paper256':      dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=0.5, lrate=0.0025, gamma=1,    ema=20,  ramp=None, map=8),
        'paper512':      dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=1,   lrate=0.0025, gamma=0.5,  ema=20,  ramp=None, map=8),
        'paper1024':     dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=2,    ema=10,  ramp=None, map=8),
        'cifar':         dict(ref_gpus=2,  kimg=100000, mb=64, mbstd=32, fmaps=0.5, lrate=0.0025, gamma=0.01, ema=500, ramp=0.05, map=2),
        'cifarbaseline': dict(ref_gpus=2,  kimg=100000, mb=64, mbstd=32, fmaps=0.5, lrate=0.0025, gamma=0.01, ema=500, ramp=0.05, map=8),
    }

    assert cfg in cfg_specs
    spec = dnnlib.EasyDict(cfg_specs[cfg])
    if cfg == 'auto':
        desc += f'{gpus:d}'
        spec.ref_gpus = gpus
        spec.mb = max(min(gpus * min(4096 // res, 32), 64), gpus) # keep gpu memory consumption at bay
        spec.mbstd = min(spec.mb // gpus, 4) # other hyperparams behave more predictably if mbstd group size remains fixed
        spec.fmaps = 1 if res >= 512 else 0.5
        spec.lrate = 0.002 if res >= 1024 else 0.0025
        spec.gamma = 0.0002 * (res ** 2) / spec.mb # heuristic formula
        spec.ema = spec.mb * 10 / 32

    args.total_kimg = spec.kimg
    args.minibatch_size = spec.mb
    args.minibatch_gpu = spec.mb // spec.ref_gpus
    args.D_args.mbstd_group_size = spec.mbstd
    args.G_args.fmap_base = args.D_args.fmap_base = int(spec.fmaps * 16384)
    args.G_args.fmap_max = args.D_args.fmap_max = 512
    args.G_opt_args.learning_rate = args.D_opt_args.learning_rate = spec.lrate
    args.loss_args.r1_gamma = spec.gamma
    args.G_smoothing_kimg = spec.ema
    args.G_smoothing_rampup = spec.ramp
    args.G_args.mapping_layers = spec.map
    args.G_args.num_fp16_res = args.D_args.num_fp16_res = 4 # enable mixed-precision training
    args.G_args.conv_clamp = args.D_args.conv_clamp = 256 # clamp activations to avoid float16 overflow

    if cfg == 'cifar':
        args.loss_args.pl_weight = 0 # disable path length regularization
        args.G_args.style_mixing_prob = None # disable style mixing
        args.D_args.architecture = 'orig' # disable residual skip connections

    if gamma is not None:
        assert isinstance(gamma, float)
        if not gamma >= 0:
            raise UserError('--gamma must be non-negative')
        desc += f'-gamma{gamma:g}'
        args.loss_args.r1_gamma = gamma

    if kimg is not None:
        assert isinstance(kimg, int)
        if not kimg >= 1:
            raise UserError('--kimg must be at least 1')
        desc += f'-kimg{kimg:d}'
        args.total_kimg = kimg

    # ---------------------------------------------------
    # Discriminator augmentation: aug, p, target, augpipe
    # ---------------------------------------------------

    if aug is None:
        aug = 'ada'
    else:
        assert isinstance(aug, str)
        desc += f'-{aug}'

    if aug == 'ada':
        args.augment_args.tune_heuristic = 'rt'
        args.augment_args.tune_target = 0.6

    elif aug == 'noaug':
        pass

    elif aug == 'fixed':
        if p is None:
            raise UserError(f'--aug={aug} requires specifying --p')

    elif aug == 'adarv':
        if not validation_set_available:
            raise UserError(f'--aug={aug} requires separate validation set; please see "python dataset_tool.py pack -h"')
        args.augment_args.tune_heuristic = 'rv'
        args.augment_args.tune_target = 0.5

    else:
        raise UserError(f'--aug={aug} not supported')

    if p is not None:
        assert isinstance(p, float)
        if aug != 'fixed':
            raise UserError('--p can only be specified with --aug=fixed')
        if not 0 <= p <= 1:
            raise UserError('--p must be between 0 and 1')
        desc += f'-p{p:g}'
        args.augment_args.initial_strength = p

    if target is not None:
        assert isinstance(target, float)
        if aug not in ['ada', 'adarv']:
            raise UserError('--target can only be specified with --aug=ada or --aug=adarv')
        if not 0 <= target <= 1:
            raise UserError('--target must be between 0 and 1')
        desc += f'-target{target:g}'
        args.augment_args.tune_target = target

    assert augpipe is None or isinstance(augpipe, str)
    if augpipe is None:
        augpipe = 'bgc'
    else:
        if aug == 'noaug':
            raise UserError('--augpipe cannot be specified with --aug=noaug')
        desc += f'-{augpipe}'

    augpipe_specs = {
        'blit':     dict(xflip=1, rotate90=1, xint=1),
        'geom':     dict(scale=1, rotate=1, aniso=1, xfrac=1),
        'color':    dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'filter':   dict(imgfilter=1),
        'noise':    dict(noise=1),
        'cutout':   dict(cutout=1),
        'bg':       dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
        'bgc':      dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'bgcf':     dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1),
        'bgcfn':    dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1),
        'bgcfnc':   dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),
    }

    assert augpipe in augpipe_specs
    if aug != 'noaug':
        args.augment_args.apply_func = 'training.augment.augment_pipeline'
        args.augment_args.apply_args = augpipe_specs[augpipe]

    # ---------------------------------
    # Comparison methods: cmethod, dcap
    # ---------------------------------

    assert cmethod is None or isinstance(cmethod, str)
    if cmethod is None:
        cmethod = 'nocmethod'
    else:
        desc += f'-{cmethod}'

    if cmethod == 'nocmethod':
        pass

    elif cmethod == 'bcr':
        args.loss_args.func_name = 'training.loss.cmethods'
        args.loss_args.bcr_real_weight = 10
        args.loss_args.bcr_fake_weight = 10
        args.loss_args.bcr_augment = dnnlib.EasyDict(func_name='training.augment.augment_pipeline', xint=1, xint_max=1/32)

    elif cmethod == 'zcr':
        args.loss_args.func_name = 'training.loss.cmethods'
        args.loss_args.zcr_gen_weight = 0.02
        args.loss_args.zcr_dis_weight = 0.2
        args.G_args.num_fp16_res = args.D_args.num_fp16_res = 0 # disable mixed-precision training
        args.G_args.conv_clamp = args.D_args.conv_clamp = None

    elif cmethod == 'pagan':
        if aug != 'noaug':
            raise UserError(f'--cmethod={cmethod} is not compatible with discriminator augmentation; please specify --aug=noaug')
        args.D_args.use_pagan = True
        args.augment_args.tune_heuristic = 'rt' # enable ada heuristic
        args.augment_args.pop('apply_func', None) # disable discriminator augmentation
        args.augment_args.pop('apply_args', None)
        args.augment_args.tune_target = 0.95

    elif cmethod == 'wgangp':
        if aug != 'noaug':
            raise UserError(f'--cmethod={cmethod} is not compatible with discriminator augmentation; please specify --aug=noaug')
        if gamma is not None:
            raise UserError(f'--cmethod={cmethod} is not compatible with --gamma')
        args.loss_args = dnnlib.EasyDict(func_name='training.loss.wgangp')
        args.G_opt_args.learning_rate = args.D_opt_args.learning_rate = 0.001
        args.G_args.num_fp16_res = args.D_args.num_fp16_res = 0 # disable mixed-precision training
        args.G_args.conv_clamp = args.D_args.conv_clamp = None
        args.lazy_regularization = False

    elif cmethod == 'auxrot':
        if args.train_dataset_args.max_label_size > 0:
            raise UserError(f'--cmethod={cmethod} is not compatible with label conditioning; please specify a dataset without labels')
        args.loss_args.func_name = 'training.loss.cmethods'
        args.loss_args.auxrot_alpha = 10
        args.loss_args.auxrot_beta = 5
        args.D_args.score_max = 5 # prepare D to output 5 scalars per image instead of just 1

    elif cmethod == 'spectralnorm':
        args.D_args.use_spectral_norm = True

    elif cmethod == 'shallowmap':
        if args.G_args.mapping_layers == 2:
            raise UserError(f'--cmethod={cmethod} is a no-op for --cfg={cfg}')
        args.G_args.mapping_layers = 2

    elif cmethod == 'adropout':
        if aug != 'noaug':
            raise UserError(f'--cmethod={cmethod} is not compatible with discriminator augmentation; please specify --aug=noaug')
        args.D_args.adaptive_dropout = 1
        args.augment_args.tune_heuristic = 'rt' # enable ada heuristic
        args.augment_args.pop('apply_func', None) # disable discriminator augmentation
        args.augment_args.pop('apply_args', None)
        args.augment_args.tune_target = 0.6

    else:
        raise UserError(f'--cmethod={cmethod} not supported')

    if dcap is not None:
        assert isinstance(dcap, float)
        if not dcap > 0:
            raise UserError('--dcap must be positive')
        desc += f'-dcap{dcap:g}'
        args.D_args.fmap_base = max(int(args.D_args.fmap_base * dcap), 1)
        args.D_args.fmap_max = max(int(args.D_args.fmap_max * dcap), 1)

    # ----------------------------------
    # Transfer learning: resume, freezed
    # ----------------------------------

    resume_specs = {
        'ffhq256':      'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl',
        'ffhq512':      'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/transfer-learning-source-nets/ffhq-res512-mirror-stylegan2-noaug.pkl',
        'ffhq1024':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/transfer-learning-source-nets/ffhq-res1024-mirror-stylegan2-noaug.pkl',
        'celebahq256':  'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/transfer-learning-source-nets/celebahq-res256-mirror-paper256-kimg100000-ada-target0.5.pkl',
        'lsundog256':   'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/transfer-learning-source-nets/lsundog-res256-paper256-kimg100000-noaug.pkl',
    }

    assert resume is None or isinstance(resume, str)
    if resume is None:
        resume = 'noresume'
    elif resume == 'noresume':
        desc += '-noresume'
    elif resume in resume_specs:
        desc += f'-resume{resume}'
        args.resume_pkl = resume_specs[resume] # predefined url
    else:
        desc += '-resumecustom'
        args.resume_pkl = resume # custom path or url

    if resume != 'noresume':
        args.augment_args.tune_kimg = 100 # make ADA react faster at the beginning
        args.G_smoothing_rampup = None # disable EMA rampup

    if freezed is not None:
        assert isinstance(freezed, int)
        if not freezed >= 0:
            raise UserError('--freezed must be non-negative')
        desc += f'-freezed{freezed:d}'
        args.D_args.freeze_layers = freezed

    return desc, args

#----------------------------------------------------------------------------

def run_training(outdir, seed, dry_run, **hyperparam_options):
    # Setup training options.
    tflib.init_tf({'rnd.np_random_seed': seed})
    run_desc, training_options = setup_training_options(**hyperparam_options)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    training_options.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{run_desc}')
    assert not os.path.exists(training_options.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(training_options, indent=2))
    print()
    print(f'Output directory:  {training_options.run_dir}')
    print(f'Training data:     {training_options.train_dataset_args.path}')
    print(f'Training length:   {training_options.total_kimg} kimg')
    print(f'Resolution:        {training_options.train_dataset_args.resolution}')
    print(f'Number of GPUs:    {training_options.num_gpus}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Kick off training.
    print('Creating output directory...')
    os.makedirs(training_options.run_dir)
    with open(os.path.join(training_options.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(training_options, f, indent=2)
    with dnnlib.util.Logger(os.path.join(training_options.run_dir, 'log.txt')):
        training_loop.training_loop(**training_options)

#----------------------------------------------------------------------------

def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def _parse_comma_sep(s):
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

_cmdline_help_epilog = '''examples:

  # Train custom dataset using 1 GPU.
  python %(prog)s --outdir=~/training-runs --gpus=1 --data=~/datasets/custom

  # Train class-conditional CIFAR-10 using 2 GPUs.
  python %(prog)s --outdir=~/training-runs --gpus=2 --data=~/datasets/cifar10c \\
      --cfg=cifar

  # Transfer learn MetFaces from FFHQ using 4 GPUs.
  python %(prog)s --outdir=~/training-runs --gpus=4 --data=~/datasets/metfaces \\
      --cfg=paper1024 --mirror=1 --resume=ffhq1024 --snap=10

  # Reproduce original StyleGAN2 config F.
  python %(prog)s --outdir=~/training-runs --gpus=8 --data=~/datasets/ffhq \\
      --cfg=stylegan2 --res=1024 --mirror=1 --aug=noaug

available base configs (--cfg):
  auto           Automatically select reasonable defaults based on resolution
                 and GPU count. Good starting point for new datasets.
  stylegan2      Reproduce results for StyleGAN2 config F at 1024x1024.
  paper256       Reproduce results for FFHQ and LSUN Cat at 256x256.
  paper512       Reproduce results for BreCaHAD and AFHQ at 512x512.
  paper1024      Reproduce results for MetFaces at 1024x1024.
  cifar          Reproduce results for CIFAR-10 (tuned configuration).
  cifarbaseline  Reproduce results for CIFAR-10 (baseline configuration).

transfer learning source networks (--resume):
  ffhq256        FFHQ trained at 256x256 resolution.
  ffhq512        FFHQ trained at 512x512 resolution.
  ffhq1024       FFHQ trained at 1024x1024 resolution.
  celebahq256    CelebA-HQ trained at 256x256 resolution.
  lsundog256     LSUN Dog trained at 256x256 resolution.
  <path or URL>  Custom network pickle.
'''

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Train a GAN using the techniques described in the paper\n"Training Generative Adversarial Networks with Limited Data".',
        epilog=_cmdline_help_epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    group = parser.add_argument_group('general options')
    group.add_argument('--outdir', help='Where to save the results (required)', required=True, metavar='DIR')
    group.add_argument('--gpus', help='Number of GPUs to use (default: 1 gpu)', type=int, metavar='INT')
    group.add_argument('--snap', help='Snapshot interval (default: 50 ticks)', type=int, metavar='INT')
    group.add_argument('--seed', help='Random seed (default: %(default)s)', type=int, default=1000, metavar='INT')
    group.add_argument('-n', '--dry-run', help='Print training options and exit', action='store_true', default=False)

    group = parser.add_argument_group('training dataset')
    group.add_argument('--data',   help='Training dataset path (required)', metavar='PATH', required=True)
    group.add_argument('--res',    help='Dataset resolution (default: highest available)', type=int, metavar='INT')
    group.add_argument('--mirror', help='Augment dataset with x-flips (default: false)', type=_str_to_bool, metavar='BOOL')

    group = parser.add_argument_group('metrics')
    group.add_argument('--metrics',    help='Comma-separated list or "none" (default: fid50k_full)', type=_parse_comma_sep, metavar='LIST')
    group.add_argument('--metricdata', help='Dataset to evaluate metrics against (optional)', metavar='PATH')

    group = parser.add_argument_group('base config')
    group.add_argument('--cfg',   help='Base config (default: auto)', choices=['auto', 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar', 'cifarbaseline'])
    group.add_argument('--gamma', help='Override R1 gamma', type=float, metavar='FLOAT')
    group.add_argument('--kimg',  help='Override training duration', type=int, metavar='INT')

    group = parser.add_argument_group('discriminator augmentation')
    group.add_argument('--aug',    help='Augmentation mode (default: ada)', choices=['noaug', 'ada', 'fixed', 'adarv'])
    group.add_argument('--p',      help='Specify augmentation probability for --aug=fixed', type=float, metavar='FLOAT')
    group.add_argument('--target', help='Override ADA target for --aug=ada and --aug=adarv', type=float)
    group.add_argument('--augpipe', help='Augmentation pipeline (default: bgc)', choices=['blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc', 'bgcf', 'bgcfn', 'bgcfnc'])

    group = parser.add_argument_group('comparison methods')
    group.add_argument('--cmethod', help='Comparison method (default: nocmethod)', choices=['nocmethod', 'bcr', 'zcr', 'pagan', 'wgangp', 'auxrot', 'spectralnorm', 'shallowmap', 'adropout'])
    group.add_argument('--dcap',    help='Multiplier for discriminator capacity', type=float, metavar='FLOAT')

    group = parser.add_argument_group('transfer learning')
    group.add_argument('--resume',  help='Resume from network pickle (default: noresume)')
    group.add_argument('--freezed', help='Freeze-D (default: 0 discriminator layers)', type=int, metavar='INT')

    args = parser.parse_args()
    try:
        run_training(**vars(args))
    except UserError as err:
        print(f'Error: {err}')
        exit(1)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
