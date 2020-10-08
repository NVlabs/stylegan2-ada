# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Calculate quality metrics for previous training run or pretrained network pickle."""

import os
import argparse
import json
import pickle
import dnnlib
import dnnlib.tflib as tflib

from metrics import metric_defaults

#----------------------------------------------------------------------------

class UserError(Exception):
    pass

#----------------------------------------------------------------------------

def calc_metrics(network_pkl, metric_names, metricdata, mirror, gpus):
    tflib.init_tf()

    # Initialize metrics.
    metrics = []
    for name in metric_names:
        if name not in metric_defaults.metric_defaults:
            raise UserError('\n'.join(['--metrics can only contain the following values:', 'none'] + list(metric_defaults.metric_defaults.keys())))
        metrics.append(dnnlib.util.construct_class_by_name(**metric_defaults.metric_defaults[name]))

    # Load network.
    if not dnnlib.util.is_url(network_pkl, allow_file_urls=True) and not os.path.isfile(network_pkl):
        raise UserError('--network must point to a file or URL')
    print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        _G, _D, Gs = pickle.load(f)
        Gs.print_layers()

    # Look up training options.
    run_dir = None
    training_options = None
    if os.path.isfile(network_pkl):
        potential_run_dir = os.path.dirname(network_pkl)
        potential_json_file = os.path.join(potential_run_dir, 'training_options.json')
        if os.path.isfile(potential_json_file):
            print(f'Looking up training options from "{potential_json_file}"...')
            run_dir = potential_run_dir
            with open(potential_json_file, 'rt') as f:
                training_options = json.load(f, object_pairs_hook=dnnlib.EasyDict)
    if training_options is None:
        print('Could not look up training options; will rely on --metricdata and --mirror')

    # Choose dataset options.
    dataset_options = dnnlib.EasyDict()
    if training_options is not None:
        dataset_options.update(training_options.metric_dataset_args)
    dataset_options.resolution = Gs.output_shapes[0][-1]
    dataset_options.max_label_size = Gs.input_shapes[1][-1]
    if metricdata is not None:
        if not os.path.isdir(metricdata):
            raise UserError('--metricdata must point to a directory containing *.tfrecords')
        dataset_options.path = metricdata
    if mirror is not None:
        dataset_options.mirror_augment = mirror
    if 'path' not in dataset_options:
        raise UserError('--metricdata must be specified explicitly')

    # Print dataset options.
    print()
    print('Dataset options:')
    print(json.dumps(dataset_options, indent=2))

    # Evaluate metrics.
    for metric in metrics:
        print()
        print(f'Evaluating {metric.name}...')
        metric.configure(dataset_args=dataset_options, run_dir=run_dir)
        metric.run(network_pkl=network_pkl, num_gpus=gpus)

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

  # Previous training run: look up options automatically, save result to text file.
  python %(prog)s --metrics=pr50k3_full \\
      --network=~/training-runs/00000-ffhq10k-res64-auto1/network-snapshot-000000.pkl

  # Pretrained network pickle: specify dataset explicitly, print result to stdout.
  python %(prog)s --metrics=fid50k_full --metricdata=~/datasets/ffhq --mirror=1 \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/ffhq.pkl

available metrics:

  ADA paper:
    fid50k_full  Frechet inception distance against the full dataset.
    kid50k_full  Kernel inception distance against the full dataset.
    pr50k3_full  Precision and recall againt the full dataset.
    is50k        Inception score for CIFAR-10.

  Legacy: StyleGAN2
    fid50k       Frechet inception distance against 50k real images.
    kid50k       Kernel inception distance against 50k real images.
    pr50k3       Precision and recall against 50k real images.
    ppl2_wend    Perceptual path length in W at path endpoints against full image.

  Legacy: StyleGAN
    ppl_zfull    Perceptual path length in Z for full paths against cropped image.
    ppl_wfull    Perceptual path length in W for full paths against cropped image.
    ppl_zend     Perceptual path length in Z at path endpoints against cropped image.
    ppl_wend     Perceptual path length in W at path endpoints against cropped image.
    ls           Linear separability with respect to CelebA attributes.
'''

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Calculate quality metrics for previous training run or pretrained network pickle.',
        epilog=_cmdline_help_epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--network',    help='Network pickle filename or URL', dest='network_pkl', metavar='PATH')
    parser.add_argument('--metrics',    help='Comma-separated list or "none" (default: %(default)s)', dest='metric_names', type=_parse_comma_sep, default='fid50k_full', metavar='LIST')
    parser.add_argument('--metricdata', help='Dataset to evaluate metrics against (default: look up from training options)', metavar='PATH')
    parser.add_argument('--mirror',     help='Whether the dataset was augmented with x-flips during training (default: look up from training options)', type=_str_to_bool, metavar='BOOL')
    parser.add_argument('--gpus',       help='Number of GPUs to use (default: %(default)s)', type=int, default=1, metavar='INT')

    args = parser.parse_args()
    try:
        calc_metrics(**vars(args))
    except UserError as err:
        print(f'Error: {err}')
        exit(1)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
