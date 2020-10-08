# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Default metric definitions."""

from dnnlib import EasyDict

#----------------------------------------------------------------------------

metric_defaults = EasyDict([(args.name, args) for args in [
    # ADA paper.
    EasyDict(name='fid50k_full',     class_name='metrics.frechet_inception_distance.FID', max_reals=None, num_fakes=50000, minibatch_per_gpu=8, force_dataset_args=dict(shuffle=False, max_images=None, repeat=False, mirror_augment=False)),
    EasyDict(name='kid50k_full',     class_name='metrics.kernel_inception_distance.KID',  max_reals=1000000, num_fakes=50000, minibatch_per_gpu=8, force_dataset_args=dict(shuffle=False, max_images=None, repeat=False, mirror_augment=False)),
    EasyDict(name='pr50k3_full',     class_name='metrics.precision_recall.PR',            max_reals=200000, num_fakes=50000, nhood_size=3, minibatch_per_gpu=8, row_batch_size=10000, col_batch_size=10000, force_dataset_args=dict(shuffle=False, max_images=None, repeat=False, mirror_augment=False)),
    EasyDict(name='is50k',           class_name='metrics.inception_score.IS',             num_images=50000, num_splits=10, minibatch_per_gpu=8, force_dataset_args=dict(shuffle=False, max_images=None)),

    # Legacy: StyleGAN2.
    EasyDict(name='fid50k',          class_name='metrics.frechet_inception_distance.FID', max_reals=50000, num_fakes=50000, minibatch_per_gpu=8, force_dataset_args=dict(shuffle=False, max_images=None)),
    EasyDict(name='kid50k',          class_name='metrics.kernel_inception_distance.KID',  max_reals=50000, num_fakes=50000, minibatch_per_gpu=8, force_dataset_args=dict(shuffle=False, max_images=None)),
    EasyDict(name='pr50k3',          class_name='metrics.precision_recall.PR',            max_reals=50000, num_fakes=50000, nhood_size=3, minibatch_per_gpu=8, row_batch_size=10000, col_batch_size=10000, force_dataset_args=dict(shuffle=False, max_images=None)),
    EasyDict(name='ppl2_wend',       class_name='metrics.perceptual_path_length.PPL',     num_samples=50000, epsilon=1e-4, space='w', sampling='end', crop=False, minibatch_per_gpu=2, force_dataset_args=dict(shuffle=False, max_images=None), force_G_kwargs=dict(dtype='float32', mapping_dtype='float32', num_fp16_res=0)),

    # Legacy: StyleGAN.
    EasyDict(name='ppl_zfull',       class_name='metrics.perceptual_path_length.PPL',     num_samples=50000, epsilon=1e-4, space='z', sampling='full', crop=True, minibatch_per_gpu=2, force_dataset_args=dict(shuffle=False, max_images=None), force_G_kwargs=dict(dtype='float32', mapping_dtype='float32', num_fp16_res=0)),
    EasyDict(name='ppl_wfull',       class_name='metrics.perceptual_path_length.PPL',     num_samples=50000, epsilon=1e-4, space='w', sampling='full', crop=True, minibatch_per_gpu=2, force_dataset_args=dict(shuffle=False, max_images=None), force_G_kwargs=dict(dtype='float32', mapping_dtype='float32', num_fp16_res=0)),
    EasyDict(name='ppl_zend',        class_name='metrics.perceptual_path_length.PPL',     num_samples=50000, epsilon=1e-4, space='z', sampling='end', crop=True, minibatch_per_gpu=2, force_dataset_args=dict(shuffle=False, max_images=None), force_G_kwargs=dict(dtype='float32', mapping_dtype='float32', num_fp16_res=0)),
    EasyDict(name='ppl_wend',        class_name='metrics.perceptual_path_length.PPL',     num_samples=50000, epsilon=1e-4, space='w', sampling='end', crop=True, minibatch_per_gpu=2, force_dataset_args=dict(shuffle=False, max_images=None), force_G_kwargs=dict(dtype='float32', mapping_dtype='float32', num_fp16_res=0)),
    EasyDict(name='ls',              class_name='metrics.linear_separability.LS',         num_samples=200000, num_keep=100000, attrib_indices=range(40), minibatch_per_gpu=4, force_dataset_args=dict(shuffle=False, max_images=None)),
]])

#----------------------------------------------------------------------------
