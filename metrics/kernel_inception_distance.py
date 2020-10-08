# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Kernel Inception Distance (KID) from the paper
"Demystifying MMD GANs"."""

import os
import pickle
import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib

from metrics import metric_base

#----------------------------------------------------------------------------

def compute_kid(feat_real, feat_fake, num_subsets=100, max_subset_size=1000):
    n = feat_real.shape[1]
    m = min(min(feat_real.shape[0], feat_fake.shape[0]), max_subset_size)
    t = 0
    for _subset_idx in range(num_subsets):
        x = feat_fake[np.random.choice(feat_fake.shape[0], m, replace=False)]
        y = feat_real[np.random.choice(feat_real.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    return t / num_subsets / m

#----------------------------------------------------------------------------

class KID(metric_base.MetricBase):
    def __init__(self, max_reals, num_fakes, minibatch_per_gpu, use_cached_real_stats=True, **kwargs):
        super().__init__(**kwargs)
        self.max_reals = max_reals
        self.num_fakes = num_fakes
        self.minibatch_per_gpu = minibatch_per_gpu
        self.use_cached_real_stats = use_cached_real_stats

    def _evaluate(self, Gs, G_kwargs, num_gpus, **_kwargs): # pylint: disable=arguments-differ
        minibatch_size = num_gpus * self.minibatch_per_gpu
        with dnnlib.util.open_url('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metrics/inception_v3_features.pkl') as f: # identical to http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
            feature_net = pickle.load(f)

        # Calculate statistics for reals.
        cache_file = self._get_cache_file_for_reals(max_reals=self.max_reals)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        if self.use_cached_real_stats and os.path.isfile(cache_file):
            with open(cache_file, 'rb') as f:
                feat_real = pickle.load(f)
        else:
            feat_real = []
            for images, _labels, num in self._iterate_reals(minibatch_size):
                if self.max_reals is not None:
                    num = min(num, self.max_reals - len(feat_real))
                if images.shape[1] == 1:
                    images = np.tile(images, [1, 3, 1, 1])
                feat_real += list(feature_net.run(images, num_gpus=num_gpus, assume_frozen=True))[:num]
                if self.max_reals is not None and len(feat_real) >= self.max_reals:
                    break
            feat_real = np.stack(feat_real)
            with open(cache_file, 'wb') as f:
                pickle.dump(feat_real, f)

        # Construct TensorFlow graph.
        result_expr = []
        for gpu_idx in range(num_gpus):
            with tf.device('/gpu:%d' % gpu_idx):
                Gs_clone = Gs.clone()
                feature_net_clone = feature_net.clone()
                latents = tf.random_normal([self.minibatch_per_gpu] + Gs_clone.input_shape[1:])
                labels = self._get_random_labels_tf(self.minibatch_per_gpu)
                images = Gs_clone.get_output_for(latents, labels, **G_kwargs)
                if images.shape[1] == 1: images = tf.tile(images, [1, 3, 1, 1])
                images = tflib.convert_images_to_uint8(images)
                result_expr.append(feature_net_clone.get_output_for(images))

        # Calculate statistics for fakes.
        feat_fake = []
        for begin in range(0, self.num_fakes, minibatch_size):
            self._report_progress(begin, self.num_fakes)
            feat_fake += list(np.concatenate(tflib.run(result_expr), axis=0))
        feat_fake = np.stack(feat_fake[:self.num_fakes])

        # Calculate KID.
        kid = compute_kid(feat_real, feat_fake)
        self._report_result(np.real(kid), fmt='%-12.8f')

#----------------------------------------------------------------------------
