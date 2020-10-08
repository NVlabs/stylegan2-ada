# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Frechet Inception Distance (FID) from the paper
"GANs trained by a two time-scale update rule converge to a local Nash equilibrium"."""

import os
import pickle
import numpy as np
import scipy
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib

from metrics import metric_base

#----------------------------------------------------------------------------

class FID(metric_base.MetricBase):
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
                mu_real, sigma_real = pickle.load(f)
        else:
            nfeat = feature_net.output_shape[1]
            mu_real = np.zeros(nfeat)
            sigma_real = np.zeros([nfeat, nfeat])
            num_real = 0
            for images, _labels, num in self._iterate_reals(minibatch_size):
                if self.max_reals is not None:
                    num = min(num, self.max_reals - num_real)
                if images.shape[1] == 1:
                    images = np.tile(images, [1, 3, 1, 1])
                for feat in list(feature_net.run(images, num_gpus=num_gpus, assume_frozen=True))[:num]:
                    mu_real += feat
                    sigma_real += np.outer(feat, feat)
                    num_real += 1
                if self.max_reals is not None and num_real >= self.max_reals:
                    break
            mu_real /= num_real
            sigma_real /= num_real
            sigma_real -= np.outer(mu_real, mu_real)
            with open(cache_file, 'wb') as f:
                pickle.dump((mu_real, sigma_real), f)

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
        mu_fake = np.mean(feat_fake, axis=0)
        sigma_fake = np.cov(feat_fake, rowvar=False)

        # Calculate FID.
        m = np.square(mu_fake - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_fake, sigma_real), disp=False) # pylint: disable=no-member
        dist = m + np.trace(sigma_fake + sigma_real - 2*s)
        self._report_result(np.real(dist))

#----------------------------------------------------------------------------
