# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Inception Score (IS) from the paper
"Improved techniques for training GANs"."""

import pickle
import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib

from metrics import metric_base

#----------------------------------------------------------------------------

class IS(metric_base.MetricBase):
    def __init__(self, num_images, num_splits, minibatch_per_gpu, **kwargs):
        super().__init__(**kwargs)
        self.num_images = num_images
        self.num_splits = num_splits
        self.minibatch_per_gpu = minibatch_per_gpu

    def _evaluate(self, Gs, G_kwargs, num_gpus, **_kwargs): # pylint: disable=arguments-differ
        minibatch_size = num_gpus * self.minibatch_per_gpu
        with dnnlib.util.open_url('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metrics/inception_v3_softmax.pkl') as f:
            inception = pickle.load(f)
        activations = np.empty([self.num_images, inception.output_shape[1]], dtype=np.float32)

        # Construct TensorFlow graph.
        result_expr = []
        for gpu_idx in range(num_gpus):
            with tf.device(f'/gpu:{gpu_idx}'):
                Gs_clone = Gs.clone()
                inception_clone = inception.clone()
                latents = tf.random_normal([self.minibatch_per_gpu] + Gs_clone.input_shape[1:])
                labels = self._get_random_labels_tf(self.minibatch_per_gpu)
                images = Gs_clone.get_output_for(latents, labels, **G_kwargs)
                if images.shape[1] == 1: images = tf.tile(images, [1, 3, 1, 1])
                images = tflib.convert_images_to_uint8(images)
                result_expr.append(inception_clone.get_output_for(images))

        # Calculate activations for fakes.
        for begin in range(0, self.num_images, minibatch_size):
            self._report_progress(begin, self.num_images)
            end = min(begin + minibatch_size, self.num_images)
            activations[begin:end] = np.concatenate(tflib.run(result_expr), axis=0)[:end-begin]

        # Calculate IS.
        scores = []
        for i in range(self.num_splits):
            part = activations[i * self.num_images // self.num_splits : (i + 1) * self.num_images // self.num_splits]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
        self._report_result(np.mean(scores), suffix='_mean')
        self._report_result(np.std(scores), suffix='_std')

#----------------------------------------------------------------------------
