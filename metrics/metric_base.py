# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Common definitions for quality metrics."""

import os
import time
import hashlib
import pickle
import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib

from training import dataset

#----------------------------------------------------------------------------
# Base class for metrics.

class MetricBase:
    def __init__(self, name, force_dataset_args={}, force_G_kwargs={}):
        # Constructor args.
        self.name = name
        self.force_dataset_args = force_dataset_args
        self.force_G_kwargs = force_G_kwargs

        # Configuration.
        self._dataset_args  = dnnlib.EasyDict()
        self._run_dir       = None
        self._progress_fn   = None

        # Internal state.
        self._results       = []
        self._network_name  = ''
        self._eval_time     = 0
        self._dataset       = None

    def configure(self, dataset_args={}, run_dir=None, progress_fn=None):
        self._dataset_args = dnnlib.EasyDict(dataset_args)
        self._dataset_args.update(self.force_dataset_args)
        self._run_dir = run_dir
        self._progress_fn = progress_fn

    def run(self, network_pkl, num_gpus=1, G_kwargs=dict(is_validation=True)):
        self._results = []
        self._network_name = os.path.splitext(os.path.basename(network_pkl))[0]
        self._eval_time = 0
        self._dataset = None

        with tf.Graph().as_default(), tflib.create_session().as_default(): # pylint: disable=not-context-manager
            self._report_progress(0, 1)
            time_begin = time.time()
            with dnnlib.util.open_url(network_pkl) as f:
                G, D, Gs = pickle.load(f)

            G_kwargs = dnnlib.EasyDict(G_kwargs)
            G_kwargs.update(self.force_G_kwargs)
            self._evaluate(G=G, D=D, Gs=Gs, G_kwargs=G_kwargs, num_gpus=num_gpus)

            self._eval_time = time.time() - time_begin # pylint: disable=attribute-defined-outside-init
            self._report_progress(1, 1)
            if self._dataset is not None:
                self._dataset.close()
                self._dataset = None

        result_str = self.get_result_str()
        print(result_str)
        if self._run_dir is not None and os.path.isdir(self._run_dir):
            with open(os.path.join(self._run_dir, f'metric-{self.name}.txt'), 'at') as f:
                f.write(result_str + '\n')

    def get_result_str(self):
        title = self._network_name
        if len(title) > 29:
            title = '...' + title[-26:]
        result_str = f'{title:<30s} time {dnnlib.util.format_time(self._eval_time):<12s}'
        for res in self._results:
            result_str += f' {self.name}{res.suffix} {res.fmt % res.value}'
        return result_str.strip()

    def update_autosummaries(self):
        for res in self._results:
            tflib.autosummary.autosummary('Metrics/' + self.name + res.suffix, res.value)

    def _evaluate(self, **_kwargs):
        raise NotImplementedError # to be overridden by subclasses

    def _report_result(self, value, suffix='', fmt='%-10.4f'):
        self._results += [dnnlib.EasyDict(value=value, suffix=suffix, fmt=fmt)]

    def _report_progress(self, cur, total):
        if self._progress_fn is not None:
            self._progress_fn(cur, total)

    def _get_cache_file_for_reals(self, extension='pkl', **kwargs):
        all_args = dnnlib.EasyDict(metric_name=self.name)
        all_args.update(self._dataset_args)
        all_args.update(kwargs)
        md5 = hashlib.md5(repr(sorted(all_args.items())).encode('utf-8'))
        dataset_name = os.path.splitext(os.path.basename(self._dataset_args.path))[0]
        return dnnlib.make_cache_dir_path('metrics', f'{md5.hexdigest()}-{self.name}-{dataset_name}.{extension}')

    def _get_dataset_obj(self):
        if self._dataset is None:
            self._dataset = dataset.load_dataset(**self._dataset_args)
        return self._dataset

    def _iterate_reals(self, minibatch_size):
        print(f'Calculating real image statistics for {self.name}...')
        dataset_obj = self._get_dataset_obj()
        while True:
            images = []
            labels = []
            for _ in range(minibatch_size):
                image, label = dataset_obj.get_minibatch_np(1)
                if image is None:
                    break
                images.append(image)
                labels.append(label)
            num = len(images)
            if num == 0:
                break
            images = np.concatenate(images + [images[-1]] * (minibatch_size - num), axis=0)
            labels = np.concatenate(labels + [labels[-1]] * (minibatch_size - num), axis=0)
            yield images, labels, num
            if num < minibatch_size:
                break

    def _get_random_labels_tf(self, minibatch_size):
        return self._get_dataset_obj().get_random_labels_tf(minibatch_size)

#----------------------------------------------------------------------------
