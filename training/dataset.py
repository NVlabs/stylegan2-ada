# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Streaming images and labels from dataset created with dataset_tool.py."""

import os
import glob
import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib

#----------------------------------------------------------------------------
# Dataset class that loads images from tfrecords files.

class TFRecordDataset:
    def __init__(self,
        tfrecord_dir,               # Directory containing a collection of tfrecords files.
        resolution      = None,     # Dataset resolution, None = autodetect.
        label_file      = None,     # Relative path of the labels file, None = autodetect.
        max_label_size  = 0,        # 0 = no labels, 'full' = full labels, <int> = N first label components.
        max_images      = None,     # Maximum number of images to use, None = use all images.
        max_validation  = 10000,    # Maximum size of the validation set, None = use all available images.
        mirror_augment  = False,    # Apply mirror augment?
        repeat          = True,     # Repeat dataset indefinitely?
        shuffle         = True,     # Shuffle images?
        shuffle_mb      = 4096,     # Shuffle data within specified window (megabytes), 0 = disable shuffling.
        prefetch_mb     = 2048,     # Amount of data to prefetch (megabytes), 0 = disable prefetching.
        buffer_mb       = 256,      # Read buffer size (megabytes).
        num_threads     = 2,        # Number of concurrent threads.
        _is_validation  = False,
):
        self.tfrecord_dir       = tfrecord_dir
        self.resolution         = None
        self.resolution_log2    = None
        self.shape              = []        # [channels, height, width]
        self.dtype              = 'uint8'
        self.label_file         = label_file
        self.label_size         = None      # components
        self.label_dtype        = None
        self.has_validation_set = None
        self.mirror_augment     = mirror_augment
        self.repeat             = repeat
        self.shuffle            = shuffle
        self._max_validation    = max_validation
        self._np_labels         = None
        self._tf_minibatch_in   = None
        self._tf_labels_var     = None
        self._tf_labels_dataset = None
        self._tf_datasets       = dict()
        self._tf_iterator       = None
        self._tf_init_ops       = dict()
        self._tf_minibatch_np   = None
        self._cur_minibatch     = -1
        self._cur_lod           = -1

        # List files in the dataset directory.
        assert os.path.isdir(self.tfrecord_dir)
        all_files = sorted(glob.glob(os.path.join(self.tfrecord_dir, '*')))
        self.has_validation_set = (self._max_validation > 0) and any(os.path.basename(f).startswith('validation-') for f in all_files)
        all_files = [f for f in all_files if os.path.basename(f).startswith('validation-') == _is_validation]

        # Inspect tfrecords files.
        tfr_files = [f for f in all_files if f.endswith('.tfrecords')]
        assert len(tfr_files) >= 1
        tfr_shapes = []
        for tfr_file in tfr_files:
            tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
            for record in tf.python_io.tf_record_iterator(tfr_file, tfr_opt):
                tfr_shapes.append(self.parse_tfrecord_np(record).shape)
                break

        # Autodetect label filename.
        if self.label_file is None:
            guess = [f for f in all_files if f.endswith('.labels')]
            if len(guess):
                self.label_file = guess[0]
        elif not os.path.isfile(self.label_file):
            guess = os.path.join(self.tfrecord_dir, self.label_file)
            if os.path.isfile(guess):
                self.label_file = guess

        # Determine shape and resolution.
        max_shape = max(tfr_shapes, key=np.prod)
        self.resolution = resolution if resolution is not None else max_shape[1]
        self.resolution_log2 = int(np.log2(self.resolution))
        self.shape = [max_shape[0], self.resolution, self.resolution]
        tfr_lods = [self.resolution_log2 - int(np.log2(shape[1])) for shape in tfr_shapes]
        assert all(shape[0] == max_shape[0] for shape in tfr_shapes)
        assert all(shape[1] == shape[2] for shape in tfr_shapes)
        assert all(shape[1] == self.resolution // (2**lod) for shape, lod in zip(tfr_shapes, tfr_lods))
        assert all(lod in tfr_lods for lod in range(self.resolution_log2 - 1))

        # Load labels.
        assert max_label_size == 'full' or max_label_size >= 0
        self._np_labels = np.zeros([1<<30, 0], dtype=np.float32)
        if self.label_file is not None and max_label_size != 0:
            self._np_labels = np.load(self.label_file)
            assert self._np_labels.ndim == 2
        if max_label_size != 'full' and self._np_labels.shape[1] > max_label_size:
            self._np_labels = self._np_labels[:, :max_label_size]
        if max_images is not None and self._np_labels.shape[0] > max_images:
            self._np_labels = self._np_labels[:max_images]
        self.label_size = self._np_labels.shape[1]
        self.label_dtype = self._np_labels.dtype.name

        # Build TF expressions.
        with tf.name_scope('Dataset'), tf.device('/cpu:0'), tf.control_dependencies(None):
            self._tf_minibatch_in = tf.placeholder(tf.int64, name='minibatch_in', shape=[])
            self._tf_labels_var = tflib.create_var_with_large_initial_value(self._np_labels, name='labels_var')
            self._tf_labels_dataset = tf.data.Dataset.from_tensor_slices(self._tf_labels_var)
            for tfr_file, tfr_shape, tfr_lod in zip(tfr_files, tfr_shapes, tfr_lods):
                if tfr_lod < 0:
                    continue
                dset = tf.data.TFRecordDataset(tfr_file, compression_type='', buffer_size=buffer_mb<<20)
                if max_images is not None:
                    dset = dset.take(max_images)
                dset = dset.map(self.parse_tfrecord_tf, num_parallel_calls=num_threads)
                dset = tf.data.Dataset.zip((dset, self._tf_labels_dataset))
                bytes_per_item = np.prod(tfr_shape) * np.dtype(self.dtype).itemsize
                if self.shuffle and shuffle_mb > 0:
                    dset = dset.shuffle(((shuffle_mb << 20) - 1) // bytes_per_item + 1)
                if self.repeat:
                    dset = dset.repeat()
                if prefetch_mb > 0:
                    dset = dset.prefetch(((prefetch_mb << 20) - 1) // bytes_per_item + 1)
                dset = dset.batch(self._tf_minibatch_in)
                self._tf_datasets[tfr_lod] = dset
            self._tf_iterator = tf.data.Iterator.from_structure(self._tf_datasets[0].output_types, self._tf_datasets[0].output_shapes)
            self._tf_init_ops = {lod: self._tf_iterator.make_initializer(dset) for lod, dset in self._tf_datasets.items()}

    def close(self):
        pass

    # Use the given minibatch size and level-of-detail for the data returned by get_minibatch_tf().
    def configure(self, minibatch_size, lod=0):
        lod = int(np.floor(lod))
        assert minibatch_size >= 1 and lod in self._tf_datasets
        if self._cur_minibatch != minibatch_size or self._cur_lod != lod:
            self._tf_init_ops[lod].run({self._tf_minibatch_in: minibatch_size})
            self._cur_minibatch = minibatch_size
            self._cur_lod = lod

    # Get next minibatch as TensorFlow expressions.
    def get_minibatch_tf(self):
        images, labels = self._tf_iterator.get_next()
        if self.mirror_augment:
            images = tf.cast(images, tf.float32)
            images = tf.where(tf.random_uniform([tf.shape(images)[0]]) < 0.5, images, tf.reverse(images, [3]))
            images = tf.cast(images, self.dtype)
        return images, labels

    # Get next minibatch as NumPy arrays.
    def get_minibatch_np(self, minibatch_size, lod=0): # => (images, labels) or (None, None)
        self.configure(minibatch_size, lod)
        if self._tf_minibatch_np is None:
            with tf.name_scope('Dataset'):
                self._tf_minibatch_np = self.get_minibatch_tf()
        try:
            return tflib.run(self._tf_minibatch_np)
        except tf.errors.OutOfRangeError:
            return None, None

    # Get random labels as TensorFlow expression.
    def get_random_labels_tf(self, minibatch_size): # => labels
        with tf.name_scope('Dataset'):
            if self.label_size > 0:
                with tf.device('/cpu:0'):
                    return tf.gather(self._tf_labels_var, tf.random_uniform([minibatch_size], 0, self._np_labels.shape[0], dtype=tf.int32))
            return tf.zeros([minibatch_size, 0], self.label_dtype)

    # Get random labels as NumPy array.
    def get_random_labels_np(self, minibatch_size): # => labels
        if self.label_size > 0:
            return self._np_labels[np.random.randint(self._np_labels.shape[0], size=[minibatch_size])]
        return np.zeros([minibatch_size, 0], self.label_dtype)

    # Load validation set as NumPy array.
    def load_validation_set_np(self):
        images = []
        labels = []
        if self.has_validation_set:
            validation_set = TFRecordDataset(
                tfrecord_dir=self.tfrecord_dir, resolution=self.shape[2], max_label_size=self.label_size,
                max_images=self._max_validation, repeat=False, shuffle=False, prefetch_mb=0, _is_validation=True)
            validation_set.configure(1)
            while True:
                image, label = validation_set.get_minibatch_np(1)
                if image is None:
                    break
                images.append(image)
                labels.append(label)
        images = np.concatenate(images, axis=0) if len(images) else np.zeros([0] + self.shape, dtype=self.dtype)
        labels = np.concatenate(labels, axis=0) if len(labels) else np.zeros([0, self.label_size], self.label_dtype)
        assert list(images.shape[1:]) == self.shape
        assert labels.shape[1] == self.label_size
        assert images.shape[0] <= self._max_validation
        return images, labels

    # Parse individual image from a tfrecords file into TensorFlow expression.
    @staticmethod
    def parse_tfrecord_tf(record):
        features = tf.parse_single_example(record, features={
            'shape': tf.FixedLenFeature([3], tf.int64),
            'data': tf.FixedLenFeature([], tf.string)})
        data = tf.decode_raw(features['data'], tf.uint8)
        return tf.reshape(data, features['shape'])

    # Parse individual image from a tfrecords file into NumPy array.
    @staticmethod
    def parse_tfrecord_np(record):
        ex = tf.train.Example()
        ex.ParseFromString(record)
        shape = ex.features.feature['shape'].int64_list.value # pylint: disable=no-member
        data = ex.features.feature['data'].bytes_list.value[0] # pylint: disable=no-member
        return np.fromstring(data, np.uint8).reshape(shape)

#----------------------------------------------------------------------------
# Construct a dataset object using the given options.

def load_dataset(path=None, resolution=None, max_images=None, max_label_size=0, mirror_augment=False, repeat=True, shuffle=True, seed=None):
    _ = seed
    assert os.path.isdir(path)
    return TFRecordDataset(
        tfrecord_dir=path,
        resolution=resolution, max_images=max_images, max_label_size=max_label_size,
        mirror_augment=mirror_augment, repeat=repeat, shuffle=shuffle)

#----------------------------------------------------------------------------
