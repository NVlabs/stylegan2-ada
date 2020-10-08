# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Tool for creating multi-resolution TFRecords datasets."""

import os
import sys
import glob
import argparse
import threading
import six.moves.queue as Queue
import traceback
import numpy as np
import tensorflow as tf
import PIL.Image
import dnnlib.tflib as tflib
import scipy
import scipy.ndimage
import scipy.misc
import datetime
from tqdm import tqdm

from training import dataset

#----------------------------------------------------------------------------

def error(msg):
    print('Error: ' + msg)
    exit(1)

#----------------------------------------------------------------------------

class TFRecordExporter:
    def __init__(self, tfrecord_dir, expected_images, print_progress=True, progress_interval=10, tfr_prefix=None):
        self.tfrecord_dir       = tfrecord_dir
        if tfr_prefix is None:
            self.tfr_prefix = os.path.join(self.tfrecord_dir, os.path.basename(self.tfrecord_dir))
        else:
            self.tfr_prefix = os.path.join(self.tfrecord_dir, tfr_prefix)
        self.expected_images    = expected_images
        self.cur_images         = 0
        self.shape              = None
        self.resolution_log2    = None
        self.tfr_writers        = []
        self.print_progress     = print_progress
        self.progress_interval  = progress_interval

        if self.print_progress:
            name = '' if tfr_prefix is None else f' ({tfr_prefix})'
            print(f'Creating dataset "{tfrecord_dir}"{name}')
        if not os.path.isdir(self.tfrecord_dir):
            os.makedirs(self.tfrecord_dir)
        assert os.path.isdir(self.tfrecord_dir)

    def close(self):
        if self.print_progress:
            print('%-40s\r' % 'Flushing data...', end='', flush=True)
        for tfr_writer in self.tfr_writers:
            tfr_writer.close()
        self.tfr_writers = []
        if self.print_progress:
            print('%-40s\r' % '', end='', flush=True)
            print('Added %d images.' % self.cur_images)

    def choose_shuffled_order(self): # Note: Images and labels must be added in shuffled order.
        order = np.arange(self.expected_images)
        np.random.RandomState(123).shuffle(order)
        return order

    def add_image(self, img):
        if self.print_progress and self.cur_images % self.progress_interval == 0:
            print('%d / %d\r' % (self.cur_images, self.expected_images), end='', flush=True)
        if self.shape is None:
            self.shape = img.shape
            self.resolution_log2 = int(np.log2(self.shape[1]))
            assert self.shape[0] in [1, 3]
            assert self.shape[1] == self.shape[2]
            assert self.shape[1] == 2**self.resolution_log2
            tfr_opt = tf.io.TFRecordOptions(tf.compat.v1.io.TFRecordCompressionType.NONE)
            for lod in range(self.resolution_log2 - 1):
                tfr_file = self.tfr_prefix + '-r%02d.tfrecords' % (self.resolution_log2 - lod)
                self.tfr_writers.append(tf.io.TFRecordWriter(tfr_file, tfr_opt))
        assert img.shape == self.shape
        for lod, tfr_writer in enumerate(self.tfr_writers):
            if lod:
                img = img.astype(np.float32)
                img = (img[:, 0::2, 0::2] + img[:, 0::2, 1::2] + img[:, 1::2, 0::2] + img[:, 1::2, 1::2]) * 0.25
            quant = np.rint(img).clip(0, 255).astype(np.uint8)
            ex = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=quant.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()]))}))
            tfr_writer.write(ex.SerializeToString())
        self.cur_images += 1

    def add_labels(self, labels):
        if self.print_progress:
            print('%-40s\r' % 'Saving labels...', end='', flush=True)
        assert labels.shape[0] == self.cur_images
        with open(self.tfr_prefix + '-rxx.labels', 'wb') as f:
            np.save(f, labels.astype(np.float32))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ----------------------------------------------------------------------------

class HDF5Exporter:
    def __init__(self, h5_filename, resolution, channels, compress=False, expected_images=0, print_progress=True, progress_interval=10):
        rlog2 = int(np.floor(np.log2(resolution)))
        assert resolution == 2 ** rlog2

        self.h5_filename        = h5_filename
        self.resolution         = resolution
        self.channels           = channels
        self.expected_images    = expected_images
        self.cur_images         = 0
        self.h5_file            = None
        self.h5_lods            = []
        self.buffers            = []
        self.buffer_sizes       = []
        self.print_progress     = print_progress
        self.progress_interval  = progress_interval

        if self.print_progress:
            print('Creating dataset "%s"' % h5_filename)
        import h5py  # conda install h5py
        self.h5_file = h5py.File(h5_filename, 'w')
        for lod in range(rlog2, -1, -1):
            r = 2 ** lod
            c = channels
            bytes_per_item = c * (r ** 2)
            chunk_size = int(np.ceil(128.0 / bytes_per_item))
            buffer_size = int(np.ceil(512.0 * np.exp2(20) / bytes_per_item))
            compression = 'gzip' if compress else None
            compression_opts = 4 if compress else None
            lod = self.h5_file.create_dataset(
                'data%dx%d' % (r, r), shape=(0, c, r, r), dtype=np.uint8,
                maxshape=(None, c, r, r), chunks=(chunk_size, c, r, r),
                compression=compression, compression_opts=compression_opts)
            self.h5_lods.append(lod)
            self.buffers.append(np.zeros((buffer_size, c, r, r), dtype=np.uint8))
            self.buffer_sizes.append(0)

    def close(self):
        if self.print_progress:
            print('%-40s\r' % 'Flushing data...', end='', flush=True)
        for lod in range(len(self.h5_lods)):
            self._flush_lod(lod)
        self.h5_file.close()
        self.h5_file = None
        self.h5_lods = None
        if self.print_progress:
            print('%-40s\r' % '', end='', flush=True)
            print('Added %d images.' % self.cur_images)

    def add_image(self, img):
        self.add_images(np.stack([img]))

    def add_images(self, img):
        assert img.ndim == 4 and img.shape[1] == self.channels and img.shape[2] == img.shape[3]
        assert img.shape[2] >= self.resolution and img.shape[2] == 2 ** int(np.floor(np.log2(img.shape[2])))
        if self.print_progress and (self.cur_images - 1) % self.progress_interval >= self.progress_interval - img.shape[0]:
            print('%d / %d\r' % (self.cur_images, self.expected_images), end='', flush=True)

        for lod in range(len(self.h5_lods)):
            while img.shape[2] > self.resolution // (2 ** lod):
                img = img.astype(np.float32)
                img = (img[:, :, 0::2, 0::2] + img[:, :, 0::2, 1::2] + img[:, :, 1::2, 0::2] + img[:, :, 1::2, 1::2]) * 0.25
            quant = np.uint8(np.clip(np.round(img), 0, 255))
            ofs = 0
            while ofs < quant.shape[0]:
                num = min(quant.shape[0] - ofs, self.buffers[lod].shape[0] - self.buffer_sizes[lod])
                self.buffers[lod][self.buffer_sizes[lod]: self.buffer_sizes[lod] + num] = quant[ofs: ofs + num]
                self.buffer_sizes[lod] += num
                if self.buffer_sizes[lod] == self.buffers[lod].shape[0]:
                    self._flush_lod(lod)
                ofs += num
        self.cur_images += img.shape[0]

    def add_labels(self, labels):
        if self.print_progress:
            print('%-40s\r' % 'Saving labels...', end='', flush=True)
        assert labels.shape[0] == self.cur_images
        with open(os.path.splitext(self.h5_filename)[0] + '-labels.npy', 'wb') as f:
            np.save(f, labels.astype(np.float32))

    def _flush_lod(self, lod):
        num = self.buffer_sizes[lod]
        if num > 0:
            self.h5_lods[lod].resize(self.h5_lods[lod].shape[0] + num, axis=0)
            self.h5_lods[lod][-num:] = self.buffers[lod][:num]
            self.buffer_sizes[lod] = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

#----------------------------------------------------------------------------

class ExceptionInfo(object):
    def __init__(self):
        self.value = sys.exc_info()[1]
        self.traceback = traceback.format_exc()

#----------------------------------------------------------------------------

class WorkerThread(threading.Thread):
    def __init__(self, task_queue):
        threading.Thread.__init__(self)
        self.task_queue = task_queue

    def run(self):
        while True:
            func, args, result_queue = self.task_queue.get()
            if func is None:
                break
            try:
                result = func(*args)
            except:
                result = ExceptionInfo()
            result_queue.put((result, args))

#----------------------------------------------------------------------------

class ThreadPool(object):
    def __init__(self, num_threads):
        assert num_threads >= 1
        self.task_queue = Queue.Queue()
        self.result_queues = dict()
        self.num_threads = num_threads
        for _idx in range(self.num_threads):
            thread = WorkerThread(self.task_queue)
            thread.daemon = True
            thread.start()

    def add_task(self, func, args=()):
        assert hasattr(func, '__call__') # must be a function
        if func not in self.result_queues:
            self.result_queues[func] = Queue.Queue()
        self.task_queue.put((func, args, self.result_queues[func]))

    def get_result(self, func): # returns (result, args)
        result, args = self.result_queues[func].get()
        if isinstance(result, ExceptionInfo):
            print('\n\nWorker thread caught an exception:\n' + result.traceback)
            raise result.value
        return result, args

    def finish(self):
        for _idx in range(self.num_threads):
            self.task_queue.put((None, (), None))

    def __enter__(self): # for 'with' statement
        return self

    def __exit__(self, *excinfo):
        self.finish()

    def process_items_concurrently(self, item_iterator, process_func=lambda x: x, pre_func=lambda x: x, post_func=lambda x: x, max_items_in_flight=None):
        if max_items_in_flight is None: max_items_in_flight = self.num_threads * 4
        assert max_items_in_flight >= 1
        results = []
        retire_idx = [0]

        def task_func(prepared, _idx):
            return process_func(prepared)

        def retire_result():
            processed, (_prepared, idx) = self.get_result(task_func)
            results[idx] = processed
            while retire_idx[0] < len(results) and results[retire_idx[0]] is not None:
                yield post_func(results[retire_idx[0]])
                results[retire_idx[0]] = None
                retire_idx[0] += 1

        for idx, item in enumerate(item_iterator):
            prepared = pre_func(item)
            results.append(None)
            self.add_task(func=task_func, args=(prepared, idx))
            while retire_idx[0] < idx - max_items_in_flight + 2:
                for res in retire_result(): yield res
        while retire_idx[0] < len(results):
            for res in retire_result(): yield res

#----------------------------------------------------------------------------

def info(tfrecord_dir):
    print()
    print('%-20s%s' % ('Dataset name:', os.path.basename(tfrecord_dir)))

    bytes_total = 0
    bytes_max = 0
    num_files = 0
    for f in sorted(glob.glob(os.path.join(tfrecord_dir, '*'))):
        if os.path.isfile(f):
            fs = os.stat(f).st_size
            bytes_total += fs
            bytes_max = max(bytes_max, fs)
            num_files += 1
    print('%-20s%.2f' % ('Total size GB:', bytes_total / (1 << 30)))
    print('%-20s%.2f' % ('Largest file GB:', bytes_max / (1 << 30)))
    print('%-20s%d' % ('Num files:', num_files))

    tflib.init_tf()
    dset = dataset.TFRecordDataset(tfrecord_dir, max_label_size='full', repeat=False, shuffle=False)
    tflib.init_uninitialized_vars()

    print('%-20s%d' % ('Image width:', dset.shape[2]))
    print('%-20s%d' % ('Image height:', dset.shape[1]))
    print('%-20s%d' % ('Image channels:', dset.shape[0]))
    print('%-20s%s' % ('Image datatype:', dset.dtype))
    print('%-20s%d' % ('Label size:', dset.label_size))
    print('%-20s%s' % ('Label datatype:', dset.label_dtype))

    num_images = 0
    label_min = np.finfo(np.float64).max
    label_max = np.finfo(np.float64).min
    label_norm = 0
    lod = max(dset.resolution_log2 - 2, 0)
    while True:
        print('\r%-20s%d' % ('Num images:', num_images), end='', flush=True)
        _images, labels = dset.get_minibatch_np(10000, lod=lod) # not accurate
        if labels is None:
            break
        num_images += labels.shape[0]
        if dset.label_size:
            label_min = min(label_min, np.min(labels))
            label_max = max(label_max, np.max(labels))
            label_norm += np.sum(np.sqrt(np.sum(np.square(labels), axis=1)))

    print('\r%-20s%d' % ('Num images:', num_images))
    print('%-20s%s' % ('Label range:', '%g -- %g' % (label_min, label_max) if num_images and dset.label_size else 'n/a'))
    print('%-20s%s' % ('Label L2 norm:', '%g' % (label_norm / num_images) if num_images and dset.label_size else 'n/a'))
    print()

#----------------------------------------------------------------------------

def display(tfrecord_dir):
    print('Loading dataset "%s"' % tfrecord_dir)
    tflib.init_tf()
    dset = dataset.TFRecordDataset(tfrecord_dir, max_label_size='full', repeat=False, shuffle=False)
    tflib.init_uninitialized_vars()
    import cv2  # pip install opencv-python

    idx = 0
    while True:
        images, labels = dset.get_minibatch_np(1)
        if images is None:
            break
        if idx == 0:
            print('Displaying images')
            cv2.namedWindow('dataset_tool')
            print('Press SPACE or ENTER to advance, ESC to exit')
        print('\nidx = %-8d\nlabel = %s' % (idx, labels[0].tolist()))
        cv2.imshow('dataset_tool', images[0].transpose(1, 2, 0)[:, :, ::-1]) # CHW => HWC, RGB => BGR
        idx += 1
        if cv2.waitKey() == 27:
            break
    print('\nDisplayed %d images.' % idx)

#----------------------------------------------------------------------------

def extract(tfrecord_dir, output_dir):
    print('Loading dataset "%s"' % tfrecord_dir)
    tflib.init_tf()
    dset = dataset.TFRecordDataset(tfrecord_dir, max_label_size=0, repeat=False, shuffle=False)
    tflib.init_uninitialized_vars()

    print('Extracting images to "%s"' % output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    idx = 0
    while True:
        if idx % 10 == 0:
            print('%d\r' % idx, end='', flush=True)
        images, _labels = dset.get_minibatch_np(1)
        if images is None:
            break
        if images.shape[1] == 1:
            img = PIL.Image.fromarray(images[0][0], 'L')
        else:
            img = PIL.Image.fromarray(images[0].transpose(1, 2, 0), 'RGB')
        img.save(os.path.join(output_dir, 'img%08d.png' % idx))
        idx += 1
    print('Extracted %d images.' % idx)

#----------------------------------------------------------------------------

def compare(tfrecord_dir_a, tfrecord_dir_b, ignore_labels):
    max_label_size = 0 if ignore_labels else 'full'
    print('Loading dataset "%s"' % tfrecord_dir_a)
    tflib.init_tf()
    dset_a = dataset.TFRecordDataset(tfrecord_dir_a, max_label_size=max_label_size, repeat=False, shuffle=False)
    print('Loading dataset "%s"' % tfrecord_dir_b)
    dset_b = dataset.TFRecordDataset(tfrecord_dir_b, max_label_size=max_label_size, repeat=False, shuffle=False)
    tflib.init_uninitialized_vars()

    print('Comparing datasets')
    idx = 0
    identical_images = 0
    identical_labels = 0
    while True:
        if idx % 100 == 0:
            print('%d\r' % idx, end='', flush=True)
        images_a, labels_a = dset_a.get_minibatch_np(1)
        images_b, labels_b = dset_b.get_minibatch_np(1)
        if images_a is None or images_b is None:
            if images_a is not None or images_b is not None:
                print('Datasets contain different number of images')
            break
        if images_a.shape == images_b.shape and np.all(images_a == images_b):
            identical_images += 1
        else:
            print('Image %d is different' % idx)
        if labels_a.shape == labels_b.shape and np.all(labels_a == labels_b):
            identical_labels += 1
        else:
            print('Label %d is different' % idx)
        idx += 1
    print('Identical images: %d / %d' % (identical_images, idx))
    if not ignore_labels:
        print('Identical labels: %d / %d' % (identical_labels, idx))

#----------------------------------------------------------------------------

def create_mnist(tfrecord_dir, mnist_dir):
    print('Loading MNIST from "%s"' % mnist_dir)
    import gzip
    with gzip.open(os.path.join(mnist_dir, 'train-images-idx3-ubyte.gz'), 'rb') as file:
        images = np.frombuffer(file.read(), np.uint8, offset=16)
    with gzip.open(os.path.join(mnist_dir, 'train-labels-idx1-ubyte.gz'), 'rb') as file:
        labels = np.frombuffer(file.read(), np.uint8, offset=8)
    images = images.reshape(-1, 1, 28, 28)
    images = np.pad(images, [(0,0), (0,0), (2,2), (2,2)], 'constant', constant_values=0)
    assert images.shape == (60000, 1, 32, 32) and images.dtype == np.uint8
    assert labels.shape == (60000,) and labels.dtype == np.uint8
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9
    onehot = np.zeros((labels.size, np.max(labels) + 1), dtype=np.float32)
    onehot[np.arange(labels.size), labels] = 1.0

    with TFRecordExporter(tfrecord_dir, images.shape[0]) as tfr:
        order = tfr.choose_shuffled_order()
        for idx in range(order.size):
            tfr.add_image(images[order[idx]])
        tfr.add_labels(onehot[order])

#----------------------------------------------------------------------------

def create_mnistrgb(tfrecord_dir, mnist_dir, num_images=1000000, random_seed=123):
    print('Loading MNIST from "%s"' % mnist_dir)
    import gzip
    with gzip.open(os.path.join(mnist_dir, 'train-images-idx3-ubyte.gz'), 'rb') as file:
        images = np.frombuffer(file.read(), np.uint8, offset=16)
    images = images.reshape(-1, 28, 28)
    images = np.pad(images, [(0,0), (2,2), (2,2)], 'constant', constant_values=0)
    assert images.shape == (60000, 32, 32) and images.dtype == np.uint8
    assert np.min(images) == 0 and np.max(images) == 255

    with TFRecordExporter(tfrecord_dir, num_images) as tfr:
        rnd = np.random.RandomState(random_seed)
        for _idx in range(num_images):
            tfr.add_image(images[rnd.randint(images.shape[0], size=3)])

#----------------------------------------------------------------------------

def create_cifar10(tfrecord_dir, cifar10_dir, ignore_labels):
    print('Loading CIFAR-10 from "%s"' % cifar10_dir)
    import pickle
    images = []
    labels = []
    for batch in range(1, 6):
        with open(os.path.join(cifar10_dir, 'data_batch_%d' % batch), 'rb') as file:
            data = pickle.load(file, encoding='latin1')
        images.append(data['data'].reshape(-1, 3, 32, 32))
        labels.append(data['labels'])
    images = np.concatenate(images)
    labels = np.concatenate(labels)
    assert ignore_labels in [0, 1]
    assert images.shape == (50000, 3, 32, 32) and images.dtype == np.uint8
    assert labels.shape == (50000,) and labels.dtype in [np.int32, np.int64]
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9
    onehot = np.zeros((labels.size, np.max(labels) + 1), dtype=np.float32)
    onehot[np.arange(labels.size), labels] = 1.0

    with TFRecordExporter(tfrecord_dir, images.shape[0]) as tfr:
        order = tfr.choose_shuffled_order()
        for idx in range(order.size):
            tfr.add_image(images[order[idx]])
        if not ignore_labels:
            tfr.add_labels(onehot[order])

#----------------------------------------------------------------------------

def create_cifar100(tfrecord_dir, cifar100_dir):
    print('Loading CIFAR-100 from "%s"' % cifar100_dir)
    import pickle
    with open(os.path.join(cifar100_dir, 'train'), 'rb') as file:
        data = pickle.load(file, encoding='latin1')
    images = data['data'].reshape(-1, 3, 32, 32)
    labels = np.array(data['fine_labels'])
    assert images.shape == (50000, 3, 32, 32) and images.dtype == np.uint8
    assert labels.shape == (50000,) and labels.dtype == np.int32
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 99
    onehot = np.zeros((labels.size, np.max(labels) + 1), dtype=np.float32)
    onehot[np.arange(labels.size), labels] = 1.0

    with TFRecordExporter(tfrecord_dir, images.shape[0]) as tfr:
        order = tfr.choose_shuffled_order()
        for idx in range(order.size):
            tfr.add_image(images[order[idx]])
        tfr.add_labels(onehot[order])

#----------------------------------------------------------------------------

def create_svhn(tfrecord_dir, svhn_dir):
    print('Loading SVHN from "%s"' % svhn_dir)
    import pickle
    images = []
    labels = []
    for batch in range(1, 4):
        with open(os.path.join(svhn_dir, 'train_%d.pkl' % batch), 'rb') as file:
            data = pickle.load(file, encoding='latin1')
        images.append(data[0])
        labels.append(data[1])
    images = np.concatenate(images)
    labels = np.concatenate(labels)
    assert images.shape == (73257, 3, 32, 32) and images.dtype == np.uint8
    assert labels.shape == (73257,) and labels.dtype == np.uint8
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9
    onehot = np.zeros((labels.size, np.max(labels) + 1), dtype=np.float32)
    onehot[np.arange(labels.size), labels] = 1.0

    with TFRecordExporter(tfrecord_dir, images.shape[0]) as tfr:
        order = tfr.choose_shuffled_order()
        for idx in range(order.size):
            tfr.add_image(images[order[idx]])
        tfr.add_labels(onehot[order])

#----------------------------------------------------------------------------

def create_lsun(tfrecord_dir, lmdb_dir, resolution=256, max_images=None):
    print('Loading LSUN dataset from "%s"' % lmdb_dir)
    import lmdb # pip install lmdb # pylint: disable=import-error
    import cv2 # pip install opencv-python
    import io
    with lmdb.open(lmdb_dir, readonly=True, lock=False).begin(write=False) as txn:
        total_images = txn.stat()['entries']
        if max_images is None:
            max_images = total_images
        with TFRecordExporter(tfrecord_dir, max_images) as tfr:
            for _idx, (_key, value) in enumerate(txn.cursor()):
                try:
                    try:
                        img = cv2.imdecode(np.fromstring(value, dtype=np.uint8), 1)
                        if img is None:
                            raise IOError('cv2.imdecode failed')
                        img = img[:, :, ::-1] # BGR => RGB
                    except IOError:
                        img = np.asarray(PIL.Image.open(io.BytesIO(value)))
                    crop = np.min(img.shape[:2])
                    img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
                    img = PIL.Image.fromarray(img, 'RGB')
                    img = img.resize((resolution, resolution), PIL.Image.ANTIALIAS)
                    img = np.asarray(img)
                    img = img.transpose([2, 0, 1]) # HWC => CHW
                    tfr.add_image(img)
                except:
                    print(sys.exc_info()[1])
                if tfr.cur_images == max_images:
                    break

#----------------------------------------------------------------------------

def create_lsun_wide(tfrecord_dir, lmdb_dir, width=512, height=384, max_images=None):
    assert width == 2 ** int(np.round(np.log2(width)))
    assert height <= width
    print('Loading LSUN dataset from "%s"' % lmdb_dir)
    import lmdb # pip install lmdb # pylint: disable=import-error
    import cv2 # pip install opencv-python
    import io
    with lmdb.open(lmdb_dir, readonly=True).begin(write=False) as txn:
        total_images = txn.stat()['entries']
        if max_images is None:
            max_images = total_images
        with TFRecordExporter(tfrecord_dir, max_images, print_progress=False) as tfr:
            for idx, (_key, value) in enumerate(txn.cursor()):
                try:
                    try:
                        img = cv2.imdecode(np.fromstring(value, dtype=np.uint8), 1)
                        if img is None:
                            raise IOError('cv2.imdecode failed')
                        img = img[:, :, ::-1] # BGR => RGB
                    except IOError:
                        img = np.asarray(PIL.Image.open(io.BytesIO(value)))

                    ch = int(np.round(width * img.shape[0] / img.shape[1]))
                    if img.shape[1] < width or ch < height:
                        continue

                    img = img[(img.shape[0] - ch) // 2 : (img.shape[0] + ch) // 2]
                    img = PIL.Image.fromarray(img, 'RGB')
                    img = img.resize((width, height), PIL.Image.ANTIALIAS)
                    img = np.asarray(img)
                    img = img.transpose([2, 0, 1]) # HWC => CHW

                    canvas = np.zeros([3, width, width], dtype=np.uint8)
                    canvas[:, (width - height) // 2 : (width + height) // 2] = img
                    tfr.add_image(canvas)
                    print('\r%d / %d => %d ' % (idx + 1, total_images, tfr.cur_images), end='')

                except:
                    print(sys.exc_info()[1])
                if tfr.cur_images == max_images:
                    break
    print()

#----------------------------------------------------------------------------

def create_celeba(tfrecord_dir, celeba_dir, cx=89, cy=121):
    print('Loading CelebA from "%s"' % celeba_dir)
    glob_pattern = os.path.join(celeba_dir, 'img_align_celeba_png', '*.png')
    image_filenames = sorted(glob.glob(glob_pattern))
    expected_images = 202599
    if len(image_filenames) != expected_images:
        error('Expected to find %d images' % expected_images)

    with TFRecordExporter(tfrecord_dir, len(image_filenames)) as tfr:
        order = tfr.choose_shuffled_order()
        for idx in range(order.size):
            img = np.asarray(PIL.Image.open(image_filenames[order[idx]]))
            assert img.shape == (218, 178, 3)
            img = img[cy - 64 : cy + 64, cx - 64 : cx + 64]
            img = img.transpose(2, 0, 1) # HWC => CHW
            tfr.add_image(img)

#----------------------------------------------------------------------------

def create_from_images(tfrecord_dir, image_dir, shuffle):
    print('Loading images from "%s"' % image_dir)
    image_filenames = sorted(glob.glob(os.path.join(image_dir, '*')))
    if len(image_filenames) == 0:
        error('No input images found')

    img = np.asarray(PIL.Image.open(image_filenames[0]))
    resolution = img.shape[0]
    channels = img.shape[2] if img.ndim == 3 else 1
    if img.shape[1] != resolution:
        error('Input images must have the same width and height')
    if resolution != 2 ** int(np.floor(np.log2(resolution))):
        error('Input image resolution must be a power-of-two')
    if channels not in [1, 3]:
        error('Input images must be stored as RGB or grayscale')

    with TFRecordExporter(tfrecord_dir, len(image_filenames)) as tfr:
        order = tfr.choose_shuffled_order() if shuffle else np.arange(len(image_filenames))
        for idx in range(order.size):
            img = np.asarray(PIL.Image.open(image_filenames[order[idx]]))
            if channels == 1:
                img = img[np.newaxis, :, :] # HW => CHW
            else:
                img = img.transpose([2, 0, 1]) # HWC => CHW
            tfr.add_image(img)

#----------------------------------------------------------------------------

def create_from_hdf5(tfrecord_dir, hdf5_filename, shuffle):
    print('Loading HDF5 archive from "%s"' % hdf5_filename)
    import h5py # conda install h5py
    with h5py.File(hdf5_filename, 'r') as hdf5_file:
        hdf5_data = max([value for key, value in hdf5_file.items() if key.startswith('data')], key=lambda lod: lod.shape[3])
        with TFRecordExporter(tfrecord_dir, hdf5_data.shape[0]) as tfr:
            order = tfr.choose_shuffled_order() if shuffle else np.arange(hdf5_data.shape[0])
            for idx in range(order.size):
                tfr.add_image(hdf5_data[order[idx]])
            npy_filename = os.path.splitext(hdf5_filename)[0] + '-labels.npy'
            if os.path.isfile(npy_filename):
                tfr.add_labels(np.load(npy_filename)[order])

#----------------------------------------------------------------------------

def convert_to_hdf5(hdf5_filename, tfrecord_dir, compress):
    print('Loading dataset "%s"' % tfrecord_dir)
    tflib.init_tf()
    dset = dataset.TFRecordDataset(tfrecord_dir, max_label_size='full', repeat=False, shuffle=False)
    tflib.init_uninitialized_vars()
    with HDF5Exporter(hdf5_filename, resolution=dset.shape[1], channels=dset.shape[0], compress=compress) as h5:
        all_labels = []
        while True:
            images, labels = dset.get_minibatch_np(1)
            if images is None:
                break
            h5.add_images(images)
            all_labels.append(labels)
        all_labels = np.concatenate(all_labels)
        if all_labels.size:
            h5.add_labels(all_labels)

#----------------------------------------------------------------------------

def hdf5_from_images(hdf5_filename, image_dir, compress):
    print('Loading images from "%s"' % image_dir)
    image_filenames = sorted(glob.glob(os.path.join(image_dir, '*')))
    if len(image_filenames) == 0:
        error('No input images found')

    img = np.asarray(PIL.Image.open(image_filenames[0]))
    resolution = img.shape[0]
    channels = img.shape[2] if img.ndim == 3 else 1
    if img.shape[1] != resolution:
        error('Input images must have the same width and height')
    if resolution != 2 ** int(np.floor(np.log2(resolution))):
        error('Input image resolution must be a power-of-two')
    if channels not in [1, 3]:
        error('Input images must be stored as RGB or grayscale')

    with HDF5Exporter(hdf5_filename, resolution=resolution, channels=channels, compress=compress, expected_images=len(image_filenames)) as h5:
        for image_filename in image_filenames:
            img = np.asarray(PIL.Image.open(image_filename))
            if channels == 1:
                img = img[np.newaxis, :, :] # HW => CHW
            else:
                img = img.transpose([2, 0, 1]) # HWC => CHW
            h5.add_image(img)

#----------------------------------------------------------------------------

def make_png_path(outdir, idx):
    idx_str = f'{idx:08d}'
    return f'{os.path.join(outdir, idx_str[:5])}/img{idx_str}.png'

def unpack(tfrecord_dir, output_dir, resolution=None):
    print('Loading dataset "%s"' % tfrecord_dir)
    tflib.init_tf()
    dset = dataset.TFRecordDataset(tfrecord_dir, max_label_size='full', repeat=False, shuffle=False)
    tflib.init_uninitialized_vars()

    print('Extracting images to "%s"' % output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    idx = 0
    labels = []
    while True:
        if idx % 10 == 0:
            print('%d\r' % idx, end='', flush=True)
        images, lbls = dset.get_minibatch_np(1)
        if images is None:
            break
        if images.shape[1] == 1:
            img = PIL.Image.fromarray(images[0][0], 'L')
        else:
            img = PIL.Image.fromarray(images[0].transpose(1, 2, 0), 'RGB')
        if resolution is not None:
            img = img.resize((resolution, resolution), PIL.Image.ANTIALIAS)
        assert lbls.shape[0] == 1
        labels.append(lbls[0])
        png_fname = make_png_path(output_dir, idx)
        os.makedirs(os.path.dirname(png_fname), exist_ok=True)
        img.save(png_fname)
        idx += 1
    np.savez(f'{output_dir}/pack_extras.npz', labels=np.array(labels, dtype=np.uint8), num_images=idx)
    print('Extracted %d images.' % idx)

#----------------------------------------------------------------------------

def pack(unpacked_dir, tfrecord_dir, num_train=None, num_validation=None, mirror=0, seed=None):

    def export_samples(source_idx, tfr_prefix):
        if source_idx.shape[0] == 0: return
        if source_idx.shape != (source_idx.shape[0], 2):
            assert len(source_idx.shape) == 1
            source_idx = np.stack([np.zeros(source_idx.shape[0], dtype=np.uint8), source_idx], axis=-1)
        with TFRecordExporter(tfrecord_dir, len(source_idx), tfr_prefix=tfr_prefix) as tfr:
            for mirror, idx in source_idx:
                img = np.asarray(PIL.Image.open(make_png_path(unpacked_dir, idx)))
                img = img.transpose([2, 0, 1]) # HWC => CHW
                if mirror != 0:
                    img = img[:, :, ::-1]
                tfr.add_image(img)
            tfr.add_labels(labels_onehot[source_idx[:,1]])

    print(f'Loading an unpacked dataset from "{unpacked_dir}"')

    meta = np.load(f'{unpacked_dir}/pack_extras.npz')
    num_images = int(meta['num_images'])
    labels_onehot = meta['labels']
    assert (labels_onehot.shape[0] == num_images) and (len(labels_onehot.shape) == 2)

    order = np.arange(num_images)
    if seed is not None:
        np.random.RandomState(seed).shuffle(order)

    # Size the training and validation sets based on command line args.
    #
    # If the training set size is not specified on the command line, use all
    # except what's set aside for the validation set.
    n_train = num_train if num_train is not None else num_images
    n_valid = num_validation
    if num_train is None:
        n_train -= n_valid
        assert n_train > 0
    assert (n_train + n_valid) <= num_images

    train_idx = order[0:n_train]
    valid_idx = order[n_train:n_train+n_valid]

    if mirror != 0:
        n = train_idx.shape[0]
        train_idx = np.concatenate([
            np.stack([np.zeros(n, dtype=np.uint8), train_idx], axis=-1),
            np.stack([np.ones(n, dtype=np.uint8), train_idx], axis=-1)
        ])
        if seed is not None:
            np.random.RandomState(seed).shuffle(train_idx)

    tfr = os.path.basename(tfrecord_dir)
    export_samples(train_idx, tfr_prefix=tfr)
    export_samples(valid_idx, tfr_prefix=f'validation-{tfr}')

#----------------------------------------------------------------------------

def extract_brecahad_crops(brecahad_dir, output_dir, cropsize=256):
    params = {
        256: { 'overlap': 0.0 },
        512: { 'overlap': 0.5 }
    }
    if cropsize not in params:
        print('--cropsize must be one of:', ', '.join(str(x) for x in params.keys()))
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    incr = int(cropsize*(1-params[cropsize]['overlap']))
    out_idx = 0
    for fname in tqdm(sorted(glob.glob(os.path.join(brecahad_dir, '*.tif')))):
        src = PIL.Image.open(fname)
        w, h = src.size
        for x in range(0, w-cropsize+1, incr):
            for y in range(0, h-cropsize+1, incr):
                cropimg = src.crop((x, y, x+cropsize, y+cropsize))
                cropimg.save(os.path.join(output_dir, f'{out_idx:04d}.png'))
                out_idx += 1
    print(f'Extracted {out_idx} image crops.')

#----------------------------------------------------------------------------

def execute_cmdline(argv):
    prog = argv[0]
    parser = argparse.ArgumentParser(
        prog        = prog,
        description = 'Tool for creating multi-resolution TFRecords datasets for StyleGAN and ProGAN.',
        epilog      = 'Type "%s <command> -h" for more information.' % prog)

    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True
    def add_command(cmd, desc, example=None):
        epilog = 'Example: %s %s' % (prog, example) if example is not None else None
        return subparsers.add_parser(cmd, description=desc, help=desc, epilog=epilog)

    p = add_command(    'info',             'Display general info about dataset.',
                                            'info datasets/mnist')
    p.add_argument(     'tfrecord_dir',     help='Directory containing dataset')

    p = add_command(    'display',          'Display images in dataset.',
                                            'display datasets/mnist')
    p.add_argument(     'tfrecord_dir',     help='Directory containing dataset')

    p = add_command(    'extract',          'Extract images from dataset.',
                                            'extract datasets/mnist mnist-images')
    p.add_argument(     'tfrecord_dir',     help='Directory containing dataset')
    p.add_argument(     'output_dir',       help='Directory to extract the images into')

    p = add_command(    'compare',          'Compare two datasets.',
                                            'compare datasets/mydataset datasets/mnist')
    p.add_argument(     'tfrecord_dir_a',   help='Directory containing first dataset')
    p.add_argument(     'tfrecord_dir_b',   help='Directory containing second dataset')
    p.add_argument(     '--ignore_labels',  help='Ignore labels (default: 0)', type=int, default=0)

    p = add_command(    'create_mnist',     'Create dataset for MNIST.',
                                            'create_mnist datasets/mnist ~/downloads/mnist')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'mnist_dir',        help='Directory containing MNIST')

    p = add_command(    'create_mnistrgb',  'Create dataset for MNIST-RGB.',
                                            'create_mnistrgb datasets/mnistrgb ~/downloads/mnist')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'mnist_dir',        help='Directory containing MNIST')
    p.add_argument(     '--num_images',     help='Number of composite images to create (default: 1000000)', type=int, default=1000000)
    p.add_argument(     '--random_seed',    help='Random seed (default: 123)', type=int, default=123)

    p = add_command(    'create_cifar10',   'Create dataset for CIFAR-10.',
                                            'create_cifar10 datasets/cifar10 ~/downloads/cifar10')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'cifar10_dir',      help='Directory containing CIFAR-10')
    p.add_argument(     '--ignore_labels',  help='Ignore labels (default: 0)', type=int, default=0)

    p = add_command(    'create_cifar100',  'Create dataset for CIFAR-100.',
                                            'create_cifar100 datasets/cifar100 ~/downloads/cifar100')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'cifar100_dir',     help='Directory containing CIFAR-100')

    p = add_command(    'create_svhn',      'Create dataset for SVHN.',
                                            'create_svhn datasets/svhn ~/downloads/svhn')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'svhn_dir',         help='Directory containing SVHN')

    p = add_command(    'create_lsun',      'Create dataset for single LSUN category.',
                                            'create_lsun datasets/lsun-car-100k ~/downloads/lsun/car_lmdb --resolution 256 --max_images 100000')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'lmdb_dir',         help='Directory containing LMDB database')
    p.add_argument(     '--resolution',     help='Output resolution (default: 256)', type=int, default=256)
    p.add_argument(     '--max_images',     help='Maximum number of images (default: none)', type=int, default=None)

    p = add_command(    'create_lsun_wide', 'Create LSUN dataset with non-square aspect ratio.',
                                            'create_lsun_wide datasets/lsun-car-512x384 ~/downloads/lsun/car_lmdb --width 512 --height 384')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'lmdb_dir',         help='Directory containing LMDB database')
    p.add_argument(     '--width',          help='Output width (default: 512)', type=int, default=512)
    p.add_argument(     '--height',         help='Output height (default: 384)', type=int, default=384)
    p.add_argument(     '--max_images',     help='Maximum number of images (default: none)', type=int, default=None)

    p = add_command(    'create_celeba',    'Create dataset for CelebA.',
                                            'create_celeba datasets/celeba ~/downloads/celeba')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'celeba_dir',       help='Directory containing CelebA')
    p.add_argument(     '--cx',             help='Center X coordinate (default: 89)', type=int, default=89)
    p.add_argument(     '--cy',             help='Center Y coordinate (default: 121)', type=int, default=121)

    p = add_command(    'create_from_images', 'Create dataset from a directory full of images.',
                                            'create_from_images datasets/mydataset myimagedir')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'image_dir',        help='Directory containing the images')
    p.add_argument(     '--shuffle',        help='Randomize image order (default: 1)', type=int, default=1)

    p = add_command(    'create_from_hdf5', 'Create dataset from legacy HDF5 archive.',
                                            'create_from_hdf5 datasets/celebahq ~/downloads/celeba-hq-1024x1024.h5')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'hdf5_filename',    help='HDF5 archive containing the images')
    p.add_argument(     '--shuffle',        help='Randomize image order (default: 1)', type=int, default=1)

    p = add_command(    'convert_to_hdf5',  'Convert dataset to legacy HDF5 archive.',
                                            'convert_to_hdf5 datasets/celebahq.h5 datasets/celebahq')
    p.add_argument(     'hdf5_filename',    help='HDF5 archive to be created')
    p.add_argument(     'tfrecord_dir',     help='Dataset directory to load the images from')
    p.add_argument(     '--compress',       help='Compress the data (default: 0)', type=int, default=0)

    p = add_command(    'hdf5_from_images', 'Create HDF5 archive from a directory of images.',
                                            'hdf5_from_images datasets/mydataset.h5 myimagedir')
    p.add_argument(     'hdf5_filename',    help='HDF5 archive to be created')
    p.add_argument(     'image_dir',        help='Directory containing the images')
    p.add_argument(     '--compress',       help='Compress the data (default: 0)', type=int, default=0)

    p = add_command(    'unpack',           'Unpack a TFRecords dataset to labels and images for later repackaging with `pack`.')
    p.add_argument(     '--tfrecord_dir',   help='Directory containing the source dataset in TFRecords format', required=True)
    p.add_argument(     '--output_dir',     help='Output directory where to extract the dataset as PNG files', required=True)
    p.add_argument(     '--resolution',     help='Resize images to (resolution,resolution) (default: None = no resizing)', type=int, default=None)

    p = add_command(    'pack',             'Repackage an unpacked dataset into TFRecords.')
    p.add_argument(     '--unpacked_dir',   help='Source directory containing an unpacked tfrecords dataset')
    p.add_argument(     '--tfrecord_dir',   help='New dataset directory to be created')
    p.add_argument(     '--num_train',      help='Number of images to pick for the training set (default: None = all)', type=int, default=None)
    p.add_argument(     '--num_validation', help='Number of images to pick for the validation set (default: 0 = no images)', type=int, default=0)
    p.add_argument(     '--mirror',         help='Number of images to pick for the training set (default: 0 = no mirroring)', type=int, default=0)
    p.add_argument(     '--seed',           help='Shuffle random seed.  (default: None = do not shuffle)', type=int, default=None)

    p = add_command(    'extract_brecahad_crops', 'Extract crops from the original BreCaHAD images')
    p.add_argument(     '--brecahad_dir',   help='Source directory for BreCaHAD images.  Should contain .tif files.', required=True)
    p.add_argument(     '--output_dir',     help='Output directory for image crops.  Will contain .png files', required=True)
    p.add_argument(     '--cropsize',       help='Crop size (resolution,resolution)', type=int, default=256)

    args = parser.parse_args(argv[1:] if len(argv) > 1 else ['-h'])
    func = globals()[args.command]
    del args.command
    func(**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    execute_cmdline(sys.argv)

#----------------------------------------------------------------------------
