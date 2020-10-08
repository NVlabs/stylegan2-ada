# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Adaptive discriminator augmentation (ADA) from the paper
"Training Generative Adversarial Networks with Limited Data"."""

import numpy as np
import tensorflow as tf
import scipy.signal
import dnnlib
import dnnlib.tflib as tflib

from training import loss

#----------------------------------------------------------------------------
# Main class for adaptive discriminator augmentation (ADA).
# - Performs adaptive tuning of augmentation strength during training.
# - Acts as a wrapper for the augmentation pipeline.
# - Keeps track of the necessary training statistics.
# - Calculates statistics for the validation set, if available.

class AdaptiveAugment:
    def __init__(self,
        apply_func       = None,    # Function representing the augmentation pipeline. Can be a fully-qualified name, a function object, or None.
        apply_args       = {},      # Keyword arguments for the augmentation pipeline.
        initial_strength = 0,       # Augmentation strength (p) to use initially.
        tune_heuristic   = None,    # Heuristic for tuning the augmentation strength dynamically: 'rt', 'rv', None.
        tune_target      = None,    # Target value for the selected heuristic.
        tune_kimg        = 500,     # Adjustment speed, measured in how many kimg it takes for the strength to increase/decrease by one unit.
        stat_decay_kimg  = 0,       # Exponential moving average to use for training statistics, measured as the half-life in kimg. 0 = disable EMA.
    ):
        tune_stats = {
            'rt': {'Loss/signs/real'},
            'rv': {'Loss/scores/fake', 'Loss/scores/real', 'Loss/scores/valid'},
            None: {},
        }
        assert tune_heuristic in tune_stats
        assert apply_func is None or isinstance(apply_func, str) or dnnlib.util.is_top_level_function(apply_func)

        # Configuration.
        self.apply_func       = dnnlib.util.get_obj_by_name(apply_func) if isinstance(apply_func, str) else apply_func
        self.apply_args       = apply_args
        self.strength         = initial_strength
        self.tune_heuristic   = tune_heuristic
        self.tune_target      = tune_target
        self.tune_kimg        = tune_kimg
        self.stat_decay_kimg  = stat_decay_kimg

        # Runtime state.
        self._tune_stats      = tune_stats[tune_heuristic]
        self._strength_var    = None
        self._acc_vars        = dict() # {name: [var, ...], ...}
        self._acc_decay_in    = None
        self._acc_decay_ops   = dict() # {name: op, ...}
        self._valid_images    = None
        self._valid_labels    = None
        self._valid_images_in = None
        self._valid_labels_in = None
        self._valid_op        = None
        self._valid_ofs       = 0

    def init_validation_set(self, D_gpus, training_set):
        assert self._valid_images is None
        images, labels = training_set.load_validation_set_np()
        if images.shape[0] == 0:
            return
        self._valid_images = images
        self._valid_labels = labels

        # Build validation graph.
        with tflib.absolute_name_scope('Validation'), tf.control_dependencies(None):
            with tf.device('/cpu:0'):
                self._valid_images_in = tf.placeholder(training_set.dtype, name='valid_images_in', shape=[None]+training_set.shape)
                self._valid_labels_in = tf.placeholder(training_set.label_dtype, name='valid_labels_in', shape=[None,training_set.label_size])
                images_in_gpus = tf.split(self._valid_images_in, len(D_gpus))
                labels_in_gpus = tf.split(self._valid_labels_in, len(D_gpus))
            ops = []
            for gpu, (D_gpu, images_in_gpu, labels_in_gpu) in enumerate(zip(D_gpus, images_in_gpus, labels_in_gpus)):
                with tf.device(f'/gpu:{gpu}'):
                    images_expr = tf.cast(images_in_gpu, tf.float32) * (2 / 255) - 1
                    D_valid = loss.eval_D(D_gpu, self, images_expr, labels_in_gpu, report='valid')
                    ops += [D_valid.scores]
                self._valid_op = tf.group(*ops)

    def apply(self, images, labels, enable=True):
        if not enable or self.apply_func is None or (self.strength == 0 and self.tune_heuristic is None):
            return images, labels
        with tf.name_scope('Augment'):
            images, labels = self.apply_func(images, labels, strength=self.get_strength_var(), **self.apply_args)
        return images, labels

    def get_strength_var(self):
        if self._strength_var is None:
            with tflib.absolute_name_scope('Augment'), tf.control_dependencies(None):
                self._strength_var = tf.Variable(np.float32(self.strength), name='strength', trainable=False)
        return self._strength_var

    def report_stat(self, name, expr):
        if name in self._tune_stats:
            expr = self._increment_acc(name, expr)
        return expr

    def tune(self, nimg_delta):
        acc = {name: self._read_and_decay_acc(name, nimg_delta) for name in self._tune_stats}
        nimg_ratio = nimg_delta / (self.tune_kimg * 1000)
        strength = self.strength

        if self.tune_heuristic == 'rt':
            assert self.tune_target is not None
            rt = acc['Loss/signs/real']
            strength += nimg_ratio * np.sign(rt - self.tune_target)

        if self.tune_heuristic == 'rv':
            assert self.tune_target is not None
            assert self._valid_images is not None
            rv = (acc['Loss/scores/real'] - acc['Loss/scores/valid']) / max(acc['Loss/scores/real'] - acc['Loss/scores/fake'], 1e-8)
            strength += nimg_ratio * np.sign(rv - self.tune_target)

        self._set_strength(strength)

    def run_validation(self, minibatch_size):
        if self._valid_images is not None:
            indices = [(self._valid_ofs + i) % self._valid_images.shape[0] for i in range(minibatch_size)]
            tflib.run(self._valid_op, {self._valid_images_in: self._valid_images[indices], self._valid_labels_in: self._valid_labels[indices]})
            self._valid_ofs += len(indices)

    def _set_strength(self, strength):
        strength = max(strength, 0)
        if self._strength_var is not None and strength != self.strength:
            tflib.set_vars({self._strength_var: strength})
        self.strength = strength

    def _increment_acc(self, name, expr):
        with tf.name_scope('acc_' + name):
            with tf.control_dependencies(None):
                acc_var = tf.Variable(tf.zeros(2), name=name, trainable=False) # [acc_num, acc_sum]
            if name not in self._acc_vars:
                self._acc_vars[name] = []
            self._acc_vars[name].append(acc_var)
            expr_num = tf.shape(tf.reshape(expr, [-1]))[0]
            expr_sum = tf.reduce_sum(expr)
            acc_op = tf.assign_add(acc_var, [expr_num, expr_sum])
            with tf.control_dependencies([acc_op]):
                return tf.identity(expr)

    def _read_and_decay_acc(self, name, nimg_delta):
        acc_vars = self._acc_vars[name]
        acc_num, acc_sum = tuple(np.sum(tflib.run(acc_vars), axis=0))
        if nimg_delta > 0:
            with tflib.absolute_name_scope('Augment'), tf.control_dependencies(None):
                if self._acc_decay_in is None:
                    self._acc_decay_in = tf.placeholder(tf.float32, name='acc_decay_in', shape=[])
                if name not in self._acc_decay_ops:
                    with tf.name_scope('acc_' + name):
                        ops = [tf.assign(var, var * self._acc_decay_in) for var in acc_vars]
                        self._acc_decay_ops[name] = tf.group(*ops)
            acc_decay = 0.5 ** (nimg_delta / (self.stat_decay_kimg * 1000)) if self.stat_decay_kimg > 0 else 0
            tflib.run(self._acc_decay_ops[name], {self._acc_decay_in: acc_decay})
        return acc_sum / acc_num if acc_num > 0 else 0

#----------------------------------------------------------------------------
# Helper for randomly gating augmentation parameters based on the given probability.

def gate_augment_params(probability, params, disabled_val):
    shape = tf.shape(params)
    cond = (tf.random_uniform(shape[:1], 0, 1) < probability)
    disabled_val = tf.broadcast_to(tf.convert_to_tensor(disabled_val, dtype=params.dtype), shape)
    return tf.where(cond, params, disabled_val)

#----------------------------------------------------------------------------
# Helpers for constructing batched transformation matrices.

def construct_batch_of_matrices(*rows):
    rows = [[tf.convert_to_tensor(x, dtype=tf.float32) for x in r] for r in rows]
    batch_elems = [x for r in rows for x in r if x.shape.rank != 0]
    assert all(x.shape.rank == 1 for x in batch_elems)
    batch_size = tf.shape(batch_elems[0])[0] if len(batch_elems) else 1
    rows = [[tf.broadcast_to(x, [batch_size]) for x in r] for r in rows]
    return tf.transpose(rows, [2, 0, 1])

def translate_2d(tx, ty):
    return construct_batch_of_matrices(
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1])

def translate_3d(tx, ty, tz):
    return construct_batch_of_matrices(
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1])

def scale_2d(sx, sy):
    return construct_batch_of_matrices(
        [sx, 0,  0],
        [0,  sy, 0],
        [0,  0,  1])

def scale_3d(sx, sy, sz):
    return construct_batch_of_matrices(
        [sx, 0,  0,  0],
        [0,  sy, 0,  0],
        [0,  0,  sz, 0],
        [0,  0,  0,  1])

def rotate_2d(theta):
    return construct_batch_of_matrices(
        [tf.cos(theta), tf.sin(-theta), 0],
        [tf.sin(theta), tf.cos(theta),  0],
        [0,             0,              1])

def rotate_3d(v, theta):
    vx = v[..., 0]; vy = v[..., 1]; vz = v[..., 2]
    s = tf.sin(theta); c = tf.cos(theta); cc = 1 - c
    return construct_batch_of_matrices(
        [vx*vx*cc+c,    vx*vy*cc-vz*s, vx*vz*cc+vy*s, 0],
        [vy*vx*cc+vz*s, vy*vy*cc+c,    vy*vz*cc-vx*s, 0],
        [vz*vx*cc-vy*s, vz*vy*cc+vx*s, vz*vz*cc+c,    0],
        [0,             0,             0,             1])

def translate_2d_inv(tx, ty):
    return translate_2d(-tx, -ty)

def scale_2d_inv(sx, sy):
    return scale_2d(1/sx, 1/sy)

def rotate_2d_inv(theta):
    return rotate_2d(-theta)

#----------------------------------------------------------------------------
# Coefficients of various wavelet decomposition low-pass filters.

wavelets = {
    'haar': [0.7071067811865476, 0.7071067811865476],
    'db1':  [0.7071067811865476, 0.7071067811865476],
    'db2':  [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025],
    'db3':  [0.035226291882100656, -0.08544127388224149, -0.13501102001039084, 0.4598775021193313, 0.8068915093133388, 0.3326705529509569],
    'db4':  [-0.010597401784997278, 0.032883011666982945, 0.030841381835986965, -0.18703481171888114, -0.02798376941698385, 0.6308807679295904, 0.7148465705525415, 0.23037781330885523],
    'db5':  [0.003335725285001549, -0.012580751999015526, -0.006241490213011705, 0.07757149384006515, -0.03224486958502952, -0.24229488706619015, 0.13842814590110342, 0.7243085284385744, 0.6038292697974729, 0.160102397974125],
    'db6':  [-0.00107730108499558, 0.004777257511010651, 0.0005538422009938016, -0.031582039318031156, 0.02752286553001629, 0.09750160558707936, -0.12976686756709563, -0.22626469396516913, 0.3152503517092432, 0.7511339080215775, 0.4946238903983854, 0.11154074335008017],
    'db7':  [0.0003537138000010399, -0.0018016407039998328, 0.00042957797300470274, 0.012550998556013784, -0.01657454163101562, -0.03802993693503463, 0.0806126091510659, 0.07130921926705004, -0.22403618499416572, -0.14390600392910627, 0.4697822874053586, 0.7291320908465551, 0.39653931948230575, 0.07785205408506236],
    'db8':  [-0.00011747678400228192, 0.0006754494059985568, -0.0003917403729959771, -0.00487035299301066, 0.008746094047015655, 0.013981027917015516, -0.04408825393106472, -0.01736930100202211, 0.128747426620186, 0.00047248457399797254, -0.2840155429624281, -0.015829105256023893, 0.5853546836548691, 0.6756307362980128, 0.3128715909144659, 0.05441584224308161],
    'sym2': [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025],
    'sym3': [0.035226291882100656, -0.08544127388224149, -0.13501102001039084, 0.4598775021193313, 0.8068915093133388, 0.3326705529509569],
    'sym4': [-0.07576571478927333, -0.02963552764599851, 0.49761866763201545, 0.8037387518059161, 0.29785779560527736, -0.09921954357684722, -0.012603967262037833, 0.0322231006040427],
    'sym5': [0.027333068345077982, 0.029519490925774643, -0.039134249302383094, 0.1993975339773936, 0.7234076904024206, 0.6339789634582119, 0.01660210576452232, -0.17532808990845047, -0.021101834024758855, 0.019538882735286728],
    'sym6': [0.015404109327027373, 0.0034907120842174702, -0.11799011114819057, -0.048311742585633, 0.4910559419267466, 0.787641141030194, 0.3379294217276218, -0.07263752278646252, -0.021060292512300564, 0.04472490177066578, 0.0017677118642428036, -0.007800708325034148],
    'sym7': [0.002681814568257878, -0.0010473848886829163, -0.01263630340325193, 0.03051551316596357, 0.0678926935013727, -0.049552834937127255, 0.017441255086855827, 0.5361019170917628, 0.767764317003164, 0.2886296317515146, -0.14004724044296152, -0.10780823770381774, 0.004010244871533663, 0.010268176708511255],
    'sym8': [-0.0033824159510061256, -0.0005421323317911481, 0.03169508781149298, 0.007607487324917605, -0.1432942383508097, -0.061273359067658524, 0.4813596512583722, 0.7771857517005235, 0.3644418948353314, -0.05194583810770904, -0.027219029917056003, 0.049137179673607506, 0.003808752013890615, -0.01495225833704823, -0.0003029205147213668, 0.0018899503327594609],
}

#----------------------------------------------------------------------------
# Versatile image augmentation pipeline from the paper
# "Training Generative Adversarial Networks with Limited Data".
#
# All augmentations are disabled by default; individual augmentations can
# be enabled by setting their probability multipliers to 1.

def augment_pipeline(
    images,                         # Input images: NCHW, float32, dynamic range [-1,+1].
    labels,                         # Input labels.
    strength         = 1,           # Overall multiplier for augmentation probability; can be a Tensor.
    debug_percentile = None,        # Percentile value for visualizing parameter ranges; None = normal operation.

    # Pixel blitting.
    xflip            = 0,           # Probability multiplier for x-flip.
    rotate90         = 0,           # Probability multiplier for 90 degree rotations.
    xint             = 0,           # Probability multiplier for integer translation.
    xint_max         = 0.125,       # Range of integer translation, relative to image dimensions.

    # General geometric transformations.
    scale            = 0,           # Probability multiplier for isotropic scaling.
    rotate           = 0,           # Probability multiplier for arbitrary rotation.
    aniso            = 0,           # Probability multiplier for anisotropic scaling.
    xfrac            = 0,           # Probability multiplier for fractional translation.
    scale_std        = 0.2,         # Log2 standard deviation of isotropic scaling.
    rotate_max       = 1,           # Range of arbitrary rotation, 1 = full circle.
    aniso_std        = 0.2,         # Log2 standard deviation of anisotropic scaling.
    xfrac_std        = 0.125,       # Standard deviation of frational translation, relative to image dimensions.

    # Color transformations.
    brightness       = 0,           # Probability multiplier for brightness.
    contrast         = 0,           # Probability multiplier for contrast.
    lumaflip         = 0,           # Probability multiplier for luma flip.
    hue              = 0,           # Probability multiplier for hue rotation.
    saturation       = 0,           # Probability multiplier for saturation.
    brightness_std   = 0.2,         # Standard deviation of brightness.
    contrast_std     = 0.5,         # Log2 standard deviation of contrast.
    hue_max          = 1,           # Range of hue rotation, 1 = full circle.
    saturation_std   = 1,           # Log2 standard deviation of saturation.

    # Image-space filtering.
    imgfilter        = 0,           # Probability multiplier for image-space filtering.
    imgfilter_bands  = [1,1,1,1],   # Probability multipliers for individual frequency bands.
    imgfilter_std    = 1,           # Log2 standard deviation of image-space filter amplification.

    # Image-space corruptions.
    noise            = 0,           # Probability multiplier for additive RGB noise.
    cutout           = 0,           # Probability multiplier for cutout.
    noise_std        = 0.1,         # Standard deviation of additive RGB noise.
    cutout_size      = 0.5,         # Size of the cutout rectangle, relative to image dimensions.
):
    # Determine input shape.
    batch, channels, height, width = images.shape.as_list()
    if batch is None:
        batch = tf.shape(images)[0]

    # -------------------------------------
    # Select parameters for pixel blitting.
    # -------------------------------------

    # Initialize inverse homogeneous 2D transform: G_inv @ pixel_out ==> pixel_in
    I_3 = tf.eye(3, batch_shape=[batch])
    G_inv = I_3

    # Apply x-flip with probability (xflip * strength).
    if xflip > 0:
        i = tf.floor(tf.random_uniform([batch], 0, 2))
        i = gate_augment_params(xflip * strength, i, 0)
        if debug_percentile is not None:
            i = tf.floor(tf.broadcast_to(debug_percentile, [batch]) * 2)
        G_inv @= scale_2d_inv(1 - 2 * i, 1)

    # Apply 90 degree rotations with probability (rotate90 * strength).
    if rotate90 > 0:
        i = tf.floor(tf.random_uniform([batch], 0, 4))
        i = gate_augment_params(rotate90 * strength, i, 0)
        if debug_percentile is not None:
            i = tf.floor(tf.broadcast_to(debug_percentile, [batch]) * 4)
        G_inv @= rotate_2d_inv(-np.pi / 2 * i)

    # Apply integer translation with probability (xint * strength).
    if xint > 0:
        t = tf.random_uniform([batch, 2], -xint_max, xint_max)
        t = gate_augment_params(xint * strength, t, 0)
        if debug_percentile is not None:
            t = (tf.broadcast_to(debug_percentile, [batch, 2]) * 2 - 1) * xint_max
        G_inv @= translate_2d_inv(tf.rint(t[:,0] * width), tf.rint(t[:,1] * height))

    # --------------------------------------------------------
    # Select parameters for general geometric transformations.
    # --------------------------------------------------------

    # Apply isotropic scaling with probability (scale * strength).
    if scale > 0:
        s = 2 ** tf.random_normal([batch], 0, scale_std)
        s = gate_augment_params(scale * strength, s, 1)
        if debug_percentile is not None:
            s = 2 ** (tflib.erfinv(tf.broadcast_to(debug_percentile, [batch]) * 2 - 1) * scale_std)
        G_inv @= scale_2d_inv(s, s)

    # Apply pre-rotation with probability p_rot.
    p_rot = 1 - tf.sqrt(tf.cast(tf.maximum(1 - rotate * strength, 0), tf.float32)) # P(pre OR post) = p
    if rotate > 0:
        theta = tf.random_uniform([batch], -np.pi * rotate_max, np.pi * rotate_max)
        theta = gate_augment_params(p_rot, theta, 0)
        if debug_percentile is not None:
            theta = (tf.broadcast_to(debug_percentile, [batch]) * 2 - 1) * np.pi * rotate_max
        G_inv @= rotate_2d_inv(-theta) # Before anisotropic scaling.

    # Apply anisotropic scaling with probability (aniso * strength).
    if aniso > 0:
        s = 2 ** tf.random_normal([batch], 0, aniso_std)
        s = gate_augment_params(aniso * strength, s, 1)
        if debug_percentile is not None:
            s = 2 ** (tflib.erfinv(tf.broadcast_to(debug_percentile, [batch]) * 2 - 1) * aniso_std)
        G_inv @= scale_2d_inv(s, 1 / s)

    # Apply post-rotation with probability p_rot.
    if rotate > 0:
        theta = tf.random_uniform([batch], -np.pi * rotate_max, np.pi * rotate_max)
        theta = gate_augment_params(p_rot, theta, 0)
        if debug_percentile is not None:
            theta = tf.zeros([batch])
        G_inv @= rotate_2d_inv(-theta) # After anisotropic scaling.

    # Apply fractional translation with probability (xfrac * strength).
    if xfrac > 0:
        t = tf.random_normal([batch, 2], 0, xfrac_std)
        t = gate_augment_params(xfrac * strength, t, 0)
        if debug_percentile is not None:
            t = tflib.erfinv(tf.broadcast_to(debug_percentile, [batch, 2]) * 2 - 1) * xfrac_std
        G_inv @= translate_2d_inv(t[:,0] * width, t[:,1] * height)

    # ----------------------------------
    # Execute geometric transformations.
    # ----------------------------------

    # Execute if the transform is not identity.
    if G_inv is not I_3:

        # Setup orthogonal lowpass filter.
        Hz = wavelets['sym6']
        Hz = np.asarray(Hz, dtype=np.float32)
        Hz = np.reshape(Hz, [-1, 1, 1]).repeat(channels, axis=1) # [tap, channel, 1]
        Hz_pad = Hz.shape[0] // 4

        # Calculate padding.
        cx = (width - 1) / 2
        cy = (height - 1) / 2
        cp = np.transpose([[-cx, -cy, 1], [cx, -cy, 1], [cx, cy, 1], [-cx, cy, 1]]) # [xyz, idx]
        cp = G_inv @ cp[np.newaxis] # [batch, xyz, idx]
        cp = cp[:, :2, :] # [batch, xy, idx]
        m_lo = tf.ceil(tf.reduce_max(-cp, axis=[0,2]) - [cx, cy] + Hz_pad * 2)
        m_hi = tf.ceil(tf.reduce_max( cp, axis=[0,2]) - [cx, cy] + Hz_pad * 2)
        m_lo = tf.clip_by_value(m_lo, [0, 0], [width-1, height-1])
        m_hi = tf.clip_by_value(m_hi, [0, 0], [width-1, height-1])

        # Pad image and adjust origin.
        images = tf.transpose(images, [0, 2, 3, 1]) # NCHW => NHWC
        pad = [[0, 0], [m_lo[1], m_hi[1]], [m_lo[0], m_hi[0]], [0, 0]]
        images = tf.pad(tensor=images, paddings=pad, mode='REFLECT')
        T_in = translate_2d(cx + m_lo[0], cy + m_lo[1])
        T_out = translate_2d_inv(cx + Hz_pad, cy + Hz_pad)
        G_inv = T_in @ G_inv @ T_out

        # Upsample.
        shape = [batch, tf.shape(images)[1] * 2, tf.shape(images)[2] * 2, channels]
        images = tf.nn.depthwise_conv2d_backprop_input(input_sizes=shape, filter=Hz[np.newaxis, :], out_backprop=images, strides=[1,2,2,1], padding='SAME', data_format='NHWC')
        images = tf.nn.depthwise_conv2d_backprop_input(input_sizes=shape, filter=Hz[:, np.newaxis], out_backprop=images, strides=[1,1,1,1], padding='SAME', data_format='NHWC')
        G_inv = scale_2d(2, 2) @ G_inv @ scale_2d_inv(2, 2) # Account for the increased resolution.

        # Execute transformation.
        transforms = tf.reshape(G_inv, [-1, 9])[:, :8]
        shape = [(height + Hz_pad * 2) * 2, (width + Hz_pad * 2) * 2]
        images = tf.contrib.image.transform(images=images, transforms=transforms, output_shape=shape, interpolation='BILINEAR')

        # Downsample and crop.
        images = tf.nn.depthwise_conv2d(input=images, filter=Hz[np.newaxis,:], strides=[1,1,1,1], padding='SAME', data_format='NHWC')
        images = tf.nn.depthwise_conv2d(input=images, filter=Hz[:,np.newaxis], strides=[1,2,2,1], padding='SAME', data_format='NHWC')
        images = images[:, Hz_pad : height + Hz_pad, Hz_pad : width + Hz_pad, :]
        images = tf.transpose(images, [0, 3, 1, 2]) # NHWC => NCHW

    # --------------------------------------------
    # Select parameters for color transformations.
    # --------------------------------------------

    # Initialize homogeneous 3D transformation matrix: C @ color_in ==> color_out
    I_4 = tf.eye(4, batch_shape=[batch])
    C = I_4

    # Apply brightness with probability (brightness * strength).
    if brightness > 0:
        b = tf.random_normal([batch], 0, brightness_std)
        b = gate_augment_params(brightness * strength, b, 0)
        if debug_percentile is not None:
            b = tflib.erfinv(tf.broadcast_to(debug_percentile, [batch]) * 2 - 1) * brightness_std
        C = translate_3d(b, b, b) @ C

    # Apply contrast with probability (contrast * strength).
    if contrast > 0:
        c = 2 ** tf.random_normal([batch], 0, contrast_std)
        c = gate_augment_params(contrast * strength, c, 1)
        if debug_percentile is not None:
            c = 2 ** (tflib.erfinv(tf.broadcast_to(debug_percentile, [batch]) * 2 - 1) * contrast_std)
        C = scale_3d(c, c, c) @ C

    # Apply luma flip with probability (lumaflip * strength).
    v = np.array([1, 1, 1, 0]) / np.sqrt(3) # Luma axis.
    if lumaflip > 0:
        i = tf.floor(tf.random_uniform([batch], 0, 2))
        i = gate_augment_params(lumaflip * strength, i, 0)
        if debug_percentile is not None:
            i = tf.floor(tf.broadcast_to(debug_percentile, [batch]) * 2)
        i = tf.reshape(i, [batch, 1, 1])
        C = (I_4 - 2 * np.outer(v, v) * i) @ C # Householder reflection.

    # Apply hue rotation with probability (hue * strength).
    if hue > 0 and channels > 1:
        theta = tf.random_uniform([batch], -np.pi * hue_max, np.pi * hue_max)
        theta = gate_augment_params(hue * strength, theta, 0)
        if debug_percentile is not None:
            theta = (tf.broadcast_to(debug_percentile, [batch]) * 2 - 1) * np.pi * hue_max
        C = rotate_3d(v, theta) @ C # Rotate around v.

    # Apply saturation with probability (saturation * strength).
    if saturation > 0 and channels > 1:
        s = 2 ** tf.random_normal([batch], 0, saturation_std)
        s = gate_augment_params(saturation * strength, s, 1)
        if debug_percentile is not None:
            s = 2 ** (tflib.erfinv(tf.broadcast_to(debug_percentile, [batch]) * 2 - 1) * saturation_std)
        s = tf.reshape(s, [batch, 1, 1])
        C = (np.outer(v, v) + (I_4 - np.outer(v, v)) * s) @ C

    # ------------------------------
    # Execute color transformations.
    # ------------------------------

    # Execute if the transform is not identity.
    if C is not I_4:
        images = tf.reshape(images, [batch, channels, height * width])
        if channels == 3:
            images = C[:, :3, :3] @ images + C[:, :3, 3:]
        elif channels == 1:
            C = tf.reduce_mean(C[:, :3, :], axis=1, keepdims=True)
            images = images * tf.reduce_sum(C[:, :, :3], axis=2, keepdims=True) + C[:, :, 3:]
        else:
            raise ValueError('Image must be RGB (3 channels) or L (1 channel)')
        images = tf.reshape(images, [batch, channels, height, width])

    # ----------------------
    # Image-space filtering.
    # ----------------------

    if imgfilter > 0:
        num_bands = 4
        assert len(imgfilter_bands) == num_bands
        expected_power = np.array([10, 1, 1, 1]) / 13 # Expected power spectrum (1/f).

        # Apply amplification for each band with probability (imgfilter * strength * band_strength).
        g = tf.ones([batch, num_bands]) # Global gain vector (identity).
        for i, band_strength in enumerate(imgfilter_bands):
            t_i = 2 ** tf.random_normal([batch], 0, imgfilter_std)
            t_i = gate_augment_params(imgfilter * strength * band_strength, t_i, 1)
            if debug_percentile is not None:
                t_i = 2 ** (tflib.erfinv(tf.broadcast_to(debug_percentile, [batch]) * 2 - 1) * imgfilter_std) if band_strength > 0 else tf.ones([batch])
            t = tf.ones([batch, num_bands]) # Temporary gain vector.
            t = tf.concat([t[:, :i], t_i[:, np.newaxis], t[:, i+1:]], axis=-1) # Replace i'th element.
            t /= tf.sqrt(tf.reduce_sum(expected_power * tf.square(t), axis=-1, keepdims=True)) # Normalize power.
            g *= t # Accumulate into global gain.

        # Construct filter bank.
        Hz_lo = wavelets['sym2']
        Hz_lo = np.asarray(Hz_lo, dtype=np.float32)     # H(z)
        Hz_hi = Hz_lo * ((-1) ** np.arange(Hz_lo.size)) # H(-z)
        Hz_lo2 = np.convolve(Hz_lo, Hz_lo[::-1]) / 2    # H(z) * H(z^-1) / 2
        Hz_hi2 = np.convolve(Hz_hi, Hz_hi[::-1]) / 2    # H(-z) * H(-z^-1) / 2
        Hz_bands = np.eye(num_bands, 1)                 # Bandpass(H(z), b_i)
        for i in range(1, num_bands):
            Hz_bands = np.dstack([Hz_bands, np.zeros_like(Hz_bands)]).reshape(num_bands, -1)[:, :-1]
            Hz_bands = scipy.signal.convolve(Hz_bands, [Hz_lo2])
            Hz_bands[i, (Hz_bands.shape[1] - Hz_hi2.size) // 2 : (Hz_bands.shape[1] + Hz_hi2.size) // 2] += Hz_hi2

        # Construct combined amplification filter.
        Hz_prime = g @ Hz_bands # [batch, tap]
        Hz_prime = tf.transpose(Hz_prime) # [tap, batch]
        Hz_prime = tf.tile(Hz_prime[:, :, np.newaxis], [1, 1, channels]) # [tap, batch, channels]
        Hz_prime = tf.reshape(Hz_prime, [-1, batch * channels, 1]) # [tap, batch * channels, 1]

        # Apply filter.
        images = tf.reshape(images, [1, -1, height, width])
        pad = Hz_bands.shape[1] // 2
        pad = [[0,0], [0,0], [pad, pad], [pad, pad]]
        images = tf.pad(tensor=images, paddings=pad, mode='REFLECT')
        images = tf.nn.depthwise_conv2d(input=images, filter=Hz_prime[np.newaxis,:], strides=[1,1,1,1], padding='VALID', data_format='NCHW')
        images = tf.nn.depthwise_conv2d(input=images, filter=Hz_prime[:,np.newaxis], strides=[1,1,1,1], padding='VALID', data_format='NCHW')
        images = tf.reshape(images, [-1, channels, height, width])

    # ------------------------
    # Image-space corruptions.
    # ------------------------

    # Apply additive RGB noise with probability (noise * strength).
    if noise > 0:
        sigma = tf.abs(tf.random_normal([batch], 0, noise_std))
        sigma = gate_augment_params(noise * strength, sigma, 0)
        if debug_percentile is not None:
            sigma = tflib.erfinv(tf.broadcast_to(debug_percentile, [batch])) * noise_std
        sigma = tf.reshape(sigma, [-1, 1, 1, 1])
        images += tf.random_normal([batch, channels, height, width]) * sigma

    # Apply cutout with probability (cutout * strength).
    if cutout > 0:
        size = tf.fill([batch, 2], cutout_size)
        size = gate_augment_params(cutout * strength, size, 0)
        center = tf.random_uniform([batch, 2], 0, 1)
        if debug_percentile is not None:
            size = tf.fill([batch, 2], cutout_size)
            center = tf.broadcast_to(debug_percentile, [batch, 2])
        size = tf.reshape(size, [batch, 2, 1, 1, 1])
        center = tf.reshape(center, [batch, 2, 1, 1, 1])
        coord_x = tf.reshape(tf.range(width, dtype=tf.float32), [1, 1, 1, width])
        coord_y = tf.reshape(tf.range(height, dtype=tf.float32), [1, 1, height, 1])
        mask_x = (tf.abs((coord_x + 0.5) / width - center[:, 0]) >= size[:, 0] / 2)
        mask_y = (tf.abs((coord_y + 0.5) / height - center[:, 1]) >= size[:, 1] / 2)
        mask = tf.cast(tf.logical_or(mask_x, mask_y), tf.float32)
        images *= mask

    return images, labels

#----------------------------------------------------------------------------
