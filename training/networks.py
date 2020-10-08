# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Network architectures from the paper
"Training Generative Adversarial Networks with Limited Data"."""

import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib.ops.upfirdn_2d import upsample_2d, downsample_2d, upsample_conv_2d, conv_downsample_2d
from dnnlib.tflib.ops.fused_bias_act import fused_bias_act

# NOTE: Do not import any application-specific modules here!
# Specify all network parameters as kwargs.

#----------------------------------------------------------------------------
# Get/create weight tensor for convolution or fully-connected layer.

def get_weight(shape, gain=1, equalized_lr=True, lrmul=1, weight_var='weight', trainable=True, use_spectral_norm=False):
    fan_in = np.prod(shape[:-1]) # [kernel, kernel, fmaps_in, fmaps_out] for conv2d, [in, out] for fully-connected.
    he_std = gain / np.sqrt(fan_in) # He init.

    # Apply equalized learning rate from the paper
    # "Progressive Growing of GANs for Improved Quality, Stability, and Variation".
    if equalized_lr:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul

    # Create variable.
    init = tf.initializers.random_normal(0, init_std)
    w = tf.get_variable(weight_var, shape=shape, initializer=init, trainable=trainable) * runtime_coef
    if use_spectral_norm:
        w = apply_spectral_norm(w, state_var=weight_var+'_sn')
    return w

#----------------------------------------------------------------------------
# Bias and activation function.

def apply_bias_act(x, act='linear', gain=None, lrmul=1, clamp=None, bias_var='bias', trainable=True):
    b = tf.get_variable(bias_var, shape=[x.shape[1]], initializer=tf.initializers.zeros(), trainable=trainable) * lrmul
    return fused_bias_act(x, b=tf.cast(b, x.dtype), act=act, gain=gain, clamp=clamp)

#----------------------------------------------------------------------------
# Fully-connected layer.

def dense_layer(x, fmaps, lrmul=1, weight_var='weight', trainable=True, use_spectral_norm=False):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], lrmul=lrmul, weight_var=weight_var, trainable=trainable, use_spectral_norm=use_spectral_norm)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)

#----------------------------------------------------------------------------
# 2D convolution op with optional upsampling, downsampling, and padding.

def conv2d(x, w, up=False, down=False, resample_kernel=None, padding=0):
    assert not (up and down)
    kernel = w.shape[0].value
    assert w.shape[1].value == kernel
    assert kernel >= 1 and kernel % 2 == 1

    w = tf.cast(w, x.dtype)
    if up:
        x = upsample_conv_2d(x, w, data_format='NCHW', k=resample_kernel, padding=padding)
    elif down:
        x = conv_downsample_2d(x, w, data_format='NCHW', k=resample_kernel, padding=padding)
    else:
        padding_mode = {0: 'SAME', -(kernel // 2): 'VALID'}[padding]
        x = tf.nn.conv2d(x, w, data_format='NCHW', strides=[1,1,1,1], padding=padding_mode)
    return x

#----------------------------------------------------------------------------
# 2D convolution layer.

def conv2d_layer(x, fmaps, kernel, up=False, down=False, resample_kernel=None, lrmul=1, trainable=True, use_spectral_norm=False):
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], lrmul=lrmul, trainable=trainable, use_spectral_norm=use_spectral_norm)
    return conv2d(x, tf.cast(w, x.dtype), up=up, down=down, resample_kernel=resample_kernel)

#----------------------------------------------------------------------------
# Modulated 2D convolution layer from the paper
# "Analyzing and Improving Image Quality of StyleGAN".

def modulated_conv2d_layer(x, y, fmaps, kernel, up=False, down=False, demodulate=True, resample_kernel=None, lrmul=1, fused_modconv=False, trainable=True, use_spectral_norm=False):
    assert not (up and down)
    assert kernel >= 1 and kernel % 2 == 1

    # Get weight.
    wshape = [kernel, kernel, x.shape[1].value, fmaps]
    w = get_weight(wshape, lrmul=lrmul, trainable=trainable, use_spectral_norm=use_spectral_norm)
    if x.dtype.name == 'float16' and not fused_modconv and demodulate:
        w *= np.sqrt(1 / np.prod(wshape[:-1])) / tf.reduce_max(tf.abs(w), axis=[0,1,2]) # Pre-normalize to avoid float16 overflow.
    ww = w[np.newaxis] # [BkkIO] Introduce minibatch dimension.

    # Modulate.
    s = dense_layer(y, fmaps=x.shape[1].value, weight_var='mod_weight', trainable=trainable, use_spectral_norm=use_spectral_norm) # [BI] Transform incoming W to style.
    s = apply_bias_act(s, bias_var='mod_bias', trainable=trainable) + 1 # [BI] Add bias (initially 1).
    if x.dtype.name == 'float16' and not fused_modconv and demodulate:
        s *= 1 / tf.reduce_max(tf.abs(s)) # Pre-normalize to avoid float16 overflow.
    ww *= tf.cast(s[:, np.newaxis, np.newaxis, :, np.newaxis], w.dtype) # [BkkIO] Scale input feature maps.

    # Demodulate.
    if demodulate:
        d = tf.rsqrt(tf.reduce_sum(tf.square(ww), axis=[1,2,3]) + 1e-8) # [BO] Scaling factor.
        ww *= d[:, np.newaxis, np.newaxis, np.newaxis, :] # [BkkIO] Scale output feature maps.

    # Reshape/scale input.
    if fused_modconv:
        x = tf.reshape(x, [1, -1, x.shape[2], x.shape[3]]) # Fused => reshape minibatch to convolution groups.
        w = tf.reshape(tf.transpose(ww, [1, 2, 3, 0, 4]), [ww.shape[1], ww.shape[2], ww.shape[3], -1])
    else:
        x *= tf.cast(s[:, :, np.newaxis, np.newaxis], x.dtype) # [BIhw] Not fused => scale input activations.

    # 2D convolution.
    x = conv2d(x, tf.cast(w, x.dtype), up=up, down=down, resample_kernel=resample_kernel)

    # Reshape/scale output.
    if fused_modconv:
        x = tf.reshape(x, [-1, fmaps, x.shape[2], x.shape[3]]) # Fused => reshape convolution groups back to minibatch.
    elif demodulate:
        x *= tf.cast(d[:, :, np.newaxis, np.newaxis], x.dtype) # [BOhw] Not fused => scale output activations.
    return x

#----------------------------------------------------------------------------
# Normalize 2nd raw moment of the given activation tensor along specified axes.

def normalize_2nd_moment(x, axis=1, eps=1e-8):
    return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=axis, keepdims=True) + eps)

#----------------------------------------------------------------------------
# Minibatch standard deviation layer from the paper
# "Progressive Growing of GANs for Improved Quality, Stability, and Variation".

def minibatch_stddev_layer(x, group_size=None, num_new_features=1):
    if group_size is None:
        group_size = tf.shape(x)[0]
    else:
        group_size = tf.minimum(group_size, tf.shape(x)[0]) # Minibatch must be divisible by (or smaller than) group_size.

    G = group_size
    F = num_new_features
    _N, C, H, W = x.shape.as_list()
    c = C // F

    y = tf.cast(x, tf.float32)                # [NCHW]   Cast to FP32.
    y = tf.reshape(y, [G, -1, F, c, H, W])    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
    y -= tf.reduce_mean(y, axis=0)            # [GnFcHW] Subtract mean over group.
    y = tf.reduce_mean(tf.square(y), axis=0)  # [nFcHW]  Calc variance over group.
    y = tf.sqrt(y + 1e-8)                     # [nFcHW]  Calc stddev over group.
    y = tf.reduce_mean(y, axis=[2,3,4])       # [nF]     Take average over channels and pixels.
    y = tf.cast(y, x.dtype)                   # [nF]     Cast back to original data type.
    y = tf.reshape(y, [-1, F, 1, 1])          # [nF11]   Add missing dimensions.
    y = tf.tile(y, [G, 1, H, W])              # [NFHW]   Replicate over group and pixels.
    return tf.concat([x, y], axis=1)          # [NCHW]   Append to input as new channels.

#----------------------------------------------------------------------------
# Spectral normalization from the paper
# "Spectral Normalization for Generative Adversarial Networks".

def apply_spectral_norm(w, state_var='sn', iterations=1, eps=1e-8):
    fmaps = w.shape[-1].value
    w_mat = tf.reshape(w, [-1, fmaps])
    u_var = tf.get_variable(state_var, shape=[1,fmaps], initializer=tf.initializers.random_normal(), trainable=False)

    u = u_var
    for _ in range(iterations):
        v = tf.matmul(u, w_mat, transpose_b=True)
        v *= tf.rsqrt(tf.reduce_sum(tf.square(v)) + eps)
        u = tf.matmul(v, w_mat)
        sigma_inv = tf.rsqrt(tf.reduce_sum(tf.square(u)) + eps)
        u *= sigma_inv

    with tf.control_dependencies([tf.assign(u_var, u)]):
        return w * sigma_inv

#----------------------------------------------------------------------------
# Main generator network.
# Composed of two sub-networks (mapping and synthesis) that are defined below.

def G_main(
    latents_in,                                     # First input: Latent vectors (Z) [minibatch, latent_size].
    labels_in,                                      # Second input: Conditioning labels [minibatch, label_size].

    # Evaluation mode.
    is_training             = False,                # Network is under training? Enables and disables specific features.
    is_validation           = False,                # Network is under validation? Chooses which value to use for truncation_psi.
    return_dlatents         = False,                # Return dlatents (W) in addition to the images?

    # Truncation & style mixing.
    truncation_psi          = 0.5,                  # Style strength multiplier for the truncation trick. None = disable.
    truncation_cutoff       = None,                 # Number of layers for which to apply the truncation trick. None = disable.
    truncation_psi_val      = None,                 # Value for truncation_psi to use during validation.
    truncation_cutoff_val   = None,                 # Value for truncation_cutoff to use during validation.
    dlatent_avg_beta        = 0.995,                # Decay for tracking the moving average of W during training. None = disable.
    style_mixing_prob       = 0.9,                  # Probability of mixing styles during training. None = disable.

    # Sub-networks.
    components              = dnnlib.EasyDict(),    # Container for sub-networks. Retained between calls.
    mapping_func            = 'G_mapping',          # Build func name for the mapping network.
    synthesis_func          = 'G_synthesis',        # Build func name for the synthesis network.
    is_template_graph       = False,                # True = template graph constructed by the Network class, False = actual evaluation.

    **kwargs,                                       # Arguments for sub-networks (mapping and synthesis).
):
    # Validate arguments.
    assert not is_training or not is_validation
    assert isinstance(components, dnnlib.EasyDict)
    if is_validation:
        truncation_psi = truncation_psi_val
        truncation_cutoff = truncation_cutoff_val
    if is_training or (truncation_psi is not None and not tflib.is_tf_expression(truncation_psi) and truncation_psi == 1):
        truncation_psi = None
    if is_training:
        truncation_cutoff = None
    if not is_training or (dlatent_avg_beta is not None and not tflib.is_tf_expression(dlatent_avg_beta) and dlatent_avg_beta == 1):
        dlatent_avg_beta = None
    if not is_training or (style_mixing_prob is not None and not tflib.is_tf_expression(style_mixing_prob) and style_mixing_prob <= 0):
        style_mixing_prob = None

    # Setup components.
    if 'synthesis' not in components:
        components.synthesis = tflib.Network('G_synthesis', func_name=globals()[synthesis_func], **kwargs)
    num_layers = components.synthesis.input_shape[1]
    dlatent_size = components.synthesis.input_shape[2]
    if 'mapping' not in components:
        components.mapping = tflib.Network('G_mapping', func_name=globals()[mapping_func], dlatent_broadcast=num_layers, **kwargs)

    # Evaluate mapping network.
    dlatents = components.mapping.get_output_for(latents_in, labels_in, is_training=is_training, **kwargs)
    dlatents = tf.cast(dlatents, tf.float32)

    # Update moving average of W.
    dlatent_avg = tf.get_variable('dlatent_avg', shape=[dlatent_size], initializer=tf.initializers.zeros(), trainable=False)
    if dlatent_avg_beta is not None:
        with tf.variable_scope('DlatentAvg'):
            batch_avg = tf.reduce_mean(dlatents[:, 0], axis=0)
            update_op = tf.assign(dlatent_avg, tflib.lerp(batch_avg, dlatent_avg, dlatent_avg_beta))
            with tf.control_dependencies([update_op]):
                dlatents = tf.identity(dlatents)

    # Perform style mixing regularization.
    if style_mixing_prob is not None:
        with tf.variable_scope('StyleMix'):
            latents2 = tf.random_normal(tf.shape(latents_in))
            dlatents2 = components.mapping.get_output_for(latents2, labels_in, is_training=is_training, **kwargs)
            dlatents2 = tf.cast(dlatents2, tf.float32)
            layer_idx = np.arange(num_layers)[np.newaxis, :, np.newaxis]
            mixing_cutoff = tf.cond(
                tf.random_uniform([], 0.0, 1.0) < style_mixing_prob,
                lambda: tf.random_uniform([], 1, num_layers, dtype=tf.int32),
                lambda: num_layers)
            dlatents = tf.where(tf.broadcast_to(layer_idx < mixing_cutoff, tf.shape(dlatents)), dlatents, dlatents2)

    # Apply truncation.
    if truncation_psi is not None:
        with tf.variable_scope('Truncation'):
            layer_idx = np.arange(num_layers)[np.newaxis, :, np.newaxis]
            layer_psi = np.ones(layer_idx.shape, dtype=np.float32)
            if truncation_cutoff is None:
                layer_psi *= truncation_psi
            else:
                layer_psi = tf.where(layer_idx < truncation_cutoff, layer_psi * truncation_psi, layer_psi)
            dlatents = tflib.lerp(dlatent_avg, dlatents, layer_psi)

    # Evaluate synthesis network.
    images_out = components.synthesis.get_output_for(dlatents, is_training=is_training, force_clean_graph=is_template_graph, **kwargs)
    images_out = tf.identity(images_out, name='images_out')
    if return_dlatents:
        return images_out, dlatents
    return images_out

#----------------------------------------------------------------------------
# Generator mapping network.

def G_mapping(
    latents_in,                             # First input: Latent vectors (Z) [minibatch, latent_size].
    labels_in,                              # Second input: Conditioning labels [minibatch, label_size].

    # Input & output dimensions.
    latent_size             = 512,          # Latent vector (Z) dimensionality.
    label_size              = 0,            # Label dimensionality, 0 if no labels.
    dlatent_size            = 512,          # Disentangled latent (W) dimensionality.
    dlatent_broadcast       = None,         # Output disentangled latent (W) as [minibatch, dlatent_size] or [minibatch, dlatent_broadcast, dlatent_size].

    # Internal details.
    mapping_layers          = 8,            # Number of mapping layers.
    mapping_fmaps           = None,         # Number of activations in the mapping layers, None = same as dlatent_size.
    mapping_lrmul           = 0.01,         # Learning rate multiplier for the mapping layers.
    mapping_nonlinearity    = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
    normalize_latents       = True,         # Normalize latent vectors (Z) before feeding them to the mapping layers?
    label_fmaps             = None,         # Label embedding dimensionality, None = same as latent_size.
    dtype                   = 'float32',    # Data type to use for intermediate activations and outputs.

    **_kwargs,                              # Ignore unrecognized keyword args.
):
    # Inputs.
    latents_in.set_shape([None, latent_size])
    labels_in.set_shape([None, label_size])
    latents_in = tf.cast(latents_in, dtype)
    labels_in = tf.cast(labels_in, dtype)
    x = latents_in

    # Normalize latents.
    if normalize_latents:
        with tf.variable_scope('Normalize'):
            x = normalize_2nd_moment(x)

    # Embed labels, normalize, and concatenate with latents.
    if label_size > 0:
        with tf.variable_scope('LabelEmbed'):
            fmaps = label_fmaps if label_fmaps is not None else latent_size
            y = labels_in
            y = apply_bias_act(dense_layer(y, fmaps=fmaps))
            y = normalize_2nd_moment(y)
            x = tf.concat([x, y], axis=1)

    # Mapping layers.
    for layer_idx in range(mapping_layers):
        with tf.variable_scope(f'Dense{layer_idx}'):
            fmaps = mapping_fmaps if mapping_fmaps is not None and layer_idx < mapping_layers - 1 else dlatent_size
            x = apply_bias_act(dense_layer(x, fmaps=fmaps, lrmul=mapping_lrmul), act=mapping_nonlinearity, lrmul=mapping_lrmul)

    # Broadcast.
    if dlatent_broadcast is not None:
        with tf.variable_scope('Broadcast'):
            x = tf.tile(x[:, np.newaxis], [1, dlatent_broadcast, 1])

    # Output.
    assert x.dtype == tf.as_dtype(dtype)
    return tf.identity(x, name='dlatents_out')

#----------------------------------------------------------------------------
# Generator synthesis network.

def G_synthesis(
    dlatents_in,                        # Input: Disentangled latents (W) [minibatch, num_layers, dlatent_size].

    # Input & output dimensions.
    dlatent_size        = 512,          # Disentangled latent (W) dimensionality.
    num_channels        = 3,            # Number of output color channels.
    resolution          = 1024,         # Output resolution.

    # Capacity.
    fmap_base           = 16384,        # Overall multiplier for the number of feature maps.
    fmap_decay          = 1,            # Log2 feature map reduction when doubling the resolution.
    fmap_min            = 1,            # Minimum number of feature maps in any layer.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    fmap_const          = None,         # Number of feature maps in the constant input layer. None = default.

    # Internal details.
    use_noise           = True,         # Enable noise inputs?
    randomize_noise     = True,         # True = randomize noise inputs every time (non-deterministic), False = read noise inputs from variables.
    architecture        = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
    nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
    dtype               = 'float32',    # Data type to use for intermediate activations and outputs.
    num_fp16_res        = 0,            # Use FP16 for the N highest resolutions, regardless of dtype.
    conv_clamp          = None,         # Clamp the output of convolution layers to [-conv_clamp, +conv_clamp], None = disable clamping.
    resample_kernel     = [1,3,3,1],    # Low-pass filter to apply when resampling activations, None = box filter.
    fused_modconv       = False,        # Implement modulated_conv2d_layer() using grouped convolution?

    **_kwargs,                          # Ignore unrecognized keyword args.
):
    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)
    assert architecture in ['orig', 'skip', 'resnet']
    act = nonlinearity
    num_layers = resolution_log2 * 2 - 2

    # Disentangled latent (W).
    dlatents_in.set_shape([None, num_layers, dlatent_size])
    dlatents_in = tf.cast(dlatents_in, dtype)

    # Noise inputs.
    noise_inputs = []
    if use_noise:
        for layer_idx in range(num_layers - 1):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2**res, 2**res]
            noise_inputs.append(tf.get_variable(f'noise{layer_idx}', shape=shape, initializer=tf.initializers.random_normal(), trainable=False))

    # Single convolution layer with all the bells and whistles.
    def layer(x, layer_idx, fmaps, kernel, up=False):
        x = modulated_conv2d_layer(x, dlatents_in[:, layer_idx], fmaps=fmaps, kernel=kernel, up=up, resample_kernel=resample_kernel, fused_modconv=fused_modconv)
        if use_noise:
            if randomize_noise:
                noise = tf.random_normal([tf.shape(x)[0], 1, x.shape[2], x.shape[3]], dtype=x.dtype)
            else:
                noise = tf.cast(noise_inputs[layer_idx], x.dtype)
            noise_strength = tf.get_variable('noise_strength', shape=[], initializer=tf.initializers.zeros())
            x += noise * tf.cast(noise_strength, x.dtype)
        return apply_bias_act(x, act=act, clamp=conv_clamp)

    # Main block for one resolution.
    def block(x, res): # res = 3..resolution_log2
        x = tf.cast(x, 'float16' if res > resolution_log2 - num_fp16_res else dtype)
        t = x
        with tf.variable_scope('Conv0_up'):
            x = layer(x, layer_idx=res*2-5, fmaps=nf(res-1), kernel=3, up=True)
        with tf.variable_scope('Conv1'):
            x = layer(x, layer_idx=res*2-4, fmaps=nf(res-1), kernel=3)
        if architecture == 'resnet':
            with tf.variable_scope('Skip'):
                t = conv2d_layer(t, fmaps=nf(res-1), kernel=1, up=True, resample_kernel=resample_kernel)
                x = (x + t) * (1 / np.sqrt(2))
        return x

    # Upsampling block.
    def upsample(y):
        with tf.variable_scope('Upsample'):
            return upsample_2d(y, k=resample_kernel)

    # ToRGB block.
    def torgb(x, y, res): # res = 2..resolution_log2
        with tf.variable_scope('ToRGB'):
            t = modulated_conv2d_layer(x, dlatents_in[:, res*2-3], fmaps=num_channels, kernel=1, demodulate=False, fused_modconv=fused_modconv)
            t = apply_bias_act(t, clamp=conv_clamp)
            t = tf.cast(t, dtype)
            if y is not None:
                t += tf.cast(y, t.dtype)
            return t

    # Layers for 4x4 resolution.
    y = None
    with tf.variable_scope('4x4'):
        with tf.variable_scope('Const'):
            fmaps = fmap_const if fmap_const is not None else nf(1)
            x = tf.get_variable('const', shape=[1, fmaps, 4, 4], initializer=tf.initializers.random_normal())
            x = tf.tile(tf.cast(x, dtype), [tf.shape(dlatents_in)[0], 1, 1, 1])
        with tf.variable_scope('Conv'):
            x = layer(x, layer_idx=0, fmaps=nf(1), kernel=3)
        if architecture == 'skip':
            y = torgb(x, y, 2)

    # Layers for >=8x8 resolutions.
    for res in range(3, resolution_log2 + 1):
        with tf.variable_scope(f'{2**res}x{2**res}'):
            x = block(x, res)
            if architecture == 'skip':
                y = upsample(y)
            if architecture == 'skip' or res == resolution_log2:
                y = torgb(x, y, res)

    images_out = y
    assert images_out.dtype == tf.as_dtype(dtype)
    return tf.identity(images_out, name='images_out')

#----------------------------------------------------------------------------
# Discriminator.

def D_main(
    images_in,                          # First input: Images [minibatch, channel, height, width].
    labels_in,                          # Second input: Conditioning labels [minibatch, label_size].

    # Input dimensions.
    num_channels        = 3,            # Number of input color channels. Overridden based on dataset.
    resolution          = 1024,         # Input resolution. Overridden based on dataset.
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.

    # Capacity.
    fmap_base           = 16384,        # Overall multiplier for the number of feature maps.
    fmap_decay          = 1,            # Log2 feature map reduction when doubling the resolution.
    fmap_min            = 1,            # Minimum number of feature maps in any layer.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.

    # Internal details.
    mapping_layers      = 0,            # Number of additional mapping layers for the conditioning labels.
    mapping_fmaps       = None,         # Number of activations in the mapping layers, None = default.
    mapping_lrmul       = 0.1,          # Learning rate multiplier for the mapping layers.
    architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet'.
    nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
    mbstd_group_size    = None,         # Group size for the minibatch standard deviation layer, None = entire minibatch.
    mbstd_num_features  = 1,            # Number of features for the minibatch standard deviation layer, 0 = disable.
    dtype               = 'float32',    # Data type to use for intermediate activations and outputs.
    num_fp16_res        = 0,            # Use FP16 for the N highest resolutions, regardless of dtype.
    conv_clamp          = None,         # Clamp the output of convolution layers to [-conv_clamp, +conv_clamp], None = disable clamping.
    resample_kernel     = [1,3,3,1],    # Low-pass filter to apply when resampling activations, None = box filter.

    # Comparison methods.
    augment_strength    = 0,            # AdaptiveAugment.get_strength_var() for pagan & adropout.
    use_pagan           = False,        # pagan: Enable?
    pagan_num           = 16,           # pagan: Number of active bits with augment_strength=1.
    pagan_fade          = 0.5,          # pagan: Relative duration of fading in new bits.
    score_size          = 1,            # auxrot: Number of scalars to output. Can vary between evaluations.
    score_max           = 1,            # auxrot: Maximum number of scalars to output. Must be set at construction time.
    use_spectral_norm   = False,        # spectralnorm: Enable?
    adaptive_dropout    = 0,            # adropout: Standard deviation to use with augment_strength=1, 0 = disable.
    freeze_layers       = 0,            # Freeze-D: Number of layers to freeze.

    **_kwargs,                          # Ignore unrecognized keyword args.
):
    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)
    assert architecture in ['orig', 'skip', 'resnet']
    if mapping_fmaps is None:
        mapping_fmaps = nf(0)
    act = nonlinearity

    # Inputs.
    images_in.set_shape([None, num_channels, resolution, resolution])
    labels_in.set_shape([None, label_size])
    images_in = tf.cast(images_in, dtype)
    labels_in = tf.cast(labels_in, dtype)

    # Label embedding and mapping.
    if label_size > 0:
        y = labels_in
        with tf.variable_scope('LabelEmbed'):
            y = apply_bias_act(dense_layer(y, fmaps=mapping_fmaps))
            y = normalize_2nd_moment(y)
        for idx in range(mapping_layers):
            with tf.variable_scope(f'Mapping{idx}'):
                y = apply_bias_act(dense_layer(y, fmaps=mapping_fmaps, lrmul=mapping_lrmul), act=act, lrmul=mapping_lrmul)
        labels_in = y

    # Adaptive multiplicative dropout.
    def adrop(x):
        if adaptive_dropout != 0:
            s = [tf.shape(x)[0], x.shape[1]] + [1] * (x.shape.rank - 2)
            x *= tf.cast(tf.exp(tf.random_normal(s) * (augment_strength * adaptive_dropout)), x.dtype)
        return x

    # Freeze-D.
    cur_layer_idx = 0
    def is_next_layer_trainable():
        nonlocal cur_layer_idx
        trainable = (cur_layer_idx >= freeze_layers)
        cur_layer_idx += 1
        return trainable

    # Construct PA-GAN bit vector.
    pagan_bits = None
    pagan_signs = None
    if use_pagan:
        with tf.variable_scope('PAGAN'):
            idx = tf.range(pagan_num, dtype=tf.float32)
            active = (augment_strength * pagan_num - idx - 1) / max(pagan_fade, 1e-8) + 1
            prob = tf.clip_by_value(active[np.newaxis, :], 0, 1) * 0.5
            rnd = tf.random_uniform([tf.shape(images_in)[0], pagan_num])
            pagan_bits = tf.cast(rnd < prob, dtype=tf.float32)
            pagan_signs = tf.reduce_prod(1 - pagan_bits * 2, axis=1, keepdims=True)

    # FromRGB block.
    def fromrgb(x, y, res): # res = 2..resolution_log2
        with tf.variable_scope('FromRGB'):
            trainable = is_next_layer_trainable()
            t = tf.cast(y, 'float16' if res > resolution_log2 - num_fp16_res else dtype)
            t = adrop(conv2d_layer(t, fmaps=nf(res-1), kernel=1, trainable=trainable))
            if pagan_bits is not None:
                with tf.variable_scope('PAGAN'):
                    t += dense_layer(tf.cast(pagan_bits, t.dtype), fmaps=nf(res-1), trainable=trainable)[:, :, np.newaxis, np.newaxis]
            t = apply_bias_act(t, act=act, clamp=conv_clamp, trainable=trainable)
            if x is not None:
                t += tf.cast(x, t.dtype)
            return t

    # Main block for one resolution.
    def block(x, res): # res = 2..resolution_log2
        x = tf.cast(x, 'float16' if res > resolution_log2 - num_fp16_res else dtype)
        t = x
        with tf.variable_scope('Conv0'):
            trainable = is_next_layer_trainable()
            x = apply_bias_act(adrop(conv2d_layer(x, fmaps=nf(res-1), kernel=3, trainable=trainable, use_spectral_norm=use_spectral_norm)), act=act, clamp=conv_clamp, trainable=trainable)
        with tf.variable_scope('Conv1_down'):
            trainable = is_next_layer_trainable()
            x = apply_bias_act(adrop(conv2d_layer(x, fmaps=nf(res-2), kernel=3, down=True, resample_kernel=resample_kernel, trainable=trainable, use_spectral_norm=use_spectral_norm)), act=act, clamp=conv_clamp, trainable=trainable)
        if architecture == 'resnet':
            with tf.variable_scope('Skip'):
                trainable = is_next_layer_trainable()
                t = adrop(conv2d_layer(t, fmaps=nf(res-2), kernel=1, down=True, resample_kernel=resample_kernel, trainable=trainable))
                x = (x + t) * (1 / np.sqrt(2))
        return x

    # Downsampling block.
    def downsample(y):
        with tf.variable_scope('Downsample'):
            return downsample_2d(y, k=resample_kernel)

    # Layers for >=8x8 resolutions.
    x = None
    y = images_in
    for res in range(resolution_log2, 2, -1):
        with tf.variable_scope(f'{2**res}x{2**res}'):
            if architecture == 'skip' or res == resolution_log2:
                x = fromrgb(x, y, res)
            x = block(x, res)
            if architecture == 'skip':
                y = downsample(y)

    # Layers for 4x4 resolution.
    with tf.variable_scope('4x4'):
        if architecture == 'skip':
            x = fromrgb(x, y, 2)
        x = tf.cast(x, dtype)
        if mbstd_num_features > 0:
            with tf.variable_scope('MinibatchStddev'):
                x = minibatch_stddev_layer(x, mbstd_group_size, mbstd_num_features)
        with tf.variable_scope('Conv'):
            trainable = is_next_layer_trainable()
            x = apply_bias_act(adrop(conv2d_layer(x, fmaps=nf(1), kernel=3, trainable=trainable, use_spectral_norm=use_spectral_norm)), act=act, clamp=conv_clamp, trainable=trainable)
        with tf.variable_scope('Dense0'):
            trainable = is_next_layer_trainable()
            x = apply_bias_act(adrop(dense_layer(x, fmaps=nf(0), trainable=trainable)), act=act, trainable=trainable)

    # Output layer (always trainable).
    with tf.variable_scope('Output'):
        if label_size > 0:
            assert score_max == 1
            x = apply_bias_act(dense_layer(x, fmaps=mapping_fmaps))
            x = tf.reduce_sum(x * labels_in, axis=1, keepdims=True) / np.sqrt(mapping_fmaps)
        else:
            x = apply_bias_act(dense_layer(x, fmaps=score_max))
        if pagan_signs is not None:
            assert score_max == 1
            x *= pagan_signs
    scores_out = x[:, :score_size]

    # Output.
    assert scores_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(scores_out, name='scores_out')
    return scores_out

#----------------------------------------------------------------------------
