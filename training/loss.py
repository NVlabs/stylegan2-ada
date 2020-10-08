# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""

import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary

#----------------------------------------------------------------------------
# Report statistic for all interested parties (AdaptiveAugment and tfevents).

def report_stat(aug, name, value):
    if aug is not None:
        value = aug.report_stat(name, value)
    value = autosummary(name, value)
    return value

#----------------------------------------------------------------------------
# Report loss terms and collect them into EasyDict.

def report_loss(aug, G_loss, D_loss, G_reg=None, D_reg=None):
    assert G_loss is not None and D_loss is not None
    terms = dnnlib.EasyDict(G_reg=None, D_reg=None)
    terms.G_loss = report_stat(aug, 'Loss/G/loss', G_loss)
    terms.D_loss = report_stat(aug, 'Loss/D/loss', D_loss)
    if G_reg is not None: terms.G_reg = report_stat(aug, 'Loss/G/reg', G_reg)
    if D_reg is not None: terms.D_reg = report_stat(aug, 'Loss/D/reg', D_reg)
    return terms

#----------------------------------------------------------------------------
# Evaluate G and return results as EasyDict.

def eval_G(G, latents, labels, return_dlatents=False):
    r = dnnlib.EasyDict()
    r.args = dnnlib.EasyDict()
    r.args.is_training = True
    if return_dlatents:
        r.args.return_dlatents = True
    r.images = G.get_output_for(latents, labels, **r.args)

    r.dlatents = None
    if return_dlatents:
        r.images, r.dlatents = r.images
    return r

#----------------------------------------------------------------------------
# Evaluate D and return results as EasyDict.

def eval_D(D, aug, images, labels, report=None, augment_inputs=True, return_aux=0):
    r = dnnlib.EasyDict()
    r.images_aug = images
    r.labels_aug = labels
    if augment_inputs and aug is not None:
        r.images_aug, r.labels_aug = aug.apply(r.images_aug, r.labels_aug)

    r.args = dnnlib.EasyDict()
    r.args.is_training = True
    if aug is not None:
        r.args.augment_strength = aug.get_strength_var()
    if return_aux > 0:
        r.args.score_size = return_aux + 1
    r.scores = D.get_output_for(r.images_aug, r.labels_aug, **r.args)

    r.aux = None
    if return_aux:
        r.aux = r.scores[:, 1:]
        r.scores = r.scores[:, :1]

    if report is not None:
        report_ops = [
            report_stat(aug, 'Loss/scores/' + report, r.scores),
            report_stat(aug, 'Loss/signs/' + report, tf.sign(r.scores)),
            report_stat(aug, 'Loss/squares/' + report, tf.square(r.scores)),
        ]
        with tf.control_dependencies(report_ops):
            r.scores = tf.identity(r.scores)
    return r

#----------------------------------------------------------------------------
# Non-saturating logistic loss with R1 and path length regularizers, used
# in the paper "Analyzing and Improving the Image Quality of StyleGAN".

def stylegan2(G, D, aug, fake_labels, real_images, real_labels, r1_gamma=10, pl_minibatch_shrink=2, pl_decay=0.01, pl_weight=2, **_kwargs):
    # Evaluate networks for the main loss.
    minibatch_size = tf.shape(fake_labels)[0]
    fake_latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    G_fake = eval_G(G, fake_latents, fake_labels, return_dlatents=True)
    D_fake = eval_D(D, aug, G_fake.images, fake_labels, report='fake')
    D_real = eval_D(D, aug, real_images, real_labels, report='real')

    # Non-saturating logistic loss from "Generative Adversarial Nets".
    with tf.name_scope('Loss_main'):
        G_loss = tf.nn.softplus(-D_fake.scores) # -log(sigmoid(D_fake.scores)), pylint: disable=invalid-unary-operand-type
        D_loss = tf.nn.softplus(D_fake.scores) # -log(1 - sigmoid(D_fake.scores))
        D_loss += tf.nn.softplus(-D_real.scores) # -log(sigmoid(D_real.scores)), pylint: disable=invalid-unary-operand-type
        G_reg = 0
        D_reg = 0

    # R1 regularizer from "Which Training Methods for GANs do actually Converge?".
    if r1_gamma != 0:
        with tf.name_scope('Loss_R1'):
            r1_grads = tf.gradients(tf.reduce_sum(D_real.scores), [real_images])[0]
            r1_penalty = tf.reduce_sum(tf.square(r1_grads), axis=[1,2,3])
            r1_penalty = report_stat(aug, 'Loss/r1_penalty', r1_penalty)
            D_reg += r1_penalty * (r1_gamma * 0.5)

    # Path length regularizer from "Analyzing and Improving the Image Quality of StyleGAN".
    if pl_weight != 0:
        with tf.name_scope('Loss_PL'):

            # Evaluate the regularization term using a smaller minibatch to conserve memory.
            G_pl = G_fake
            if pl_minibatch_shrink > 1:
                pl_minibatch_size = minibatch_size // pl_minibatch_shrink
                pl_latents = fake_latents[:pl_minibatch_size]
                pl_labels = fake_labels[:pl_minibatch_size]
                G_pl = eval_G(G, pl_latents, pl_labels, return_dlatents=True)

            # Compute |J*y|.
            pl_noise = tf.random_normal(tf.shape(G_pl.images)) / np.sqrt(np.prod(G.output_shape[2:]))
            pl_grads = tf.gradients(tf.reduce_sum(G_pl.images * pl_noise), [G_pl.dlatents])[0]
            pl_lengths = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(pl_grads), axis=2), axis=1))

            # Track exponential moving average of |J*y|.
            with tf.control_dependencies(None):
                pl_mean_var = tf.Variable(name='pl_mean', trainable=False, initial_value=0, dtype=tf.float32)
            pl_mean = pl_mean_var + pl_decay * (tf.reduce_mean(pl_lengths) - pl_mean_var)
            pl_update = tf.assign(pl_mean_var, pl_mean)

            # Calculate (|J*y|-a)^2.
            with tf.control_dependencies([pl_update]):
                pl_penalty = tf.square(pl_lengths - pl_mean)
                pl_penalty = report_stat(aug, 'Loss/pl_penalty', pl_penalty)

            # Apply weight.
            #
            # Note: The division in pl_noise decreases the weight by num_pixels, and the reduce_mean
            # in pl_lengths decreases it by num_affine_layers. The effective weight then becomes:
            #
            # gamma_pl = pl_weight / num_pixels / num_affine_layers
            # = 2 / (r^2) / (log2(r) * 2 - 2)
            # = 1 / (r^2 * (log2(r) - 1))
            # = ln(2) / (r^2 * (ln(r) - ln(2))
            #
            G_reg += tf.tile(pl_penalty, [pl_minibatch_shrink]) * pl_weight

    return report_loss(aug, G_loss, D_loss, G_reg, D_reg)

#----------------------------------------------------------------------------
# Hybrid loss used for comparison methods used in the paper
# "Training Generative Adversarial Networks with Limited Data".

def cmethods(G, D, aug, fake_labels, real_images, real_labels,
    r1_gamma=10, r2_gamma=0,
    pl_minibatch_shrink=2, pl_decay=0.01, pl_weight=2,
    bcr_real_weight=0, bcr_fake_weight=0, bcr_augment=None,
    zcr_gen_weight=0, zcr_dis_weight=0, zcr_noise_std=0.1,
    auxrot_alpha=0, auxrot_beta=0,
    **_kwargs,
):
    # Evaluate networks for the main loss.
    minibatch_size = tf.shape(fake_labels)[0]
    fake_latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    G_fake = eval_G(G, fake_latents, fake_labels)
    D_fake = eval_D(D, aug, G_fake.images, fake_labels, report='fake')
    D_real = eval_D(D, aug, real_images, real_labels, report='real')

    # Non-saturating logistic loss from "Generative Adversarial Nets".
    with tf.name_scope('Loss_main'):
        G_loss = tf.nn.softplus(-D_fake.scores) # -log(sigmoid(D_fake.scores)), pylint: disable=invalid-unary-operand-type
        D_loss = tf.nn.softplus(D_fake.scores) # -log(1 - sigmoid(D_fake.scores))
        D_loss += tf.nn.softplus(-D_real.scores) # -log(sigmoid(D_real.scores)), pylint: disable=invalid-unary-operand-type
        G_reg = 0
        D_reg = 0

    # R1 and R2 regularizers from "Which Training Methods for GANs do actually Converge?".
    if r1_gamma != 0 or r2_gamma != 0:
        with tf.name_scope('Loss_R1R2'):
            if r1_gamma != 0:
                r1_grads = tf.gradients(tf.reduce_sum(D_real.scores), [real_images])[0]
                r1_penalty = tf.reduce_sum(tf.square(r1_grads), axis=[1,2,3])
                r1_penalty = report_stat(aug, 'Loss/r1_penalty', r1_penalty)
                D_reg += r1_penalty * (r1_gamma * 0.5)
            if r2_gamma != 0:
                r2_grads = tf.gradients(tf.reduce_sum(D_fake.scores), [G_fake.images])[0]
                r2_penalty = tf.reduce_sum(tf.square(r2_grads), axis=[1,2,3])
                r2_penalty = report_stat(aug, 'Loss/r2_penalty', r2_penalty)
                D_reg += r2_penalty * (r2_gamma * 0.5)

    # Path length regularizer from "Analyzing and Improving the Image Quality of StyleGAN".
    if pl_weight != 0:
        with tf.name_scope('Loss_PL'):
            pl_minibatch_size = minibatch_size // pl_minibatch_shrink
            pl_latents = fake_latents[:pl_minibatch_size]
            pl_labels = fake_labels[:pl_minibatch_size]
            G_pl = eval_G(G, pl_latents, pl_labels, return_dlatents=True)
            pl_noise = tf.random_normal(tf.shape(G_pl.images)) / np.sqrt(np.prod(G.output_shape[2:]))
            pl_grads = tf.gradients(tf.reduce_sum(G_pl.images * pl_noise), [G_pl.dlatents])[0]
            pl_lengths = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(pl_grads), axis=2), axis=1))
            with tf.control_dependencies(None):
                pl_mean_var = tf.Variable(name='pl_mean', trainable=False, initial_value=0, dtype=tf.float32)
            pl_mean = pl_mean_var + pl_decay * (tf.reduce_mean(pl_lengths) - pl_mean_var)
            pl_update = tf.assign(pl_mean_var, pl_mean)
            with tf.control_dependencies([pl_update]):
                pl_penalty = tf.square(pl_lengths - pl_mean)
                pl_penalty = report_stat(aug, 'Loss/pl_penalty', pl_penalty)
            G_reg += tf.tile(pl_penalty, [pl_minibatch_shrink]) * pl_weight

    # bCR regularizer from "Improved consistency regularization for GANs".
    if (bcr_real_weight != 0 or bcr_fake_weight != 0) and bcr_augment is not None:
        with tf.name_scope('Loss_bCR'):
            if bcr_real_weight != 0:
                bcr_real_images, bcr_real_labels = dnnlib.util.call_func_by_name(D_real.images_aug, D_real.labels_aug, **bcr_augment)
                D_bcr_real = eval_D(D, aug, bcr_real_images, bcr_real_labels, report='real_bcr', augment_inputs=False)
                bcr_real_penalty = tf.square(D_bcr_real.scores - D_real.scores)
                bcr_real_penalty = report_stat(aug, 'Loss/bcr_penalty/real', bcr_real_penalty)
                D_loss += bcr_real_penalty * bcr_real_weight # NOTE: Must not use lazy regularization for this term.
            if bcr_fake_weight != 0:
                bcr_fake_images, bcr_fake_labels = dnnlib.util.call_func_by_name(D_fake.images_aug, D_fake.labels_aug, **bcr_augment)
                D_bcr_fake = eval_D(D, aug, bcr_fake_images, bcr_fake_labels, report='fake_bcr', augment_inputs=False)
                bcr_fake_penalty = tf.square(D_bcr_fake.scores - D_fake.scores)
                bcr_fake_penalty = report_stat(aug, 'Loss/bcr_penalty/fake', bcr_fake_penalty)
                D_loss += bcr_fake_penalty * bcr_fake_weight # NOTE: Must not use lazy regularization for this term.

    # zCR regularizer from "Improved consistency regularization for GANs".
    if zcr_gen_weight != 0 or zcr_dis_weight != 0:
        with tf.name_scope('Loss_zCR'):
            zcr_fake_latents = fake_latents + tf.random_normal([minibatch_size] + G.input_shapes[0][1:]) * zcr_noise_std
            G_zcr = eval_G(G, zcr_fake_latents, fake_labels)
            if zcr_gen_weight > 0:
                zcr_gen_penalty = -tf.reduce_mean(tf.square(G_fake.images - G_zcr.images), axis=[1,2,3])
                zcr_gen_penalty = report_stat(aug, 'Loss/zcr_gen_penalty', zcr_gen_penalty)
                G_loss += zcr_gen_penalty * zcr_gen_weight
            if zcr_dis_weight > 0:
                D_zcr = eval_D(D, aug, G_zcr.images, fake_labels, report='fake_zcr', augment_inputs=False)
                zcr_dis_penalty = tf.square(D_fake.scores - D_zcr.scores)
                zcr_dis_penalty = report_stat(aug, 'Loss/zcr_dis_penalty', zcr_dis_penalty)
                D_loss += zcr_dis_penalty * zcr_dis_weight

    # Auxiliary rotation loss from "Self-supervised GANs via auxiliary rotation loss".
    if auxrot_alpha != 0 or auxrot_beta != 0:
        with tf.name_scope('Loss_AuxRot'):
            idx = tf.range(minibatch_size * 4, dtype=tf.int32) // minibatch_size
            b0 = tf.logical_or(tf.equal(idx, 0), tf.equal(idx, 1))
            b1 = tf.logical_or(tf.equal(idx, 0), tf.equal(idx, 3))
            b2 = tf.logical_or(tf.equal(idx, 0), tf.equal(idx, 2))
            if auxrot_alpha != 0:
                auxrot_fake = tf.tile(G_fake.images, [4, 1, 1, 1])
                auxrot_fake = tf.where(b0, auxrot_fake, tf.reverse(auxrot_fake, [2]))
                auxrot_fake = tf.where(b1, auxrot_fake, tf.reverse(auxrot_fake, [3]))
                auxrot_fake = tf.where(b2, auxrot_fake, tf.transpose(auxrot_fake, [0, 1, 3, 2]))
                D_auxrot_fake = eval_D(D, aug, auxrot_fake, fake_labels, return_aux=4)
                G_loss += tf.nn.sparse_softmax_cross_entropy_with_logits(labels=idx, logits=D_auxrot_fake.aux) * auxrot_alpha
            if auxrot_beta != 0:
                auxrot_real = tf.tile(real_images, [4, 1, 1, 1])
                auxrot_real = tf.where(b0, auxrot_real, tf.reverse(auxrot_real, [2]))
                auxrot_real = tf.where(b1, auxrot_real, tf.reverse(auxrot_real, [3]))
                auxrot_real = tf.where(b2, auxrot_real, tf.transpose(auxrot_real, [0, 1, 3, 2]))
                D_auxrot_real = eval_D(D, aug, auxrot_real, real_labels, return_aux=4)
                D_loss += tf.nn.sparse_softmax_cross_entropy_with_logits(labels=idx, logits=D_auxrot_real.aux) * auxrot_beta

    return report_loss(aug, G_loss, D_loss, G_reg, D_reg)

#----------------------------------------------------------------------------
# WGAN-GP loss with epsilon penalty, used in the paper
# "Progressive Growing of GANs for Improved Quality, Stability, and Variation".

def wgangp(G, D, aug, fake_labels, real_images, real_labels, wgan_epsilon=0.001, wgan_lambda=10, wgan_target=1, **_kwargs):
    minibatch_size = tf.shape(fake_labels)[0]
    fake_latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    G_fake = eval_G(G, fake_latents, fake_labels)
    D_fake = eval_D(D, aug, G_fake.images, fake_labels, report='fake')
    D_real = eval_D(D, aug, real_images, real_labels, report='real')

    # WGAN loss from "Wasserstein Generative Adversarial Networks".
    with tf.name_scope('Loss_main'):
        G_loss = -D_fake.scores # pylint: disable=invalid-unary-operand-type
        D_loss = D_fake.scores - D_real.scores

    # Epsilon penalty from "Progressive Growing of GANs for Improved Quality, Stability, and Variation"
    with tf.name_scope('Loss_epsilon'):
        epsilon_penalty = report_stat(aug, 'Loss/epsilon_penalty', tf.square(D_real.scores))
        D_loss += epsilon_penalty * wgan_epsilon

    # Gradient penalty from "Improved Training of Wasserstein GANs".
    with tf.name_scope('Loss_GP'):
        mix_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0, 1, dtype=G_fake.images.dtype)
        mix_images = tflib.lerp(tf.cast(real_images, G_fake.images.dtype), G_fake.images, mix_factors)
        mix_labels = real_labels # NOTE: Mixing is performed without respect to fake_labels.
        D_mix = eval_D(D, aug, mix_images, mix_labels, report='mix')
        mix_grads = tf.gradients(tf.reduce_sum(D_mix.scores), [mix_images])[0]
        mix_norms = tf.sqrt(tf.reduce_sum(tf.square(mix_grads), axis=[1,2,3]))
        mix_norms = report_stat(aug, 'Loss/mix_norms', mix_norms)
        gradient_penalty = tf.square(mix_norms - wgan_target)
        D_reg = gradient_penalty * (wgan_lambda / (wgan_target**2))

    return report_loss(aug, G_loss, D_loss, None, D_reg)

#----------------------------------------------------------------------------
