#
#   ~~ Flesh Digressions ~~
#         Or, Circular Interpolation of the StyleGAN Synthesis Network's Constant Layer
#   ~~~ aydao ~~~~ 2020 ~~~
#
#   Based on halcy's circular interpolation script https://pastebin.com/RTtV2UY7
#
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import dnnlib
import dnnlib.tflib as tflib
import math
import moviepy.editor
from numpy import linalg
import numpy as np
import pickle
import argparse
from datetime import datetime

def circular_interpolation(radius, latents_persistent, latents_interpolate):

    latents_a, latents_b, latents_c = latents_persistent

    latents_axis_x = (latents_a - latents_b).flatten() / linalg.norm(latents_a - latents_b)
    latents_axis_y = (latents_a - latents_c).flatten() / linalg.norm(latents_a - latents_c)

    latents_x = math.sin(math.pi * 2.0 * latents_interpolate) * radius
    latents_y = math.cos(math.pi * 2.0 * latents_interpolate) * radius

    latents = latents_a + latents_x * latents_axis_x + latents_y * latents_axis_y
    return latents

def generate_from_generator_adaptive(psi,radius_large,radius_small,step1,step2,video_length, Gs):
    # psi = args.psi # 0.7
    # radius_large = args.radius_large # 600.0
    # radius_small = args.radius_small # 40.0
    current_position_increment = step1 # 0.005
    current_position_style_increment = step2 # 0.0025
    # video_length = args.video_length # 1.0
    output_format = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    
    # latents for the circular interpolation in latent space
    rnd = np.random
    latents_a = rnd.randn(1, Gs.input_shape[1])
    latents_b = rnd.randn(1, Gs.input_shape[1])
    latents_c = rnd.randn(1, Gs.input_shape[1])
    latents_persistent_small = (latents_a, latents_b, latents_c)

    # latents for the circular interpolation of the unrolled constant layer
    latent_size = 512 # default StyleGAN latent size
    constant_layer_size = 4 # default StyleGAN constant layer size is 4x4
    constant_layer_total = latent_size * constant_layer_size * constant_layer_size # 8192
    latents_aa = rnd.randn(1, constant_layer_total)
    latents_bb = rnd.randn(1, constant_layer_total)
    latents_cc = rnd.randn(1, constant_layer_total)
    latents_persistent_large = (latents_aa, latents_bb, latents_cc)

    # initialize the circular interpolation
    current_position = 0.0
    current_position_style = 0.0
    current_latent = circular_interpolation(radius_small, latents_persistent_small, current_position)
    current_image = Gs.run(current_latent, None, truncation_psi=psi, randomize_noise=False, output_transform=output_format)[0]
    output_frames = []

    # Create the frames while interpolating along the circle, in both the latent space and the constant layer
    while(current_position_style < video_length):

        current_position += current_position_increment
        current_position_style += current_position_style_increment

        # interpolate the weights of the constant layer
        w = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name == 'G_synthesis_1/4x4/Const/const:0'][0]
        v1 = tf.identity(tflib.run(['G_synthesis_1/4x4/Const/const:0'])[0])
        v2 = tf.reshape(v1, [1, constant_layer_total])
        v2 += circular_interpolation(radius_large, latents_persistent_large, current_position + np.pi)
        v2 = tf.reshape(v2, [1, latent_size, constant_layer_size, constant_layer_size])
        tf.get_default_session().run(tf.assign(w, v2))

        # interpolate along the latent space
        current_latent = circular_interpolation(radius_small, latents_persistent_small, current_position_style)
        current_image = images = Gs.run(current_latent, None, truncation_psi=psi, randomize_noise=False, output_transform=output_format)[0]
        output_frames.append(current_image)

        tf.get_default_session().run(tf.assign(w, v1))

        # stops at 1.0 (or whatever value to which video_length is set)
        print('Stopping at',video_length,'currently at',current_position_style, flush=True) 

    return output_frames

def main(pkl,psi,radius_large,radius_small,step1,step2,video_length=1.0):

    tflib.init_tf()
    print('Loading networks from "%s"...' % pkl)
    with dnnlib.util.open_url(pkl) as fp:
        _G, _D, Gs = pickle.load(fp)

    frames = generate_from_generator_adaptive(psi,radius_large,radius_small,step1,step2,video_length, Gs)
    frames = moviepy.editor.ImageSequenceClip(frames, fps=30)

    # Generate video at the current date and timestamp
    timestamp = datetime.now().strftime("%d-%m-%Y-%I-%M-%S-%p")
    mp4_file = './circular-'+timestamp+'.mp4'
    mp4_codec = 'libx264'
    mp4_bitrate = '15M'
    mp4_fps = 24 # 20

    frames.write_videofile(mp4_file, fps=mp4_fps, codec=mp4_codec, bitrate=mp4_bitrate)

    sess = tf.get_default_session()
    sess.close()
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description='Creates a video of a circular interpolation of the constant layer for an input StyleGAN model.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--pkl', help='A .pkl of a StyleGAN network model', required=True)
    parser.add_argument('--psi', help='The truncation psi used in the generator', default=0.7, type=float)
    parser.add_argument('--radius_large', help='The radius for the constant layer interpolation', default=300.0, type=float)
    parser.add_argument('--radius_small', help='The radius for the latent space interpolation', default=40.0, type=float)
    parser.add_argument('--step1', help='The value of the step/increment for the constant layer interpolation', default=0.005, type=float)
    parser.add_argument('--step2', help='The value of the step/increment for the latent space interpolation', default=0.0025, type=float)
    parser.add_argument('--video_length', help='The length of the video in terms of circular interpolation (recommended to keep at 1.0)', default=1.0, type=float)

    args = parser.parse_args()

    main(args.pkl, args.psi, args.radius_large, args.radius_small, args.step1, args.step2, args.video_length)
