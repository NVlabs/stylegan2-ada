# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import argparse
import sys
import os
import pickle
import re

import scipy
import numpy as np
import PIL.Image

import dnnlib
import dnnlib.tflib as tflib

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import moviepy.editor

import warnings # mostly numpy warnings for me
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

#----------------------------------------------------------------------------

def create_image_grid(images, grid_size=None):
    '''
    Args:
        images (np.array): images to place on the grid
        grid_size (tuple(int, int)): size of grid (grid_w, grid_h)
    Returns:
        grid (np.array): image grid of size grid_size
    '''
    # Some sanity check:
    assert images.ndim == 3 or images.ndim == 4
    num, img_h, img_w = images.shape[0], images.shape[1], images.shape[2]
    if grid_size is not None:
        grid_w, grid_h = tuple(grid_size)
    else:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) // grid_w + 1, 1)
    # Get the grid
    grid = np.zeros(
        [grid_h * img_h, grid_w * img_w] + list(images.shape[-1:]), dtype=images.dtype
    )
    for idx in range(num):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[y : y + img_h, x : x + img_w, ...] = images[idx]
    return grid

#----------------------------------------------------------------------------

def generate_images(network_pkl, seeds, truncation_psi, outdir, class_idx=None, dlatents_npz=None, grid=False):
    tflib.init_tf()
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as fp:
        _G, _D, Gs = pickle.load(fp)

    os.makedirs(outdir, exist_ok=True)

    # Render images for a given dlatent vector.
    if dlatents_npz is not None:
        print(f'Generating images from dlatents file "{dlatents_npz}"')
        dlatents = np.load(dlatents_npz)['dlatents']
        max_l = 2 * int(np.log2(Gs.output_shape[-1]) - 1)  # max_l=18 for 1024x1024 models
        if dlatents.shape[1:] != (max_l, 512):  # [N, max_l, 512]
            actual_size = int(2**(dlatents.shape[1]//2+1))
            print(f'''Mismatch of loaded dlatents and network! dlatents was created with network of size: {actual_size}\n
                   {network_pkl} is of size {Gs.output_shape[-1]}''')
            sys.exit(1)
        imgs = Gs.components.synthesis.run(dlatents, output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True))
        for i, img in enumerate(imgs):
            fname = f'{outdir}/dlatent{i:02d}.png'
            print (f'Saved {fname}')
            PIL.Image.fromarray(img, 'RGB').save(fname)
        return

    # Render images for dlatents initialized from random seeds.
    Gs_kwargs = {
        'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
        'randomize_noise': False
    }
    if truncation_psi is not None:
        Gs_kwargs['truncation_psi'] = truncation_psi

    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
    label = np.zeros([1] + Gs.input_shapes[1][1:])
    if class_idx is not None:
        label[:, class_idx] = 1

    images = []
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        image = Gs.run(z, label, **Gs_kwargs) # [minibatch, height, width, channel]
        images.append(image[0])
        PIL.Image.fromarray(image[0], 'RGB').save(f'{outdir}/seed{seed:04d}.png')

    # If user wants to save a grid of the generated images
    if grid:
        print('Generating image grid...')
        PIL.Image.fromarray(create_image_grid(np.array(images)), 'RGB').save(f'{outdir}/grid.png')

#----------------------------------------------------------------------------

def lerp_video(network_pkl,                # Path to pretrained model pkl file
               seeds,                      # Random seeds
               grid_w=None,                # Number of columns
               grid_h=None,                # Number of rows
               truncation_psi=1.0,         # Truncation trick
               outdir='out',               # Output dir
               slowdown=1,                 # Slowdown of the video (power of 2)
               duration_sec=30.0,          # Duration of video in seconds
               smoothing_sec=3.0,
               mp4_fps=30,
               mp4_codec="libx264",
               mp4_bitrate="16M"):
    # Sanity check regarding slowdown
    message = 'slowdown must be a power of 2 (1, 2, 4, 8, ...) and greater than 0!'
    assert slowdown & (slowdown - 1) == 0 and slowdown > 0, message
    # Initialize TensorFlow and create outdir
    tflib.init_tf()
    os.makedirs(outdir, exist_ok=True)
    # Total duration of video and number of frames to generate
    num_frames = int(np.rint(duration_sec * mp4_fps))
    total_duration = duration_sec * slowdown

    print(f'Loading network from {network_pkl}...')
    with dnnlib.util.open_url(network_pkl) as fp:
        _G, _D, Gs = pickle.load(fp)

    print("Generating latent vectors...")
    # If there's more than one seed provided and the shape isn't specified
    if grid_w == grid_h == None and len(seeds) >= 1:
        # number of images according to the seeds provided
        num = len(seeds)
        # Get the grid width and height according to num:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) // grid_w + 1, 1)
        grid_size = [grid_w, grid_h]
        # [frame, image, channel, component]:
        shape = [num_frames] + Gs.input_shape[1:]
        # Get the latents:
        all_latents = np.stack([np.random.RandomState(seed).randn(*shape).astype(np.float32) for seed in seeds], axis=1)
    # If only one seed is provided and the shape is specified
    elif None not in (grid_w, grid_h) and len(seeds) == 1:
        # Otherwise, the user gives one seed and the grid width and height:
        grid_size = [grid_w, grid_h]
        # [frame, image, channel, component]:
        shape = [num_frames, np.prod(grid_size)] + Gs.input_shape[1:]
        # Get the latents with the random state:
        random_state = np.random.RandomState(seeds)
        all_latents = random_state.randn(*shape).astype(np.float32)
    else:
        print("Error: wrong combination of arguments! Please provide \
                either one seed and the grid width and height, or a \
                list of seeds to use.")
        sys.exit(1)

    all_latents = scipy.ndimage.gaussian_filter(
        all_latents,
        [smoothing_sec * mp4_fps] + [0] * len(Gs.input_shape),
        mode="wrap"
    )
    all_latents /= np.sqrt(np.mean(np.square(all_latents)))
    # Name of the final mp4 video
    mp4 = f"{grid_w}x{grid_h}-lerp-{slowdown}xslowdown.mp4"

    # Aux function to slowdown the video by 2x
    def double_slowdown(latents, duration_sec, num_frames):
        # Make an empty latent vector with double the amount of frames
        z = np.empty(np.multiply(latents.shape, [2, 1, 1]), dtype=np.float32)
        # Populate it
        for i in range(len(latents)):
            z[2*i] = latents[i]
        # Interpolate in the odd frames
        for i in range(1, len(z), 2):
            # For the last frame, we loop to the first one
            if i == len(z) - 1:
                z[i] = (z[0] + z[i-1]) / 2
            else:
                z[i] = (z[i-1] + z[i+1]) / 2
        # We also need to double the duration_sec and num_frames
        duration_sec *= 2
        num_frames *= 2
        # Return the new latents, and the two previous quantities
        return z, duration_sec, num_frames

    while slowdown > 1:
        all_latents, duration_sec, num_frames = double_slowdown(all_latents, duration_sec, num_frames)
        slowdown //= 2

    # Define the kwargs for the Generator:
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8,
                                      nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if truncation_psi is not None:
        Gs_kwargs.truncation_psi = truncation_psi

    # Aux function: Frame generation func for moviepy.
    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
        latents = all_latents[frame_idx]
        # Get the images (with labels = None)
        images = Gs.run(latents, None, **Gs_kwargs)
        # Generate the grid for this timestamp:
        grid = create_image_grid(images, grid_size)
        # grayscale => RGB
        if grid.shape[2] == 1:
            grid = grid.repeat(3, 2)
        return grid

    # Generate video using make_frame:
    print(f'Generating interpolation video of length: {total_duration} seconds...')
    videoclip = moviepy.editor.VideoClip(make_frame, duration=duration_sec)
    videoclip.write_videofile(os.path.join(outdir, mp4),
                              fps=mp4_fps,
                              codec=mp4_codec,
                              bitrate=mp4_bitrate)

#----------------------------------------------------------------------------

# My extended version of this helper function:
def _parse_num_range(s):
    '''
    Input:
        s (str): Comma separated string of numbers 'a,b,c', a range 'a-c', or
                 even a combination of both 'a,b-c', 'a-b,c', 'a,b-c,d,e-f,...'
    Output:
        nums (list): Ordered list of ascending ints in s, with repeating values
                     deleted (can be modified to not do either of this)
    '''
    # Sanity check 0:
    # In case there's a space between the numbers (impossible due to argparse,
    # but hey, I am that paranoid):
    s = s.replace(' ', '')
    # Split w.r.t comma
    str_list = s.split(',')
    nums = []
    for el in str_list:
        if '-' in el:
            # The range will be 'a-b', so we wish to find both a and b using re:
            range_re = re.compile(r'^(\d+)-(\d+)$')
            match = range_re.match(el)
            # We get the two numbers:
            a = int(match.group(1))
            b = int(match.group(2))
            # Sanity check 1: accept 'a-b' or 'b-a', with a<=b:
            if a <= b: r = [n for n in range(a, b + 1)]
            else: r = [n for n in range(b, a + 1)]
            # Use extend since r will also be an array:
            nums.extend(r)
        else:
            # It's a single number, so just append it:
            nums.append(int(el))
    # Sanity check 2: delete repeating numbers:
    nums = list(set(nums))
    # Return the numbers in ascending order:
    return sorted(nums)

#----------------------------------------------------------------------------

_examples = '''examples:

  # Generate curated MetFaces images without truncation (Fig.10 left)
  python %(prog)s --outdir=out --trunc=1 --seeds=85,265,297,849 \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metfaces.pkl

  # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
  python %(prog)s --outdir=out --trunc=0.7 --seeds=600-605 \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metfaces.pkl

  # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
  python %(prog)s --outdir=out --trunc=1 --seeds=0-35 --class=1 \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/cifar10.pkl

  # Render image from projected latent vector
  python %(prog)s --outdir=out --dlatents=out/dlatents.npz \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/ffhq.pkl
'''

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate images using pretrained network pickle.',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    parser_generate_images = subparsers.add_parser('generate-images', help='Generate images')
    parser_generate_images.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser_generate_images.add_argument('--seeds', type=_parse_num_range, help='List of random seeds', dest='seeds', required=True)
    parser_generate_images.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', dest='truncation_psi', default=0.5)
    parser_generate_images.add_argument('--class', type=int, help='Class label (default: %(default)s)', default='unconditional', dest='class_idx')
    parser_generate_images.add_argument('--create-grid', action='store_true', help='Add flag to save the generated images in a grid', dest='grid')
    parser_generate_images.add_argument('--outdir', help='Root directory for run results (default: %(default)s)', default='out', metavar='DIR')
    parser_generate_images.set_defaults(func=generate_images)

    parser_lerp_video = subparsers.add_parser('lerp-video', help='Generate interpolation video (lerp) between random vectors')
    parser_lerp_video.add_argument('--network', help='Path to network pickle filename', dest='network_pkl', required=True)
    parser_lerp_video.add_argument('--seeds', type=_parse_num_range, help='List of random seeds', dest='seeds', required=True)
    parser_lerp_video.add_argument('--grid-w', type=int, help='Video grid width/columns (default: %(default)s)', default=None, dest='grid_w')
    parser_lerp_video.add_argument('--grid-h', type=int, help='Video grid height/rows (default: %(default)s)', default=None, dest='grid_h')
    parser_lerp_video.add_argument('--trunc', type=float, help='Truncation psi (default: %(default)s)', default=1.0, dest='truncation_psi')
    parser_lerp_video.add_argument('--slowdown', type=int, help='Slowdown the video by this amount; must be a power of 2 (default: %(default)s)', default=1, dest='slowdown')
    parser_lerp_video.add_argument('--duration-sec', type=float, help='Duration of video (default: %(default)s)', default=30.0, dest='duration_sec')
    parser_lerp_video.add_argument('--fps', type=int, help='FPS of generated video (default: %(default)s)', default=30, dest='mp4_fps')
    parser_lerp_video.add_argument('--outdir', help='Root directory for run results (default: %(default)s)', default='out', metavar='DIR')
    parser_lerp_video.set_defaults(func=lerp_video)

    args = parser.parse_args()
    kwargs = vars(args)
    subcmd = kwargs.pop('command')

    if subcmd is None:
        print('Error: missing subcommand.  Re-run with --help for usage.')
        sys.exit(1)

    func = kwargs.pop('func')
    func(**kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
