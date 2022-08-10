from ast import Assert
import os
import json
from argparse import ArgumentParser
from re import split
from tkinter.messagebox import NO
import numpy as np
from tqdm import tqdm
import imageio
from PIL import Image
import time
import config
np.random.seed(0)

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

gpu = get_freer_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
print(f'gpu is {gpu}')

# Import jax only after setting the visible gpu
import jax
import jax.numpy as jnp
import plenoxel
from jax.ops import index, index_update, index_add
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
if __name__ != "__main__":
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.001'


flags = ArgumentParser()


flags.add_argument(
    "--data_dir", '-d',
    type=str,
    default='/data/fabriziov/ct_scans/',
    help="Dataset directory e.g. ct_scans/"
)
flags.add_argument(
    "--expname",
    type=str,
    default="experiment",
    help="Experiment name."
)
flags.add_argument(
    "--scene",
    type=str,
    default='pepper/scans/',
    help="Name of the synthetic scene."
)
flags.add_argument(
    "--log_dir",
    type=str,
    default='jax_logs/',
    help="Directory to save outputs."
)
flags.add_argument(
    "--resolution",
    type=int,
    default=512,
    help="Grid size."
)
flags.add_argument(
    "--ini_rgb",
    type=float,
    default=0.0,
    help="Initial harmonics value in grid."
)
flags.add_argument(
    "--ini_sigma",
    type=float,
    default=0.1,
    help="Initial sigma value in grid."
)
flags.add_argument(
    "--radius",
    type=float,
    default=1.3,
    help="Grid radius. 1.3 works well on most scenes, but ship requires 1.5"
)
flags.add_argument(
    "--harmonic_degree",
    type=int,
    default=-1,
    help="Degree of spherical harmonics. Supports 0, 1, 2, 3, 4."
)
flags.add_argument(
    '--num_epochs',
    type=int,
    default=2,
    help='Epochs to train for.'
)
flags.add_argument(
    '--render_interval',
    type=int,
    default=30,
    help='Render images during test/val step every x images.'
)
flags.add_argument(
    '--val_interval',
    type=int,
    default=2,
    help='Run test/val step every x epochs.'
)
flags.add_argument(
    '--lr_rgb',
    type=float,
    default=None,
    help='SGD step size for rgb. Default chooses automatically based on resolution.'
    )
flags.add_argument(
    '--lr_sigma',
    type=float,
    default=0.0025,
    help='SGD step size for sigma. Default chooses automatically based on resolution.'
    )
flags.add_argument(
    '--physical_batch_size',
    type=int,
    default=2000,
    help='Number of rays per batch, to avoid OOM.'
    )
flags.add_argument(
    '--logical_batch_size',
    type=int,
    default=2000,
    help='Number of rays per optimization batch. Must be a multiple of physical_batch_size.'
    )
flags.add_argument(
    '--jitter',
    type=float,
    default=0.0,
    help='Take samples that are jittered within each voxel, where values are computed with trilinear interpolation. Parameter controls the std dev of the jitter, as a fraction of voxel_len.'
)
flags.add_argument(
    '--uniform',
    type=float,
    default=0.5,
    help='Initialize sample locations to be uniformly spaced at this interval (as a fraction of voxel_len), rather than at voxel intersections (default if uniform=0).'
)
flags.add_argument(
    '--occupancy_penalty',
    type=float,
    default=0.0,
    help='Penalty in the loss term for occupancy; encourages a sparse grid.'
)
flags.add_argument(
    '--reload_epoch',
    type=int,
    default=None,
    help='Epoch at which to resume training from a saved model.'
)
flags.add_argument(
    '--save_interval',
    type=int,
    default=1,
    help='Save the grid checkpoints after every x epochs.'
)
flags.add_argument(
    '--prune_epochs',
    type=int,
    nargs='+',
    default=[],
    help='List of epoch numbers when pruning should be done.'
)
flags.add_argument(
    '--prune_method',
    type=str,
    default='weight',
    help='Weight or sigma: prune based on contribution to training rays, or opacity.'
)
flags.add_argument(
    '--prune_threshold',
    type=float,
    default=0.001,
    help='Threshold for pruning voxels (either by weight or by sigma).'
)
flags.add_argument(
    '--split_epochs',
    type=int,
    nargs='+',
    default=[],
    help='List of epoch numbers when splitting should be done.'
)
flags.add_argument(
    '--interpolation',
    type=str,
    default='trilinear',
    help='Type of interpolation to use. Options are constant, trilinear, or tricubic.'
)
flags.add_argument(
    '--nv',
    action='store_true',
    help='Use the Neural Volumes rendering formula instead of the Max (NeRF) rendering formula.'
)
flags.add_argument(
    '--ct',
    action='store_true',
    help='Optimize sigma only, based on the gt alpha channel.'
)
flags.add_argument(
    '--nonnegative',
    action='store_true',
    help='Clip stored grid values to be nonnegative. Intended for ct.'
)
flags.add_argument(
    '--num_views',
    type=int,
    default=20,
    help='Number of CT projections to train with. Only used with Jerry-CBCT.'
)

FLAGS = flags.parse_args()
data_dir = FLAGS.data_dir + FLAGS.scene
radius = FLAGS.radius
np.random.seed(0)

# This is where I would call for a function or a seperate file that would be able to make a projection matrix for the new datasets
#   the plenoptimize_static file has a def get_jerry(root) function that reads the projection numbers from a csv file.
def get_ct_pepper(root, stage):
    # assert FLAGS.ct
    # all_c2w = []
    all_gt = []
    print('LOAD DATA', root)

    all_c2w, num_proj = config.ct(root) # Get the all_c2w array and the number of projections from the ct function in the config file

    # projection_matrices = np.genfromtxt(os.path.join(root, 'proj_mat.csv'), delimiter=',')  # [719, 12]
    # Loop through the number of projections
    for i in range(1, num_proj+1): 
        index = "{:04d}".format(i)
        im_gt = imageio.imread(os.path.join('/data/fabriziov/ct_scans/pepper/scans', f'Pepper_{index}.tif')).astype(np.float32) / 255.0
        im_gt = 1 - im_gt / np.max(im_gt)
        # print(f'im_gt ranges from {np.min(im_gt)} to {np.max(im_gt)}')
        all_gt.append(im_gt)
    # focal = 75  # cm
    
    focal = 600000
    all_gt = np.asarray(all_gt)

    mask = np.zeros(len(all_c2w))
    # n_train = 25
    # n_test = 10
    # idx = np.random.choice(len(all_c2w), n_train + n_test)
    idx = np.random.choice(len(all_c2w), 25)
    # train_idx = idx[0:n_train]
    # test_idx = idx[n_train:]
    # # mask = np.zeros_like(a)
    mask[idx] = 1
    mask = mask.astype(bool)
    if stage == 'train':
        all_gt = all_gt[mask]
        all_c2w = all_c2w[mask]
        # all_gt = all_gt[train_idx]
        # all_c2w = all_c2w[train_idx]
    elif stage == 'test':
        all_gt = all_gt[~mask]
        all_c2w = all_c2w[~mask]
        # all_gt = all_gt[test_idx]
        # all_c2w = all_c2w[test_idx]
    # print(f'all_gt has shape {all_gt.shape}')
    return focal, all_c2w, all_gt


def get_ct_pomegranate(root, stage):
    all_gt = []
    print('LOAD DATA', root)

    all_c2w, num_proj = config.ct(root) # Get the all_c2w array and the number of projections from the ct function in the config file

    # Loop through the number of projections
    for i in range(1, num_proj+1): 
        index = "{:04d}".format(i)
        im_gt = imageio.imread(os.path.join('/data/fabriziov/ct_scans/pomegranate/scans', f'Pomegranate_{index}.tif')).astype(np.float32) / 255.0
        im_gt = 1 - im_gt / np.max(im_gt)
        all_gt.append(im_gt)
    
    focal = 600000
    all_gt = np.asarray(all_gt)

    mask = np.zeros(len(all_c2w))
    # n_train = 50
    # n_test = 10
    # idx = np.random.choice(len(all_c2w), n_train + n_test)
    idx = np.random.choice(len(all_c2w), 50)
    # train_idx = idx[0:n_train]
    # test_idx = idx[n_train:]
    # # mask = np.zeros_like(a)
    mask[idx] = 1
    mask = mask.astype(bool)
    if stage == 'train':
        all_gt = all_gt[mask]
        all_c2w = all_c2w[mask]
    elif stage == 'test':
        all_gt = all_gt[~mask]
        all_c2w = all_c2w[~mask]

    return focal, all_c2w, all_gt


def get_data(root, stage):

    if root == '/data/fabriziov/ct_scans/pepper/scans/':
        focal, all_c2w, all_gt = get_ct_pepper(root, stage)  
        # idx = np.random.choice(len(all_c2w), FLAGS.num_views) # Pick a subset of the data at random
        # return focal, all_c2w[idx], all_gt[idx]
        return focal, all_c2w, all_gt

    elif root == '/data/fabriziov/ct_scans/pomegranate/scans/':
        focal, all_c2w, all_gt = get_ct_pomegranate(root, stage)
        return focal, all_c2w, all_gt
    
    all_c2w = []
    all_gt = []

    data_path = os.path.join(root, stage)
    data_json = os.path.join(root, 'transforms_' + stage + '.json')
    print('LOAD DATA', data_path)
    j = json.load(open(data_json, 'r'))

    for frame in tqdm(j['frames']):
        fpath = os.path.join(data_path, os.path.basename(frame['file_path']) + '.png')
        c2w = frame['transform_matrix']
        im_gt = imageio.imread(fpath).astype(np.float32) / 255.0
        # Adding next line from the plenotimize_static file, said to be for the linear version (whatever that means)
        # This line makes more static-y images that seen to have depth
        # im_gt = jnp.concatenate([-jnp.log(1 - 0.99*im_gt[..., 3:]), jnp.zeros((im_gt.shape[0], im_gt.shape[1], 2))], -1)

        # This line makes more white images that look cleaner(sara said to ise this one)
        im_gt = jnp.concatenate([im_gt[..., 3:], jnp.zeros((im_gt.shape[0], im_gt.shape[1], 2))], -1) # If we want to train with alpha (whatever that means)
        # Might want to comment out next line; plenoptimize_static file says to only use it when you want to train with color
        # im_gt = im_gt[..., :3] * im_gt[..., 3:] + (1.0 - im_gt[..., 3:])
        all_c2w.append(c2w)
        all_gt.append(im_gt)
    focal = 0.5 * all_gt[0].shape[1] / np.tan(0.5 * j['camera_angle_x'])
    all_gt = np.asarray(all_gt)
    all_c2w = np.asarray(all_c2w)
    return focal, all_c2w, all_gt


if __name__ == "__main__":
    # print(f'the data directory is {data_dir}')
    focal, train_c2w, train_gt = get_data(data_dir, "train")
    test_focal, test_c2w, test_gt = get_data(data_dir, "test")
    assert focal == test_focal
    H, W = train_gt[0].shape[:2]
    n_train_imgs = len(train_c2w)
    n_test_imgs = len(test_c2w)


log_dir = FLAGS.log_dir + FLAGS.expname
os.makedirs(log_dir, exist_ok=True)


automatic_lr = False
if FLAGS.lr_rgb is None or FLAGS.lr_sigma is None:
    automatic_lr = True
# Added two If statements, the content of them however, was in the original code
if FLAGS.lr_rgb is None:
    FLAGS.lr_rgb = 150 * (FLAGS.resolution ** 1.75)
if FLAGS.lr_sigma is None:
    FLAGS.lr_sigma = 51.5 * (FLAGS.resolution ** 2.37)


if FLAGS.reload_epoch is not None:
    reload_dir = os.path.join(log_dir, f'epoch_{FLAGS.reload_epoch}')
    print(f'Reloading the grid from {reload_dir}')
    data_dict = plenoxel.load_grid(dirname=reload_dir, sh_dim = (FLAGS.harmonic_degree + 1)**2)
else:
    print(f'Initializing the grid')
    data_dict = plenoxel.initialize_grid(resolution=FLAGS.resolution, ini_rgb=FLAGS.ini_rgb, ini_sigma=FLAGS.ini_sigma, harmonic_degree=FLAGS.harmonic_degree)


# low-pass filter the ground truth image so the effective resolution matches twice that of the grid
def lowpass(gt, resolution):
    if gt.ndim > 3:
        print(f'lowpass called on image with more than 3 dimensions; did you mean to use multi_lowpass?')
    H = gt.shape[0]
    W = gt.shape[1]
    im = Image.fromarray((np.squeeze(np.asarray(gt))*255).astype(np.uint8))
    im = im.resize(size=(resolution*2, resolution*2))
    im = im.resize(size=(H, W))
    return np.asarray(im) / 255.0


# low-pass filter a stack of images where the first dimension indexes over the images
def multi_lowpass(gt, resolution):
    if gt.ndim <= 3:
        print(f'multi_lowpass called on image with 3 or fewer dimensions; did you mean to use lowpass instead?')
    H = gt.shape[-3]
    W = gt.shape[-2]
    clean_gt = np.copy(gt)
    for i in range(len(gt)):
        im = Image.fromarray(np.squeeze(gt[i,...] * 255).astype(np.uint8))
        im = im.resize(size=(resolution*2, resolution*2))
        # In the plenoptimize_static file the line below was changed because jerry needed it, not sure if I would need to change it for 
        #   making ct scans in general or is this is just a jerry thing
        # OMG LOL IT IS A CT THING HAHAHA
        # Keep this line as (W, H) and not (H, W)!!!
        # or not ???
        # now it only works if it is (H,W)
        # nvm idk which is right
        # print(im)
        im = im.resize(size=(W, H))
        im = np.asarray(im) / 255.0
        clean_gt[i,...] = im
    return clean_gt


def get_loss(data_dict, c2w, gt, H, W, focal, resolution, radius, harmonic_degree, jitter, uniform, key, sh_dim, occupancy_penalty, interpolation, nv):
    rays = plenoxel.get_rays(H, W, focal, c2w)
    rgb, disp, acc, weights, voxel_ids = plenoxel.render_rays(data_dict, rays, resolution, key, radius, harmonic_degree, jitter, uniform, interpolation, nv)
    mse = jnp.mean((rgb - lowpass(gt, resolution))**2)
    # indices, data = data_dict
    loss = mse + occupancy_penalty * jnp.mean(jax.nn.relu(data_dict[-1]))
    return loss

# The plenotimize_static file has this added line to the top of this function, not sure if is necessary
@jax.partial(jax.jit, static_argnums=(3,4,5,6,7,9,11,12))
def get_loss_rays(data_dict, rays, gt, resolution, radius, harmonic_degree, jitter, uniform, key, sh_dim, occupancy_penalty, interpolation, nv):
    rgb, disp, acc, weights, voxel_ids = plenoxel.render_rays(data_dict, rays, resolution, key, radius, harmonic_degree, jitter, uniform, interpolation, nv)
    # The plenoptimize_static file has this next line a little bit different, for the "alpha" channel only ???
    # mse = jnp.mean((rgb - gt)**2)
    mse = jnp.mean((acc - gt[...,0])**2)# Optimize the alpha channel only
    # indices, data = data_dict
    loss = mse + occupancy_penalty * jnp.mean(jax.nn.relu(data_dict[-1]))
    return loss


def get_rays_np(H, W, focal, c2w):
    # all_c2w, num_proj, c = config.ct()

    # SrcToObject = 595.033142
    # SrcToDetector = 983.0
    # radius = SrcToDetector - SrcToObject
    i, j = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5, indexing='xy')
    dirs = np.stack([ -np.ones_like(i), (i-W*.5)/focal, -(j-H*.5)/focal], -1)
    # Rotate ray directions from camera frame to the world frame
    # rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = dirs @ c2w[:3, :3].T
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d

    # i, j = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5, indexing='xy')
    # dirs = np.stack([ -np.ones_like(i), (i - W*.5) / focal, (j - H*.5) / focal], axis=-1)
    # dirs = dirs.numpy()
    # # rays_d = dirs @ c2w[:3, :3].T
    # # rays_d = np.sum(dirs[..., np.newaxis, :] * -c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # rays_d = dirs @ c2w[:3, :3].T
    # # Translate camera frame's origin to the world frame. It is the origin of all rays.
    # rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))

    # c2w[:3,3] += [radius*np.cos(0), radius*np.sin(0), 0]
    # print(f'rays_o after is  {rays_o}')
    # print(f'rays_d after is  {rays_d}')
    # I literally done even know what to do anymore with this code
    # there must be a correct way to do this, but i don't know it :(
    # I got it :)
    # return rays_o, rays_d#[0,0]


def render_pose_rays(data_dict, c2w, H, W, focal, resolution, radius, harmonic_degree, jitter, uniform, key, sh_dim, batch_size, interpolation, nv):
    rays_o, rays_d = get_rays_np(H, W, focal, c2w)
    # print (f'\nc2w is {c2w[0]}')

    # print(f'rays_o after is  {rays_o}')
    
    # print(f'\nrays_d after is  {rays_d}')
    rays_o = np.reshape(rays_o, [-1,3])
    rays_d = np.reshape(rays_d, [-1,3])

    # print(f'\nrays_o at {0} is {rays_o[0]}')

    rgbs = []
    disps = []
    for i in range(int(np.ceil(H*W/batch_size))):
        start = i*batch_size
        stop = min(H*W, (i+1)*batch_size)
        if jitter > 0:
            rgbi, dispi, acci, weightsi, voxel_idsi = jax.lax.stop_gradient(plenoxel.render_rays(data_dict, (rays_o[start:stop], rays_d[start:stop]), resolution, key[start:stop], radius, harmonic_degree, jitter, uniform, interpolation, nv))
        else:
            rgbi, dispi, acci, weightsi, voxel_idsi = jax.lax.stop_gradient(plenoxel.render_rays(data_dict, (rays_o[start:stop], rays_d[start:stop]), resolution, None, radius, harmonic_degree, jitter, uniform, interpolation, nv))
        # Added this next line from the plenoptimize_static file because it was under an if statment for ct files
        rgbi = jnp.concatenate([acci[:,jnp.newaxis], jnp.zeros_like(acci)[:,jnp.newaxis], jnp.zeros_like(acci)[:,jnp.newaxis]], axis=-1)
        rgbs.append(rgbi)
        disps.append(dispi)

    
    rgb = jnp.reshape(jnp.concatenate(rgbs, axis=0), (H, W, 3))
    disp = jnp.reshape(jnp.concatenate(disps, axis=0), (H, W))
    # print("\nBefore: ")
    # print(f'rgb ranges from {np.min(rgb)} to {np.max(rgb)}')
    # print(f'disp ranges from {np.min(disp)} to {np.max(disp)}')
    return rgb, disp, None, None


def run_test_step(i, data_dict, test_c2w, test_gt, H, W, focal, FLAGS, key, name_appendage=''):
    print('Evaluating')
    sh_dim = (FLAGS.harmonic_degree + 1)**2
    tpsnr = 0.0
    for j, (c2w, gt) in tqdm(enumerate(zip(test_c2w, test_gt))):
        rgb, disp, _, _ = render_pose_rays(data_dict, c2w, H, W, focal, FLAGS.resolution, radius, FLAGS.harmonic_degree, FLAGS.jitter, FLAGS.uniform, key, sh_dim, FLAGS.physical_batch_size, FLAGS.interpolation, FLAGS.nv)
        # assert False
        #Use these prints if you need to show shapes and ranges/////////////////////////////////////// 
        
        # Adding these next two lines from the plenoptimize__static file because it was under an if statment for ct files
        rgb = jnp.concatenate((rgb[...,0,jnp.newaxis], rgb[...,0,jnp.newaxis], rgb[...,0,jnp.newaxis]), axis=-1)
        gt = jnp.concatenate((gt[...,jnp.newaxis], gt[...,jnp.newaxis], gt[...,jnp.newaxis]), axis=-1)
        
        # print(f'\nrgb ranges from {np.min(rgb)} to {np.max(rgb)}')
        # print(f'gt ranges from {np.min(gt)} to {np.max(gt)}')

        mse = jnp.mean((rgb - gt)**2)
        psnr = -10.0 * np.log(mse) / np.log(10.0)
        tpsnr += psnr

        if FLAGS.render_interval > 0 and j % FLAGS.render_interval == 0:
            # The plenoptimize_static file doesnt have this next line but another line instead of it and changed a bit of the line after
            disp3 = jnp.concatenate((disp[...,jnp.newaxis], disp[...,jnp.newaxis], disp[...,jnp.newaxis]), axis=2)
            # print(f'disp3 has shape {disp3.shape}')

            # print(f'disp3 ranges from {np.min(disp3)} to {np.max(disp3)}')

            # vis = jnp.concatenate((gt, rgb, disp3), axis=1)
            # vis = np.asarray((vis * 255)).astype(np.uint8)
            # imageio.imwrite(f"{log_dir}/{j:04}_{i:04}{name_appendage}.png", vis)
            # print(f'gt ranges from {jnp.min(gt)} to {jnp.max(gt)} and prediction ranges from {jnp.min(rgb)} to {jnp.max(rgb)} and disp ranges from {jnp.min(disp)} to {jnp.max(disp)}')
            vis = jnp.concatenate((gt, rgb, disp3), axis = 1)
            vis = np.asarray((vis*255)).astype(np.uint8)

            # print(f'vis ranges from {np.min(vis)} to {np.max(vis)}')

            imageio.imwrite(f"{log_dir}/{j:04}_{i:04}{name_appendage}.png", vis)
        del rgb, disp
    # tpsnr /= n_test_imgs
    tpsnr /= len(test_c2w)
    return tpsnr


def update_grid(old_grid, lr, grid_grad):
    if FLAGS.nonnegative:
        return jnp.clip(index_add(old_grid, index[...], -1 * lr * grid_grad), a_min=0)
    else:
        return index_add(old_grid, index[...], -1 * lr * grid_grad)


def update_grids(old_grid, lrs, grid_grad):
    # Added these two lines because the plenoptimize_static code has it under an in statement for ct
    old_grid[-1] = update_grid(old_grid[-1], lrs[-1], grid_grad[-1])# Only updates the sigma grid for CT
    return old_grid
    # for i in range(len(old_grid)):
    #     old_grid[i] = index_add(old_grid[i], index[...], -1 * lrs[i] * grid_grad[i])
    # return old_grid

# ---------------------------------------LEFT OFF HERE------------------------------------------------------
if FLAGS.physical_batch_size is not None:
    # print(f'precomputing all the training rays')
    # # Precompute all the training rays and shuffle them
    # rays = np.stack([get_rays_np(H, W, focal, p) for p in train_c2w[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
    # rays_rgb = np.concatenate([rays, multi_lowpass(train_gt[:,None], FLAGS.resolution).astype(np.float32)], 1) # [N, ro+rd+rgb, H, W,   3]
    # rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
    # rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
    # rays_rgb = rays_rgb.take(np.random.permutation(rays_rgb.shape[0]), axis=0)
    if False:
        print(f'reloading saved rays')
        rays_rgb = np.load('rays.npy')
    else:
        print(f'precomputing all the training rays')
        # Precompute all the training rays and shuffle them
        t0 = time.time()
        
        # rays = jax.vmap(fun=lambda p: plenoxel.get_rays(H, W, focal, p), in_axes=0, out_axes=0)(train_c2w[:,:3,:4])  # OOM
        rays = np.stack([get_rays_np(H, W, focal, p) for p in train_c2w[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]  # 42 seconds
        
# ASSERTED FALSE HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # print(rays[0,:,0,0,:])
        # assert False
        # print(rays)
        t1 = time.time()
        print(f'stack took {t1 - t0} seconds')
        # print(f'train_c2w has shape {train_c2w.shape}')
        # print(f'train_gt has shape {train_gt.shape}')
        train_gt = np.concatenate([train_gt[...,None], train_gt[...,None], train_gt[...,None]], -1)
        # print(f'train_gt has shape {train_gt.shape}')
        # print(multi_lowpass(train_gt[:,None], FLAGS.resolution).astype(np.float32).shape)
        # print(f'rays has shape {rays.shape}')
        rays_rgb = np.concatenate([rays, multi_lowpass(train_gt[:,None], FLAGS.resolution).astype(np.float32)], 1)  # [N, ro+rd+rgb, H, W, 3]  # 19 seconds
        t2 = time.time()
        print(f'concatenate took {t2 - t1} seconds')
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3] 
        t3 = time.time()
        print(f'transpose took {t3 - t2} seconds')
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]  # 12 seconds
        t4 = time.time()
        print(f'reshape took {t4 - t3} seconds')
        rays_rgb = rays_rgb.take(np.random.permutation(rays_rgb.shape[0]), axis=0)  # 34 seconds
        print(f'permutation took {time.time() - t4} seconds')
        # np.save('rays.npy', rays_rgb)


# print(f'generating random keys')
# split_keys_partial = jax.vmap(jax.random.split, in_axes=0, out_axes=0)
# split_keys = jax.vmap(split_keys_partial, in_axes=1, out_axes=1)
# if FLAGS.physical_batch_size is None:
#     keys = jax.vmap(jax.vmap(jax.random.PRNGKey, in_axes=0, out_axes=0), in_axes=1, out_axes=1)(jnp.reshape(jnp.arange(800*800), (800,800)))
# else: 
#     keys = jax.vmap(jax.random.PRNGKey, in_axes=0, out_axes=0)(jnp.arange(FLAGS.physical_batch_size))
# render_keys = jax.vmap(jax.random.PRNGKey, in_axes=0, out_axes=0)(jnp.arange(800*800))
if FLAGS.jitter == 0:
    render_keys = None
    keys = None
    split_keys = None
    split_keys_partial = None

@jax.jit
def rmsprop_update(avg_g, data_grad):
    # Only used for CT, so only care about sigma grid
    # return [0.9 * (avg_g_i) + 0.1 * (g_i**2) for (avg_g_i, g_i) in zip(avg_g, data_grad)]
    avg_g[-1] = 0.9 * avg_g[-1] + 0.1*data_grad[-1]**2
    return avg_g


def main():
    global rays_rgb, keys, render_keys, data_dict, FLAGS, radius, train_c2w, train_gt, test_c2w, test_gt, automatic_lr
    start_epoch = 0
    sh_dim = (FLAGS.harmonic_degree + 1)**2
    if FLAGS.reload_epoch is not None:
        start_epoch = FLAGS.reload_epoch + 1
# The file pleoptomize_static does not have all this part---------------------------------------        
    # if np.isin(FLAGS.reload_epoch, FLAGS.prune_epochs):
    #     data_dict = plenoxel.prune_grid(data_dict, method=FLAGS.prune_method, threshold=FLAGS.prune_threshold, train_c2w=train_c2w, H=H, W=W, focal=focal, batch_size=FLAGS.physical_batch_size, resolution=FLAGS.resolution, key=render_keys, radius=FLAGS.radius, harmonic_degree=FLAGS.harmonic_degree, jitter=FLAGS.jitter, uniform=FLAGS.uniform, interpolation=FLAGS.interpolation)
    # if np.isin(FLAGS.reload_epoch, FLAGS.split_epochs):
    #     data_dict = plenoxel.split_grid(data_dict)
    #     FLAGS.resolution = FLAGS.resolution * 2
    #     if automatic_lr:
    #         FLAGS.lr_rgb = 150 * (FLAGS.resolution ** 1.75)
    #         FLAGS.lr_sigma = 51.5 * (FLAGS.resolution ** 2.37)
# -----------------------------------------------------------------------------------------------
    avg_g = [0 for g_i in data_dict]
    
    for i in range(start_epoch, FLAGS.num_epochs):
        # Shuffle data before each epoch
        if FLAGS.physical_batch_size is None:
            temp = list(zip(train_c2w, train_gt))
            np.random.shuffle(temp)
            train_c2w, train_gt = zip(*temp)
        else:
            assert FLAGS.logical_batch_size % FLAGS.physical_batch_size == 0
            # Shuffle rays over all training images
            rays_rgb = rays_rgb.take(np.random.permutation(rays_rgb.shape[0]), axis=0)

        print('epoch', i)
        # plenoptimize_static does not have this next line
        # indices, data = data_dict
        if FLAGS.physical_batch_size is None:
            occupancy_penalty = FLAGS.occupancy_penalty / len(train_c2w)
            for j, (c2w, gt) in tqdm(enumerate(zip(train_c2w, train_gt)), total=len(train_c2w)):
                if FLAGS.jitter > 0:
                    splitkeys = split_keys(keys)
                    keys = splitkeys[...,0,:]
                    subkeys = splitkeys[...,1,:]
                else:
                    subkeys = None
                # This next line is slightly differnet in the plenoptimize_static code
                # mse, data_grad = jax.value_and_grad(lambda grid: get_loss((indices, grid), c2w, gt, H, W, focal, FLAGS.resolution, radius, FLAGS.harmonic_degree, FLAGS.jitter, FLAGS.uniform, subkeys, sh_dim, occupancy_penalty, FLAGS.interpolation, FLAGS.nv))(data) 
                mse, data_grad = jax.value_and_grad(lambda grid: get_loss(grid, c2w, gt, H, W, focal, FLAGS.resolution, radius, FLAGS.harmonic_degree, FLAGS.jitter, FLAGS.uniform, subkeys, sh_dim, occupancy_penalty, FLAGS.interpolation, FLAGS.nv))(data_dict) 

        else:
            occupancy_penalty = FLAGS.occupancy_penalty / (len(rays_rgb) // FLAGS.logical_batch_size)
            for k in tqdm(range(len(rays_rgb) // FLAGS.logical_batch_size)):
            # for k in tqdm(range(1000)):
            # for k in tqdm(range(0)):
                logical_grad = None
                for j in range(FLAGS.logical_batch_size // FLAGS.physical_batch_size):
                    if FLAGS.jitter > 0:
                        splitkeys = split_keys_partial(keys)
                        keys = splitkeys[...,0,:]
                        subkeys = splitkeys[...,1,:]
                    else:
                        subkeys = None
                    effective_j = k*(FLAGS.logical_batch_size // FLAGS.physical_batch_size) + j
                    batch = rays_rgb[effective_j*FLAGS.physical_batch_size:(effective_j+1)*FLAGS.physical_batch_size] # [B, 2+1, 3*?]
                    batch_rays, target_s = (batch[:,0,:], batch[:,1,:]), batch[:,2,:]
                    # This next line is slightly differnet in the plenoptimize_static code
                    # mse, data_grad = jax.value_and_grad(lambda grid: get_loss_rays((indices, grid), batch_rays, target_s, FLAGS.resolution, radius, FLAGS.harmonic_degree, FLAGS.jitter, FLAGS.uniform, subkeys, sh_dim, occupancy_penalty, FLAGS.interpolation, FLAGS.nv))(data) 
                    mse, data_grad = jax.value_and_grad(lambda grid: get_loss_rays(grid, batch_rays, target_s, FLAGS.resolution, radius, FLAGS.harmonic_degree, FLAGS.jitter, FLAGS.uniform, subkeys, sh_dim, occupancy_penalty, FLAGS.interpolation, FLAGS.nv))(data_dict) 

                    if FLAGS.logical_batch_size > FLAGS.physical_batch_size:
                        if logical_grad is None:
                            logical_grad = data_grad
                        else:
                            logical_grad = [a + b for a, b in zip(logical_grad, data_grad)]
                        del data_grad
                    del mse, batch, batch_rays, target_s, subkeys, effective_j
                lrs = [FLAGS.lr_rgb / (FLAGS.logical_batch_size // FLAGS.physical_batch_size)]*sh_dim + [FLAGS.lr_sigma / (FLAGS.logical_batch_size // FLAGS.physical_batch_size)]

                lrs  = [lr * np.cos(k /((len(rays_rgb) // FLAGS.logical_batch_size)+ 10) * (np.pi/2)) for lr in lrs]
                
                # avg_g = [0.9 * (avg_g_i) + 0.1 * (g_i**2) for (avg_g_i, g_i) in zip(avg_g, data_grad)]
                avg_g = rmsprop_update(avg_g, data_grad)

                if FLAGS.logical_batch_size > FLAGS.physical_batch_size:
                    data_dict = update_grids(data_dict, lrs, logical_grad)
                    del logical_grad
                else:
                    data_dict[-1] = update_grid(data_dict[-1], lrs[-1], data_grad[-1]/ (jnp.sqrt(avg_g[-1]) + 1e-10))  
                    del data_grad, logical_grad

        if i % FLAGS.save_interval == FLAGS.save_interval - 1 or i == FLAGS.num_epochs - 1:
            print(f'Saving checkpoint at epoch {i}')
            plenoxel.save_grid(data_dict, os.path.join(log_dir, f'epoch_{i}'))

        if i % FLAGS.val_interval == FLAGS.val_interval - 1 or i == FLAGS.num_epochs - 1:
            validation_psnr = run_test_step(i + 1, data_dict, test_c2w, test_gt, H, W, focal, FLAGS, render_keys)
            print(f'at epoch {i}, test psnr is {validation_psnr}')
        if start_epoch == FLAGS.num_epochs:
            # render_interval = FLAGS.render_interval
            # FLAGS.render_interval = 1000  # Don't save training views
            # train_psnr = run_test_step(start_epoch + 1, data_dict, train_c2w, train_gt, H, W, focal, FLAGS, render_keys)
            # print(f'at epoch {start_epoch}, train psnr is {train_psnr}')
            # FLAGS.render_interval = render_interval
            validation_psnr = run_test_step(start_epoch + 1, data_dict, test_c2w, test_gt, H, W, focal, FLAGS, render_keys)
            print(f'at epoch {start_epoch}, test psnr is {validation_psnr}')
    
if __name__ == "__main__":
    main()