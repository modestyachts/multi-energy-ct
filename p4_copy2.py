from ast import Assert
import os
import json
from argparse import ArgumentParser
from re import split
# from tkinter.messagebox import NO
import numpy as np
from tqdm import tqdm
import imageio
from PIL import Image
import time
import config as config
import config2 as config2
import math
import tifffile
np.random.seed(0)

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory | grep Free >tmp')
    memory_available = [int(x.split()[2]) for i, x in enumerate(open('tmp', 'r').readlines()) if i % 3 == 0]
    print(f"memory available = {memory_available}")
    print(f"np.argmax() = {np.argmax(memory_available)}")
    return np.argmax(memory_available)

gpu = get_freer_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
print(f'gpu is {gpu}')

# Import jax only after setting the visible gpu
import jax
print(f"jax devices is {jax.devices()}")
import jax.numpy as jnp
from functools import partial
import plenoxel_og_copy2
# from jax.ops import index, index_update, index_add
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
if __name__ != "__main__":
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.001'


flags = ArgumentParser()


flags.add_argument(
    "--data_dir", '-d',
    type=str,
    default='/data/datasets/', #for corrected projections and spike
    # default = '/data/fabriziov/ct_scans/organSegs/', #for orgsegs
    help="Dataset directory e.g. ct_scans/"
)
flags.add_argument(
    "--expname",
    type=str,
    # default="experiment",
    # default = "corr_proj_img", # for the Corrected projections
    # default = "source_proj_img", # for the source projections
    # default = "jerry_gt_img", # for jerry_gt (fake projections)
    # default = "newjerry2",
    # default = 'spike_img', #in tmux 1
    default = 'fullradius9_test_spike_tiff_res900_epoch1_tv0.001_nonneg_views720',
    # default = 'test_jerry_tiff_res64_epoch4_split2_tv0.001_nonneg',
    # default = 'organSegs_img1', # For organ Segmentations dataset
    help="Experiment name."
)
flags.add_argument(
    "--scene",
    type=str,
    # default='Corrected_Projections/', #for jerry corrected projections
    # default='scans/', # For organ Segmentations 
    default='spike-cbct/', # For Spike
    # default = 'scans/', #for jerry
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
    default=50, #900,  # 50 is for synthetic ct dataset
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
    "--radius", # affects resolution
    type=float,
    default=5, #9, # 6 for jerry and spike and 1.3 for orgSegs  # 5 is for synthetic ct dataset
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
    default=1,
    help='Epochs to train for.'
)
flags.add_argument(
    '--render_interval',
    type=int,
    default = 30, #change to 1 to get the projection images 
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
    default=0.001, #0.001 for orgSegs and jerry, but now trying 0.01 for jerry in tmux one
    # default=0.01, # for jerry has to be less than 0.05, currently trying 0.005 in tmux 1 & 0.001 in tmux 3
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
    default=[], # Try w/ 0, 1, or 2
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
flags.add_argument(
    '--cut_cube',
    action='store_true',
    help='cuts the cube in half and halves the radius aswell'
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

    print(all_c2w.shape)
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
    idx = np.random.choice(len(all_c2w), 25, replace = False)
    # train_idx = idx[0:n_train]
    # test_idx = idx[n_train:]
    # # mask = np.zeros_like(a)
    mask[idx] = 1
    mask = mask.astype(bool)
    if stage == 'train':
        all_gt = all_gt[mask]
        print(f'all_gt[mask] is {all_gt.shape}')
        all_c2w = all_c2w[mask]
        print(f'all_c2w[mask] is {all_c2w.shape}')
        # all_gt = all_gt[train_idx]
        # all_c2w = all_c2w[train_idx]
    elif stage == 'test':
        all_gt = all_gt[~mask]
        all_c2w = all_c2w[~mask]
        # all_gt = all_gt[test_idx]
        # all_c2w = all_c2w[test_idx]

    print(f'all_gt has shape {all_gt.shape}')
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
    idx = np.random.choice(len(all_c2w), 50, replace = False)
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


def get_ct_organSegs(root, stage):
    all_gt = []
    print('LOAD DATA', root)

    all_c2w, num_proj = config2.ct(root) # Get the all_c2w array and the number of projections from the ct function in the config file

    print(f'number of projections: {num_proj}')
    print(f'all_c2w has shape {all_c2w.shape}')
    print(f'all_c2w has size {all_c2w.size}')

    for i in range(0, num_proj): 
        # im_gt = 1 - imageio.imread(filenames[i]).astype(np.float32) / 255.0
        # im_gt = im_gt - np.min(im_gt) # Normalize so empty space is always zero
        index = "{:04d}".format(i)
        im_gt = imageio.imread(os.path.join('/data/fabriziov/ct_scans/organSegs/scans', f'{index}_0001.png')).astype(np.float32) / 255.0
        # im_gt = 1 - im_gt / np.max(im_gt)
        all_gt.append(im_gt[...,0])
        # all_gt.append(im_gt)

    # focal = 75  # cm
    # focal = 2.85 / 0.024
    # focal = 500
    focal = 750
    # focal = 300000
    all_gt = np.asarray(all_gt)
    print(f'all_gt has shape {all_gt.shape}')

    mask = np.zeros(len(all_c2w))
    
    idx = np.random.choice(len(all_c2w), 100, replace = False)

    mask[idx] = 1

    mask = mask.astype(bool)

    # train and test can be commented out ot get the full 360 ground truth projections
    if stage == 'train':
        all_gt = all_gt[mask]
        all_c2w = all_c2w[mask]
    elif stage == 'test':
        all_gt = all_gt[~mask]
        all_c2w = all_c2w[~mask]

    print(f'all_gt has shape {all_gt.shape}')
    return focal, all_c2w, all_gt

# This is a dataloader for the ct datasets
def get_ct_jerry(root, stage):
    all_gt = []
    all_c2w =[]
    print('LOAD DATA', root)

    # all_c2w, num_proj = config2.ct(root) # Get the all_c2w array and the number of projections from the ct function in the config file
    # all_c2w, num_proj = config2.ct('/data/datasets/jerry-cbct/projections/') # Source
    all_c2w, num_proj = config2.ct('/data/sfk/plenoxel-ct/jerry_gt')  #might have ran tmux3 with source instead of this one

    # proj_mat = np.genfromtxt(os.path.join('/data/datasets/jerry-cbct/', 'proj_mat.csv'), delimiter=',')  
    print(f'number of projections: {num_proj}')
    print(f'all_c2w has shape {all_c2w.shape}')
    print(f'all_c2w has size {all_c2w.size}')

    for i in range(0, num_proj): 
        # im_gt = 1 - imageio.imread(filenames[i]).astype(np.float32) / 255.0
        # im_gt = im_gt - np.min(im_gt) # Normalize so empty space is always zero
        index = "{:04d}".format(i)
        
        # try with the reprojections
        # im_gt = imageio.imread(os.path.join('/data/datasets/Corrected_Projections', f'Cor_Proj{index}.png')).astype(np.float32) / 255.0
        # im_gt = imageio.imread(os.path.join('/data/datasets/jerry-cbct/projections', f'Source_Projections{index}.png')).astype(np.float32) / 255.0
        im_gt = imageio.imread(os.path.join('/data/sfk/plenoxel-ct/jerry_gt', f'{index}_0001.png')).astype(np.float32) / 255.0
        # im_gt = 1 - im_gt / np.max(im_gt)
        # im_gt = 1 - im_gt
        # w2c = np.reshape(num_proj[i], (3,4))
        # w2c[:,-1] = (w2c[:,-1] - [400, 220, 200])
        # c2w = np.linalg.inv(np.concatenate([w2c, [[0,0,0,1]]], axis=0))

        all_gt.append(im_gt[..., 0])
        # all_gt.append(im_gt)
    

    # all_gt = all_gt[15:]
    # all_c2w = all_c2w[15:]

    # focal = 50
    # focal = 75  # cm
    # focal = 11
    # focal = 2.85 / 0.024 # no
    # focal = 250
    # focal = 500 # meh
    # focal = 750
    # focal = 1187.5
    focal = 2375
    # focal = 11875 # do not use 
    # focal = 2000
    # focal = 600000
    # focal = 1000000
    all_gt = np.asarray(all_gt)
    print(f'all_gt has shape {all_gt.shape}')
    # all_gt = np.asarray(all_c2w)

    mask = np.zeros(len(all_c2w))
    
    idx = np.random.choice(len(all_c2w), 100, replace = False) #tmux one
    # idx = np.random.choice(len(all_c2w), 50, replace = False) #tmux two

    mask[idx] = 1

    mask = mask.astype(bool)

    print(f'all_gt has shape {all_gt.shape}')

    # train and test can be commented out ot get the full 360 ground truth projections
    if stage == 'train':
        all_gt = all_gt[mask]
        all_c2w = all_c2w[mask]
    elif stage == 'test':
        all_gt = all_gt[~mask]
        all_c2w = all_c2w[~mask]

    # all_gt = all_gt[15:]
    # all_c2w = all_c2w[15:]
    print(f'all_gt has shape {all_gt.shape}')
    return focal, all_c2w, all_gt

def get_ct_jerry2(root, stage, max_projections, xoff, yoff, zoff):
    all_w2c = []
    all_gt = []

    print('LOAD DATA', root)
    
    # Would be good to test this for Jerry and Spike
    projection_matrices = np.genfromtxt(os.path.join('/home/fabriz/data/jerry/', 'proj_mat_jerry.csv'), delimiter=',')  # [719, 12]
    print(f'proj mat len is {len(projection_matrices)}')
    
    #Traslation matrix along x,y,z
    Tz = np.matlib.zeros((4,4))
    Tz[0,0]=1.0
    Tz[1,1]=1.0
    Tz[2,2]=1.0
    Tz[3,3]=1.0
    Tz[0,3]=-xoff #test
    Tz[1,3]=-yoff #test
    Tz[2,3]=-zoff #test

    # tif_proj = tifffile.imread('/data/datasets/jerry_tiff/Source_Projections.tif')
    tif_proj = tifffile.imread('/home/fabriz/data/jerry/jerry_corr_src_proj.tif')

    # reads #max_projections projection images
    for i in range(len(projection_matrices)-1): 
        index = "{:04d}".format(i)
        # im_gt = imageio.imread(os.path.join('/data/datasets/newJerryProj', f'NewJerryProj_{index}.png')).astype(np.float32) / 255.0
        # im_gt = imageio.imread(os.path.join('/data/datasets/New_Corrected_Projections', f'New_Cor_Proj{index}.png')).astype(np.float32) / 255.0
        
        im_gt = tif_proj[i,:,:]

        im_gt = 1 - im_gt

        # projection matrices P_(3,4)
        w2c = np.reshape(projection_matrices[i], (3,4))
        w2c = np.matmul(w2c,Tz) #applico una traslazione per centrate il volume
        #w2c[:,-1] = (w2c[:,-1] - [400, 220, 200])
        
        all_w2c.append(w2c)
        all_gt.append(im_gt)

    all_gt = np.asarray(all_gt)
    all_w2c = np.asarray(all_w2c)

    focal = 100 # 150 in tmux one, 100 in tmux two
    
    mask = np.zeros(len(all_w2c))
    print(f'max is {len(all_w2c)}')
    idx = np.random.choice(len(all_w2c), max_projections, replace = False) # Just look at a subset of projections, to save time/memory for debuggin
    mask[idx] = 1
    mask = mask.astype(bool)

    # train and test can be commented out to get the full 360 ground truth projections
    if stage == 'train':
        all_gt = all_gt[mask]
        all_w2c = all_w2c[mask]
    elif stage == 'test':
        all_gt  = all_gt[~mask]
        all_w2c = all_w2c[~mask]

    return focal, all_w2c,all_gt

def get_ct_spike(root, stage):
    # assert FLAGS.ct
    all_c2w = []
    all_gt = []
    print('using spike')
    print('LOAD DATA', root)
    # filenames = glob.glob(os.path.join(os.path.join(root, 'projections'), '*.png'))  # only has 718 projections
    projection_matrices = np.genfromtxt(os.path.join('/data/datasets/spike-cbct/', 'proj_mat_720frames.csv'), delimiter=',')  # [719, 12]
    for i in range(len(projection_matrices)-1): 
    # for i in range (360):
        # im_gt = 1 - imageio.imread(filenames[i]).astype(np.float32) / 255.0
        # im_gt = im_gt - np.min(im_gt) # Normalize so empty space is always zero
        index = "{:04d}".format(i)
        # im_gt = imageio.imread(os.path.join('./jerry_gt', f'{index}_0001.png')).astype(np.float32) / 255.0
        im_gt = imageio.imread(os.path.join('/data/datasets/spike-cbct/spike720', f'Spike92_8_16_33_proj{index}.png')).astype(np.float32) / 255.0
        # im_gt = imageio.imread(os.path.join('/data/sfk/plenoxel-ct/jerry_gt', f'{index}_0001.png')).astype(np.float32) / 255.0 #Jerry gt
        # im_gt = imageio.imread(os.path.join('/data/datasets/Corrected_Projections', f'Cor_Proj{index}.png')).astype(np.float32) / 255.0 # Corrected projections


        # im_gt = imageio.imread(filenames[i]).astype(np.float32) / 255.0
        # im_gt = -np.log(np.where(im_gt > 1/255., im_gt, 1/255.))
        # im_gt = jnp.concatenate([im_gt[...,np.newaxis], jnp.zeros((im_gt.shape[0], im_gt.shape[1], 2))], -1)
        im_gt = 1 - im_gt
        w2c = np.reshape(projection_matrices[i], (3,4))
        # w2c[:,-1] = (w2c[:,-1] - [400, 220, 200])
        # w2c[:,-1] = (w2c[:,-1] - [510, 800, -60])
        # w2c[:,-1] = (w2c[:,-1] - [465, 750, 200])
        # w2c[:,-1] = (w2c[:,-1] - [375, 745, 150])
        # w2c[:,-1] = (w2c[:,-1] - [300, 300, 300])
        # 654 664 -> 650 670 -> 550 -770

        # w2c[:,-1] = (w2c[:,-1] - [520, 800, 200])
        # w2c[:,-1] = (w2c[:,-1] - [560, 840, 200])
        # w2c[:,-1] = (w2c[:,-1] - [400, 840, 0]) #moved to the right instead
        # w2c[:,-1] = (w2c[:,-1] - [400, 840, 100]) #moves to the left
        # w2c[:,-1] = (w2c[:,-1] - [500, 900, 100]) #centered first pic
        # w2c[:,-1] = (w2c[:,-1] - [300, 700, 30]) # moved the image down
        # w2c[:,-1] = (w2c[:,-1] - [400, 700, 40]) # looks fine until image 240 (moves to right)
        # w2c[:,-1] = (w2c[:,-1] - [400, 700, 45]) #tried with  focal = 2.85 / 0.024  in tmux2
        w2c[:,-1] = (w2c[:,-1] - [300, 700, 45]) # tmux1 w/ 50 and now tmux2 w/ 25
        # w2c[:,-1] = (w2c[:,-1] - [500, 200, 100])# tmux 3
        # w2c[:,-1] = (w2c[:,-1] - [0, 600, 600])



        # invert world -> camera to get camera -> world
        c2w = np.linalg.inv(np.concatenate([w2c, [[0,0,0,1]]], axis=0))

        all_c2w.append(c2w)

        # all_gt.append(im_gt[..., 0])
        all_gt.append(im_gt)
    # focal = 1  # cm
    focal = 2
    # focal = 10
    # focal = 15 # looks good for a z of 75
    # focal = 25
    # focal = 50 #tmux one w/ z of 50
    # focal = 2.85 / 0.024 #tmux two w/ z of 45
    # focal = 200
    # focal = 500

    # iin tmux1 = radius 10
    # in tmux 2 focal 2.85/0.024
    # in tmux 3 focal = 500
    
    # focal = 1000
    all_gt = np.asarray(all_gt)
    # all_gt = all_gt[:,:,0]
    all_c2w = np.asarray(all_c2w)
    print(f'all_c2w has size {all_c2w.size}')
    print(f'all_c2w has shape {all_c2w.shape}')
    # Remove the first 15 projections because the bulb is warming up so exposure is varying

    # all_gt = np.asarray(all_gt)
    # if len(all_gt.shape) < 4:
    #     all_gt = np.concatenate((all_gt[..., None], all_gt[..., None], all_gt[..., None]), axis=-1)  # Add a fake channel dimension for CT
    # all_c2w = np.asarray(all_c2w)
    print(f'all_gt has shape {all_gt.shape}')
    # Remove the first 15 projections because the bulb is warming up so exposure is varying
    # all_gt = all_gt[15:]
    # all_c2w = all_c2w[15:]

    mask = np.zeros(len(all_c2w))
    
    idx = np.random.choice(len(all_c2w), 100, replace = False) #tmux one
    # idx = np.random.choice(len(all_c2w), 25, replace = False) #tmux two

    mask[idx] = 1

    mask = mask.astype(bool)

    print(f'all_gt has shape {all_gt.shape}')
# train and test can be commented out ot get the full 360 ground truth projections
    if stage == 'train':
        all_gt = all_gt[mask]
        all_c2w = all_c2w[mask]
    elif stage == 'test':
        all_gt = all_gt[~mask]
        all_c2w = all_c2w[~mask]

    print(f'all_gt has shape {all_gt.shape}')
    return focal, all_c2w, all_gt

def get_ct_spike2(root, stage, max_projections, xoff, yoff, zoff):
    all_w2c = []
    all_gt = []

    print('LOAD DATA', root)
    
    # Would be good to test this for Jerry and Spike
    # projection_matrices = np.genfromtxt(os.path.join('/data/datasets/jerry-cbct/', 'proj_mat.csv'), delimiter=',')  # [719, 12]
    projection_matrices = np.genfromtxt(os.path.join('/home/fabriz/data/spike/', 'proj_mat_720frames.csv'), delimiter=',')  # [719, 12] /home/fabriz/data/spike/proj_mat_720frames.csv
    # print(f'proj mat len is {len(projection_matrices)}')

    #Traslation matrix along x,y,z
    Tz = np.matlib.zeros((4,4))
    Tz[0,0]=1.0
    Tz[1,1]=1.0
    Tz[2,2]=1.0
    Tz[3,3]=1.0
    Tz[0,3]=-xoff #test
    Tz[1,3]=-yoff #test
    Tz[2,3]=-zoff #test

    tif_proj = tifffile.imread('/home/fabriz/data/spike/Spike_702_proj.tif')

    # reads #max_projections projection images
    for i in range(len(projection_matrices)): 
        index = "{:04d}".format(i)
        # im_gt = imageio.imread(os.path.join('/data/datasets/newJerryProj', f'NewJerryProj_{index}.png')).astype(np.float32) / 255.0
        # im_gt = imageio.imread(os.path.join('/data/datasets/New_Corrected_Projections', f'New_Cor_Proj{index}.png')).astype(np.float32) / 255.0
        # im_gt = imageio.imread(os.path.join('/data/datasets/spike-cbct/spike720', f'Spike92_8_16_33_proj{index}.png')).astype(np.float32) / 255.0

        
        im_gt = tif_proj[i,:,:]

        im_gt = 1 - im_gt

        # projection matrices P_(3,4)
        w2c = np.reshape(projection_matrices[i], (3,4))
        w2c = np.matmul(w2c,Tz) #applico una traslazione per centrate il volume
        #w2c[:,-1] = (w2c[:,-1] - [400, 220, 200])
        
        all_w2c.append(w2c)
        all_gt.append(im_gt)

    all_gt = np.asarray(all_gt)
    all_w2c = np.asarray(all_w2c)

    focal = 100 
    
    mask = np.zeros(len(all_w2c))
    # print(f'max is {len(all_w2c)}')
    idx = np.random.choice(len(all_w2c), max_projections, replace = False) # Just look at a subset of projections, to save time/memory for debuggin
    mask[idx] = 1
    mask = mask.astype(bool)

    # train and test can be commented out to get the full 360 ground truth projections
    if stage == 'train':
        all_gt = all_gt[mask]
        all_w2c = all_w2c[mask]
    elif stage == 'test':
        all_gt  = all_gt[~mask]
        all_w2c = all_w2c[~mask]

    return focal, all_w2c,all_gt
    # assert FLAGS.ct

# This function takesn in the given root and uses the appropriate
#   data loader to get the focal, c2w, and gt
def get_data(root, stage):
    max_projections = 720
    # to align the volume and the detector it is possibile to use a traslation matrix T to
    # premultiply the projection matrices P'=P*T
    
    # For Jerry
    # xoff =  0.0
    # yoff =  0.0
    # zoff =  -2.4

    # For Spike
    xoff =  0.0
    yoff =  -1.3 # side to side (lower goes to left)
    zoff =  -5.2 # up down (the higher the number the higher the cube goes)


    if root == '/data/fabriziov/ct_scans/pepper/scans/':
        focal, all_c2w, all_gt = get_ct_pepper(root, stage)  
        # idx = np.random.choice(len(all_c2w), FLAGS.num_views) # Pick a subset of the data at random
        # return focal, all_c2w[idx], all_gt[idx]
        return focal, all_c2w, all_gt

    elif root == '/data/fabriziov/ct_scans/pomegranate/scans/':
        focal, all_c2w, all_gt = get_ct_pomegranate(root, stage)
        return focal, all_c2w, all_gt

    elif root == '/data/fabriziov/ct_scans/organSegs/scans/':
        focal, all_c2w, all_gt = get_ct_organSegs(root, stage)
        return focal, all_c2w, all_gt

    # elif root == '/data/datasets/Corrected_Projections/':
    #     focal, all_c2w, all_gt = get_ct_jerry(root, stage)  
    #     return focal, all_c2w, all_gt

    elif root == '/data/datasets/Corrected_Projections/':
        focal, all_c2w, all_gt = get_ct_jerry2(root, stage, max_projections, xoff, yoff, zoff)  
        # idx = np.random.choice(len(all_c2w), FLAGS.num_views) # Pick a subset of the data at random
        return focal, all_c2w, all_gt
    
    elif root == '/data/datasets/spike-cbct/':
        focal, all_c2w, all_gt = get_ct_spike2(root, stage, max_projections, xoff, yoff, zoff)  
        # idx = np.random.choice(len(all_c2w), FLAGS.num_views) # Pick a subset of the data at random
        return focal, all_c2w, all_gt

    # elif root == '/data/datasets/spike-cbct/':
    #     focal, all_c2w, all_gt = get_ct_spike(root, stage)  
    #     # idx = np.random.choice(len(all_c2w), FLAGS.num_views) # Pick a subset of the data at random
    #     return focal, all_c2w, all_gt

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
        # Adding next line from the plenotimize_static file, said to be for the linear version
        # This line makes more static-y images that seen to have depth
        # im_gt = jnp.concatenate([-jnp.log(1 - 0.99*im_gt[..., 3:]), jnp.zeros((im_gt.shape[0], im_gt.shape[1], 2))], -1)

        # This line makes more white images that look cleaner(sara said to ise this one)
        im_gt = jnp.concatenate([im_gt[..., 3:], jnp.zeros((im_gt.shape[0], im_gt.shape[1], 2))], -1) # If we want to train with alpha
        # Might want to comment out next line; plenoptimize_static file says to only use it when you want to train with color
        # im_gt = im_gt[..., :3] * im_gt[..., 3:] + (1.0 - im_gt[..., 3:])
        all_c2w.append(c2w)
        all_gt.append(im_gt)
    focal = 0.5 * all_gt[0].shape[1] / np.tan(0.5 * j['camera_angle_x'])
    all_gt = np.asarray(all_gt)
    all_c2w = np.asarray(all_c2w)
    return focal, all_c2w, all_gt

# This uses calls get_data to get a training set and a test set
#   for a focal, c2w, and gt.
# If the focal is not equal to the test_focal then an AssertionError is raised. 
# The height and width are set to the shape of index 0 of the training gt from
#   the begining to index 2
# Finally the length of the trianing and test c2w's are obtained
if __name__ == "__main__":
    print(f'the data directory is {data_dir}')
    focal, train_c2w, train_gt = get_data(data_dir, "train")
    test_focal, test_c2w, test_gt = get_data(data_dir, "test")
    assert focal == test_focal
    H, W = train_gt[0].shape[:2]
    dW = 0.024
    dH = 0.024
    n_train_imgs = len(train_c2w)
    n_test_imgs = len(test_c2w)

# Sets the new log_dirs to be the exsiting log_dir plus the experiment name
#   makes the neccessary directories for it if they don't exist.
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
    # reload_dir = "./jax_logs/test_spike_tiff_res256_epoch2_split0_lr0.001/epoch_0"
    print(f'Reloading the grid from {reload_dir}')
    data_dict = plenoxel_og_copy2.load_grid(dirname=reload_dir, sh_dim = (FLAGS.harmonic_degree + 1)**2)
    # import pdb; pdb.set_trace()
else:
    print(f'Initializing the grid')
    data_dict = plenoxel_og_copy2.initialize_grid(resolution=FLAGS.resolution, ini_rgb=FLAGS.ini_rgb, ini_sigma=FLAGS.ini_sigma, harmonic_degree=FLAGS.harmonic_degree)

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
# Takes a high resolution picture, blurs it, gets the low sine waves and creates a lower resolution picture from it
def multi_lowpass(gt, resolution):
    if gt.ndim <= 3:
        print(f'multi_lowpass called on image with 3 or fewer dimensions; did you mean to use lowpass instead?')
    H = gt.shape[-3]
    W = gt.shape[-2]
    clean_gt = np.copy(gt)
    for i in range(len(gt)):
        # print(np.squeeze(gt[i,...] * 255).shape)
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

def compute_tv(t):
    # xdim, ydim, zdim = t.shape

    x_tv = jnp.abs(t[1:, :, :] - t[:-1, :, :]).mean()
    y_tv = jnp.abs(t[:, 1:, :] - t[:, :-1, :]).mean()
    z_tv = jnp.abs(t[:, :, 1:] - t[:, :, :-1]).mean()
    return x_tv + y_tv + z_tv

def get_loss(data_dict, c2w, gt, H, W, focal, resolution, radius, harmonic_degree, jitter, uniform, key, sh_dim, occupancy_penalty, interpolation, nv):
    rays = plenoxel_og_copy2.get_rays(H, W, focal, c2w)
    rgb, disp, acc, weights, voxel_ids = plenoxel_og_copy2.render_rays(data_dict, rays, resolution, key, radius, harmonic_degree, jitter, uniform, interpolation, nv)
    mse = jnp.mean((rgb - lowpass(gt, resolution))**2)
    indices, data = data_dict
    loss = mse + occupancy_penalty * jnp.mean(jax.nn.relu(data_dict[-1]))

    return loss

# The plenotimize_static file has this added line to the top of this function, not sure if is necessary
@partial(jax.jit, static_argnums=(3,4,5,6,7,9,11,12))
def get_loss_rays(data_dict, rays, gt, resolution, radius, harmonic_degree, jitter, uniform, key, sh_dim, occupancy_penalty, interpolation, nv):
    
    # data_dict = [jnp.array(d) for d in data_dict]  # Convert data_dict list to JAX array

    rgb, disp, acc, weights, voxel_ids = plenoxel_og_copy2.render_rays(data_dict, rays, resolution, key, radius, harmonic_degree, jitter, uniform, interpolation, nv)
    # The plenoptimize_static file has this next line a little bit different, for the "alpha" channel only ???
    # mse = jnp.mean((rgb - gt)**2)
    mse = jnp.mean((acc - gt[...,0])**2)# Optimize the alpha channel only
    # indices, data = data_dict

    # print(f"data type{type(data[-1])}")
    # print(f"data_dict type{type(data_dict[-1])}")

    loss = mse + occupancy_penalty * jnp.mean(jax.nn.relu(data_dict[-1]))

    # return loss
    tv = compute_tv(data_dict[-1])
    return loss + 0.001 * tv
    # @ 0.1,   PSNR = 29.9224 w/ uncorrected images
    # @ 0.01,  PSNR = 30.3231 w/ uncorrected images
    # @ 0.001, PSNR = 29.6245 w/ uncorrected images

    # @ 0.1,    PSNR = 28.6819 w/ corrected images 0.001
    # @ 0.05,   PSNR = 27.9457 w/ corrected images
    # @ 0.01,   PSNR = 27.9006 w/ corrected images and lr = 0.001 w/ new PSNR 27.2833
        # this last one with a radius of 7 has a PSNR of 29.72
    # @ 0.001,  PSNR = 27.8984 w/ corrected images ***
        # this last one with a radius of 7 has a PSNR of 27.5601
        # this last one with a radius of 8 has a PSNR of 28.8879
        # this last one with a radius of 9 has a PSNR of 30.2918
        # this last one with a radius of 9 and 200 views and has a PSNR of 32.5680 (in tmux2 for newjerry2)
    # @ 0.0001, PSNR = 26.0925 w/ corrected images and lr = 0.001 w/ new PSNR 25.5065 *** try w/ 200 views and/or increase radius
        # this last one with 200 views got a PSNR of 26.4327 in newjerry3  
        # this last one with a radius of 7 has a PSNR of 26.4108
        # this last one with a radius of 7 and 200 views and has a PSNR of 28.0484 (in tmux2 for newjerry2)
        # this last one with a radius of 8 has a PSNR of 27.4417
        # this last one with a radius of 9 has a PSNR of 28.8734
        # this last one with a radius of 9 and 200 views and has a PSNR of 31.2722 


    # @ 0.1,    PSNR = 29.3367 w/ corrected images and lr = 0.01 
    # @ 0.01,   PSNR = 29.4156 w/ corrected images and lr = 0.01 w/ new PSNR 29.6147
    # @ 0.001,  PSNR = 29.0886 w/ corrected images and lr = 0.01 
    # @ 0.0001, PSNR = 28.6264 w/ corrected images and lr = 0.01

    # @ 0.1,    PSNR = 29.6574 w/ corrected images and lr = 0.1 
    # @ 0.01,   PSNR = 29.4928 w/ corrected images and lr = 0.1 
    # @ 0.001,  PSNR = 28.8807 w/ corrected images and lr = 0.1 

    # @ 0.1,    PSNR = 15.0389 w/ corrected images and lr = 0.0001 
    # @ 0.01,   PSNR = 14.0932 w/ corrected images and lr = 0.0001 
    # @ 0.001,  PSNR = 13.8055 w/ corrected images and lr = 0.0001 

    # @ 0.5,    PSNR = 27.6756 w/ corrected images and lr = 0.001

    return loss


def get_rays_np(H, W, dH, dW, w2c):
    # get M matrix
    M = w2c[:,0:3]
    # get  p4
    p4 = w2c[:,-1]

    # compute uo,vo,sdd
    uo = (M[0,:]*M[2,:]).sum()
    vo = (M[1,:]*M[2,:]).sum()
    aU = math.sqrt((M[0,:]*M[0,:]).sum() - uo*uo)
    aV = math.sqrt((M[1,:]*M[1,:]).sum() - vo*vo)
    sdd = 0.5*(aU+aV)

    #source position in the World Reference system
    M_inv = np.linalg.inv(M)
    srcPos = -np.matmul(M_inv,p4)
    
    shiftVo = (vo - 0.5 * H * dH)
    u, v = jnp.meshgrid(jnp.linspace(0, W-1, W) + 0.5, jnp.linspace(0, H-1, H) + 0.5)
    u = u * dW # u
    v = v * dH # v
    dirs   = jnp.stack([u, v, jnp.ones_like(u)], -1) 
    rays_d = jnp.sum(dirs[..., jnp.newaxis, :] * M_inv, -1) # check if the syntax is ok
    rays_o = jnp.broadcast_to(srcPos,rays_d.shape)          # check if the syntax is ok
    return rays_o, rays_d



def render_pose_rays(data_dict, c2w, H, W, focal, resolution, radius, harmonic_degree, jitter, uniform, key, sh_dim, batch_size, interpolation, nv):
    rays_o, rays_d = get_rays_np(H, W, dH, dW, c2w)
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
            rgbi, dispi, acci, weightsi, voxel_idsi = jax.lax.stop_gradient(plenoxel_og_copy2.render_rays(data_dict, (rays_o[start:stop], rays_d[start:stop]), resolution, key[start:stop], radius, harmonic_degree, jitter, uniform, interpolation, nv))
        else:
            rgbi, dispi, acci, weightsi, voxel_idsi = jax.lax.stop_gradient(plenoxel_og_copy2.render_rays(data_dict, (rays_o[start:stop], rays_d[start:stop]), resolution, None, radius, harmonic_degree, jitter, uniform, interpolation, nv))
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
    pb = tqdm(total=len(test_c2w))
    for j, (c2w, gt) in tqdm(enumerate(zip(test_c2w, test_gt))):

        # skips tha images that arent shown // Added on to debug faster, should be taken away later
        # if FLAGS.render_interval > 0 and j % FLAGS.render_interval != 0:
        #     continue
        rgb, disp, _, _ = render_pose_rays(data_dict, c2w, H, W, focal, FLAGS.resolution, radius, FLAGS.harmonic_degree, FLAGS.jitter, FLAGS.uniform, key, sh_dim, FLAGS.physical_batch_size, FLAGS.interpolation, FLAGS.nv)
        # assert False
        # print(c2w.shape)
        # print(rgb.shape)
        # print(gt.shape)
        #Use these prints if you need to show shapes and ranges/////////////////////////////////////// 
        
        # Adding these next two lines from the plenoptimize__static file because it was under an if statment for ct files
        rgb = jnp.concatenate((rgb[...,0,jnp.newaxis], rgb[...,0,jnp.newaxis], rgb[...,0,jnp.newaxis]), axis=-1)
        gt = jnp.concatenate((gt[...,jnp.newaxis], gt[...,jnp.newaxis], gt[...,jnp.newaxis]), axis=-1)
        

        # from skimage.restoration import denoise_tv_chambolle
        # denoise_rgb = denoise_tv_chambolle((rgb*255).astype(np.uint8), weight=0.1, eps=0.0002, max_num_iter=200, channel_axis=None)

        # From pleoptimize_static-----------
        # gt = jnp.concatenate((gt[...,0,jnp.newaxis], gt[...,0,jnp.newaxis], gt[...,0,jnp.newaxis]), axis=-1)
        # ---------------------------------
        # print(rgb.shape)
        # print(gt.shape)

        # print(f'\nrgb ranges from {np.min(rgb)} to {np.max(rgb)}')
        # print(f'gt ranges from {np.min(gt)} to {np.max(gt)}')

        mse = jnp.mean((rgb - gt)**2)
        # print("bits is ", mse)
        psnr = -10.0 * np.log(mse) / np.log(10.0)
        tpsnr += psnr

        if FLAGS.render_interval > 0 and j % FLAGS.render_interval == 0:
            # The plenoptimize_static file doesnt have this next line but another line instead of it and changed a bit of the line after
            # disp3 = jnp.concatenate((disp[...,jnp.newaxis], disp[...,jnp.newaxis], disp[...,jnp.newaxis]), axis=2)
            # print(f'disp3 has shape {disp3.shape}')

            # print(f'disp3 ranges from {np.min(disp3)} to {np.max(disp3)}')

            # vis = jnp.concatenate((gt, rgb, disp3), axis=1)
            # vis = np.asarray((vis * 255)).astype(np.uint8)
            # imageio.imwrite(f"{log_dir}/{j:04}_{i:04}{name_appendage}.png", vis)
            # print(f'gt ranges from {jnp.min(gt)} to {jnp.max(gt)} and prediction ranges from {jnp.min(rgb)} to {jnp.max(rgb)} and disp ranges from {jnp.min(disp)} to {jnp.max(disp)}')
            
            vis = jnp.concatenate((gt, rgb), axis = 1)

# ***************************************************************************************************************************************************************************************************************
            # vis = rbg for new file
            # vis = rgb

            # NOT NOT BE NEEDED (ADDED 12-20-22)
            # vis = np.asarray((vis*255)).astype(np.uint8)

            # print(f'vis ranges from {np.min(vis)} to {np.max(vis)}')

            imageio.imwrite(f"{log_dir}/{j:04}_{i:04}{name_appendage}.png", (vis*255).astype(np.uint8))

            tp = tpsnr
            tp /= len(test_c2w)
            pb.set_postfix_str(f"psnr = {tp}", refresh = False)
            pb.update(1)
        del rgb, disp
    # tpsnr /= n_test_imgs
    tpsnr /= len(test_c2w)
    return tpsnr


def update_grid(old_grid, lr, grid_grad):
    # if FLAGS.nonnegative:
    #     return jnp.clip(index_add(old_grid, index[...], -1 * lr * grid_grad), a_min=0)
    # else:
    #     return index_add(old_grid, index[...], -1 * lr * grid_grad)
    
    if FLAGS.nonnegative:
        return jnp.clip(old_grid.at[...].add( -1 * lr * grid_grad), a_min=0)
    else:
        return old_grid.at[...].add(-1 * lr * grid_grad)



def update_grids(old_grid, lrs, grid_grad):
    # Added these two lines because the plenoptimize_static code has it under an if statement for ct
    old_grid[-1] = update_grid(old_grid[-1], lrs[-1], grid_grad[-1])# Only updates the sigma grid for CT
    return old_grid
    # for i in range(len(old_grid)):
    #     old_grid[i] = index_add(old_grid[i], index[...], -1 * lrs[i] * grid_grad[i])
    # return old_grid

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
        # print("hi")
        
        # rays = jax.vmap(fun=lambda p: plenoxel_og_copy2.get_rays(H, W, focal, p), in_axes=0, out_axes=0)(train_c2w[:,:3,:4])  # OOM
        rays = np.stack([get_rays_np(H, W, dH, dW, p) for p in train_c2w[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]  # 42 seconds
        
        t1 = time.time()
        print(f'stack took {t1 - t0} seconds')
        # print(f'train_c2w has shape {train_c2w.shape}')
        # print(f'train_gt has shape {train_gt.shape}')

    # Took this next line away bc it wasnt in plenoptimize_static but might need it for this------
        train_gt = np.concatenate([train_gt[...,None], train_gt[...,None], train_gt[...,None]], -1)
    # -------------------------------------------------------------------
    
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
    if np.isin(FLAGS.reload_epoch, FLAGS.prune_epochs):
        data_dict = plenoxel_og_copy2.prune_grid(data_dict, method=FLAGS.prune_method, threshold=FLAGS.prune_threshold, train_c2w=train_c2w, H=H, W=W, focal=focal, batch_size=FLAGS.physical_batch_size, resolution=FLAGS.resolution, key=render_keys, radius=FLAGS.radius, harmonic_degree=FLAGS.harmonic_degree, jitter=FLAGS.jitter, uniform=FLAGS.uniform, interpolation=FLAGS.interpolation)
    if np.isin(FLAGS.reload_epoch, FLAGS.split_epochs):
        print(FLAGS.resolution)
        data_dict = plenoxel_og_copy2.split_grid(data_dict)
        FLAGS.resolution = FLAGS.resolution * 2
        if automatic_lr:
            FLAGS.lr_rgb = 150 * (FLAGS.resolution ** 1.75)
            FLAGS.lr_sigma = 51.5 * (FLAGS.resolution ** 2.37)
# -----------------------------------------------------------------------------------------------
    avg_g = [0 for g_i in data_dict]
    
    # hi = False
    
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

        # print('epoch', i)
        pb = tqdm(total=len(test_c2w), desc = f"Epoch {i}")

        # plenoptimize_static does not have this next line
        # indices, data = data_dict
        if FLAGS.physical_batch_size is None:
            occupancy_penalty = FLAGS.occupancy_penalty / len(train_c2w)
            # pb = tqdm(total=len(train_c2w))
            for j, (c2w, gt) in tqdm(enumerate(zip(train_c2w, train_gt)), total=len(train_c2w)):
                if FLAGS.jitter > 0:
                    splitkeys = split_keys(keys)
                    keys = splitkeys[...,0,:]
                    subkeys = splitkeys[...,1,:]
                else:
                    subkeys = None
                # This next line is slightly differnet in the plenoptimize_static code
                mse, data_grad = jax.value_and_grad(lambda grid: get_loss(grid, c2w, gt, H, W, focal, FLAGS.resolution, radius, FLAGS.harmonic_degree, FLAGS.jitter, FLAGS.uniform, subkeys, sh_dim, occupancy_penalty, FLAGS.interpolation, FLAGS.nv))(data_dict) 
                # mse, data_grad = jax.value_and_grad(lambda grid: get_loss(grid, c2w, gt, H, W, focal, FLAGS.resolution, radius, FLAGS.harmonic_degree, FLAGS.jitter, FLAGS.uniform, subkeys, sh_dim, occupancy_penalty, FLAGS.interpolation, FLAGS.nv))(data_dict) 
                # pb.set_postfix_str(f"psnr = {}")
        else:
            occupancy_penalty = FLAGS.occupancy_penalty / (len(rays_rgb) // FLAGS.logical_batch_size)
            for k in tqdm(range(len(rays_rgb) // FLAGS.logical_batch_size)):
            # for k in tqdm(range(10000)):
            # for k in tqdm(range(100)):
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
                    mse, data_grad = jax.value_and_grad(lambda grid: get_loss_rays(grid, batch_rays, target_s, FLAGS.resolution, radius, FLAGS.harmonic_degree, FLAGS.jitter, FLAGS.uniform, subkeys, sh_dim, occupancy_penalty, FLAGS.interpolation, FLAGS.nv))(data_dict) 
                    # mse, data_grad = get_loss_rays((indices, data), batch_rays, target_s, FLAGS.resolution, radius, FLAGS.harmonic_degree, FLAGS.jitter, FLAGS.uniform, subkeys, sh_dim, occupancy_penalty, FLAGS.interpolation, FLAGS.nv)
                    # mse, data_grad = jax.value_and_grad(lambda grid: get_loss_rays(grid, batch_rays, target_s, FLAGS.resolution, radius, FLAGS.harmonic_degree, FLAGS.jitter, FLAGS.uniform, subkeys, sh_dim, occupancy_penalty, FLAGS.interpolation, FLAGS.nv))(data_dict) 
                    
# An error would pop up for this next line in jnp.linalg.norm(data_grad) so i added the [0] to make it work, not sure if that is correct or not
                    # import pdb; pdb.set_trace()
                    pb.set_postfix_str(f"psnr = {-10*jnp.log10(mse)}, grad norm = {jnp.linalg.norm(data_grad[0])}", refresh = False)
                    pb.update(1)

                    # import pdb; pdb.set_trace()

                    if FLAGS.logical_batch_size > FLAGS.physical_batch_size:
                        if logical_grad is None:
                            logical_grad = data_grad
                        else:
                            logical_grad = [a + b for a, b in zip(logical_grad, data_grad)]
                        del data_grad
                    del mse, batch, batch_rays, target_s, subkeys, effective_j
                lrs = [FLAGS.lr_rgb / (FLAGS.logical_batch_size // FLAGS.physical_batch_size)]*sh_dim + [FLAGS.lr_sigma / (FLAGS.logical_batch_size // FLAGS.physical_batch_size)]

                lrs  = [lr * np.cos(k /((len(rays_rgb) // FLAGS.logical_batch_size)+ 10) * (np.pi/2)) for lr in lrs]
                
                # pb.set_postfix_str(f"logical norm = {jnp.linalg.norm(logical_grad)}, data_grad norm = {jnp.linalg.norm(data_grad)}, logical batch size = {FLAGS.logical_batch_size}, physical batch size = {FLAGS.physical_batch_size}", refresh = False)
                # pb.update(1)


                # avg_g = [0.9 * (avg_g_i) + 0.1 * (g_i**2) for (avg_g_i, g_i) in zip(avg_g, data_grad)]
                avg_g = rmsprop_update(avg_g, data_grad)

                if FLAGS.logical_batch_size > FLAGS.physical_batch_size:
                    data_dict = update_grids(data_dict, lrs, logical_grad)
                    del logical_grad
                else:
                    data_dict[-1] = update_grid(data_dict[-1], lrs[-1], data_grad[-1]/ (jnp.sqrt(avg_g[-1]) + 1e-10))  
                    # data = update_grid(data[-1], lrs, data_grad) 
                    del data_grad, logical_grad



        # -------------------------------This part is needed for split epochs -----------------------------------
        # data_dict = data
        # del indices, data
        if np.isin(i, FLAGS.prune_epochs):
            data_dict = plenoxel_og_copy2.prune_grid(data_dict, method=FLAGS.prune_method, threshold=FLAGS.prune_threshold, train_c2w=train_c2w, H=H, W=W, focal=focal, batch_size=FLAGS.physical_batch_size, resolution=FLAGS.resolution, key=render_keys, radius=FLAGS.radius, harmonic_degree=FLAGS.harmonic_degree, jitter=FLAGS.jitter, uniform=FLAGS.uniform, interpolation=FLAGS.interpolation)
        if np.isin(i, FLAGS.split_epochs):
            print(f'at epoch {i}, about to split. res = {data_dict[0].shape}, flags.res = {FLAGS.resolution} ')
            data_dict = plenoxel_og_copy2.split_grid(data_dict)
            FLAGS.lr_rgb = FLAGS.lr_rgb * 3
            FLAGS.lr_sigma = FLAGS.lr_sigma * 3
            FLAGS.resolution = FLAGS.resolution * 2
            print(f'at epoch {i}, finished split. res = {data_dict[0].shape}, flags.res = {FLAGS.resolution} ')
            # if automatic_lr:
            #     FLAGS.lr_rgb = 150 * (FLAGS.resolution ** 1.75)
            #     FLAGS.lr_sigma = 51.5 * (FLAGS.resolution ** 2.37)
            # if FLAGS.physical_batch_size is not None:
            if True:
                # Recompute all the training rays at the new resolution and shuffle them
                rays = np.stack([get_rays_np(H, W, dH, dW, p) for p in train_c2w[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
                rays_rgb = np.concatenate([rays, multi_lowpass(train_gt[:,None], FLAGS.resolution).astype(np.float32)], 1) # [N, ro+rd+rgb, H, W,   3]
                rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
                rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
                rays_rgb = rays_rgb.take(np.random.permutation(rays_rgb.shape[0]), axis=0)
            avg_g = plenoxel_og_copy2.split_grid(avg_g)
            # import pdb; pdb.set_trace()
        #-------------------------------------------------------------------------------------------------------


        pb.close()

        # if FLAGS.cut_cube and i == 2:
        #     print(f'cut cube flag is working')
        #     data_dict = plenoxel_og_copy2.crop_inner_cube2(data_dict)
        #     # radius = radius/2
        #     radius = 6
        #     FLAGS.resolution = 600
        #     if True:
        #         # Recompute all the training rays at the new resolution and shuffle them
        #         rays = np.stack([get_rays_np(H, W, dH, dW, p) for p in train_c2w[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        #         rays_rgb = np.concatenate([rays, multi_lowpass(train_gt[:,None], FLAGS.resolution).astype(np.float32)], 1) # [N, ro+rd+rgb, H, W,   3]
        #         rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        #         rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        #         rays_rgb = rays_rgb.take(np.random.permutation(rays_rgb.shape[0]), axis=0)
        #     avg_g = plenoxel_og_copy2.crop_inner_cube2(avg_g)
        #     # print(type(avg_g))
        #     print(f'new cube is shape {data_dict[0].shape}, and res is = {FLAGS.resolution}')



        if i % FLAGS.save_interval == FLAGS.save_interval - 1 or i == FLAGS.num_epochs - 1:
            print(f'Saving checkpoint at epoch {i}')
            plenoxel_og_copy2.save_grid(data_dict, os.path.join(log_dir, f'epoch_{i}'))

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


        #This is where i  do the croppping of the cube
        # the crop_inner_cube function will get the inside    
        # if FLAGS.cut_cube and i == 1:
        # # if i == 1 and FLAGS.resolution == 512 and hi == False:
        #     print(f'cut cube flag is working')
        #     data_dict = plenoxel_og_copy2.crop_inner_cube(data_dict)
        #     # radius = radius/2
        #     radius = 6
        #     FLAGS.lr_rgb = FLAGS.lr_rgb / 3
        #     FLAGS.lr_sigma = FLAGS.lr_sigma / 3
        #     # FLAGS.resolution = FLAGS.resolution // 2
        #     FLAGS.resolution = 600
        #     # i = i - 1
        #     # print(f'new cube is shape {data_dict[0].shape}, avg_g shape is {avg_g[0].shape}, and res is = {FLAGS.resolution}')
        #     # print(type(avg_g))
        #     # print(f'i == {i}')
        #     if True:
        #         # Recompute all the training rays at the new resolution and shuffle them
        #         rays = np.stack([get_rays_np(H, W, dH, dW, p) for p in train_c2w[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        #         rays_rgb = np.concatenate([rays, multi_lowpass(train_gt[:,None], FLAGS.resolution).astype(np.float32)], 1) # [N, ro+rd+rgb, H, W,   3]
        #         rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        #         rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        #         rays_rgb = rays_rgb.take(np.random.permutation(rays_rgb.shape[0]), axis=0)
        #     avg_g = plenoxel_og_copy2.crop_inner_cube(avg_g)
        #     # print(type(avg_g))
        #     print(f'new cube is shape {data_dict[0].shape}, and res is = {FLAGS.resolution}')
        #     # print(f'i == {i}')

        #     # start_epoch = 0

        # if FLAGS.cut_cube and i == 2:
        #     print(f'cut cube flag is working')
        #     data_dict = plenoxel_og_copy2.crop_inner_cube2(data_dict)
        #     # radius = radius/2
        #     radius = 6
        #     FLAGS.resolution = 600
        #     if True:
        #         # Recompute all the training rays at the new resolution and shuffle them
        #         rays = np.stack([get_rays_np(H, W, dH, dW, p) for p in train_c2w[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        #         rays_rgb = np.concatenate([rays, multi_lowpass(train_gt[:,None], FLAGS.resolution).astype(np.float32)], 1) # [N, ro+rd+rgb, H, W,   3]
        #         rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        #         rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        #         rays_rgb = rays_rgb.take(np.random.permutation(rays_rgb.shape[0]), axis=0)
        #     avg_g = plenoxel_og_copy2.crop_inner_cube2(avg_g)
        #     # print(type(avg_g))
        #     print(f'new cube is shape {data_dict[0].shape}, and res is = {FLAGS.resolution}')
    
if __name__ == "__main__":
    main()