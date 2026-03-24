# Gradient Descent Provably Solves Nonlinear Tomographic Reconstruction

Sara Fridovich-Keil, Fabrizio Valdivia, Gordon Wetzstein, Benjamin Recht, Mahdi Soltanolkotabi

arXiv: <https://arxiv.org/abs/2310.03956>

Accepted for publication in IEEE Transactions on Information Theory

This repo is built on a fork from the JAX version of Plenoxels: Radiance Fields without Neural Networks, with modification to support X-ray cone beam computed tomography (CBCT) and compare linear vs nonlinear reconstruction methods. 

## Setup

We recommend setup with a conda environment, using the packages provided in `requirements.txt`.

## Voxel Optimization (aka Training)

The training file is `plenoptimize.py`; its flags specify many options to control the optimization (scene, resolution, training duration, when to prune and subdivide voxels, where the training data is, where to save rendered images and model checkpoints, etc.). You can also set the frequency of evaluation, which will compute the validation PSNR and render validation images (comparing the reconstruction to the ground truth).
