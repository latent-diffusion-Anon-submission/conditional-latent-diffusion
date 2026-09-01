# Class-Conditional Latent Diffusion for Sparse Detector Jet Image Generation

Anonymous implementation accompanying a double-blind workshop submission.

## Overview

This repository contains the implementation of a class-conditional latent
diffusion model for the generation of sparse three-channel detector jet
images.

The pipeline (specifically LDM) consists of:

1. A spatial variational autoencoder (VAE) that compresses
   64×64×3 detector images into a 4×16×16 latent representation.
2. Channel-wise latent standardization using training-set statistics.
3. A class-conditional latent diffusion model trained to generate
   quark and gluon jet representations.
4. Decoding of generated latent samples using the trained VAE.

## Data

The experiments use a quark/gluon jet-image dataset with three detector
channels: Tracks, ECAL, and HCAL.

The original images have resolution 125×125×3 and are resized and
preprocessed as described in the accompanying paper.

## Requirements

Main dependencies include:

- Python
- PyTorch
- NumPy
- h5py
- matplotlib
- scipy

## Training

The repository contains the code used to train:

- the spatial VAE;
- the class-conditional latent diffusion model.

## Reproducibility

Training hyperparameters, preprocessing details, occupancy-threshold
calibration, and evaluation procedures are documented in the accompanying
anonymous manuscript.

## Anonymity

Author information has been intentionally omitted to preserve double-blind
review.
