# Class-Conditional Latent Diffusion for Sparse Detector Jet Image Generation

Anonymous implementation accompanying a double-blind workshop submission.

🚧 Note: This repository is under active development and continuously updated.

## Overview

This repository contains the implementation of a class-conditional latent
diffusion model for the generation of sparse three-channel detector jet
images.

The complete pipeline consists of:

1. A spatial variational autoencoder (VAE) that compresses
   `64 × 64 × 3` detector images into a `4 × 16 × 16` latent representation.
2. Latent-channel standardization using statistics computed exclusively
   from the training set.
3. A class-conditional latent diffusion model trained on the standardized
   latent representation.
4. Timestep and jet-class conditioning for quark and gluon jets.
5. Decoding of generated latent samples using the trained VAE and
   validation-calibrated occupancy thresholds.
6. Evaluation using detector-level observables such as integrated response,
   active-pixel fraction, maximum response, radial response profiles, and
   Wasserstein-1 distances.

---

## Repository Structure

```text
conditional-latent-diffusion/
├── Diffusion/
│   ├── Diffusion_Final_DDPM.ipynb
│   └── readme.md
│
├── LDM/
│   ├── LDM_DIFFUSION_2.ipynb
│   └── readme.md
│
├── VAE/
│   ├── VAE_JETS_1-2.ipynb
│   └── readme.md
│
└── README.md
