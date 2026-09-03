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
│   ├── LDM_DIFFUSION_CONFERENCE_S2S.ipynb   # Final submission pipeline
│   ├── LDM_DIFFUSION_2.ipynb                # Earlier development version
│   └── readme.md
│
├── VAE/
│   ├── VAE_JETS_1-2.ipynb
│   └── readme.md
│
└── README.md
```

The `LDM/` directory contains the integrated latent-diffusion pipeline.

The `VAE/` directory contains the spatial VAE implementation and earlier
development experiments.

The `Diffusion/` directory contains the standalone diffusion-model
implementation developed before integration into the latent-diffusion
pipeline.

---

## Data

The experiments use a simulated CMS Open Data quark/gluon jet-image
dataset with three detector channels:

- Tracks
- ECAL
- HCAL

The dataset is based on:

> M. Andrews et al.,  
> *End-to-end jet classification of quarks and gluons with the CMS Open Data*,  
> Nuclear Instruments and Methods in Physics Research Section A,  
> Volume 977, 164304 (2020).  
> DOI: 10.1016/j.nima.2020.164304

The dataset used in this work is available at:

https://drive.google.com/file/d/1WO2K-SfU2dntGU4Bb3IYBp9Rh7rtTYEr/view

The complete sample contains:

- 139,306 jet events
- 69,653 gluon jets
- 69,653 quark jets

The original events are represented as:

```text
125 × 125 × 3
```

with channels corresponding to Tracks, ECAL, and HCAL.

The data provided for the present experiments are resized and stored as:

```text
64 × 64 × 3
```

images.

---

## Data Preprocessing

The provided HDF5 data had already undergone:

1. `log(1 + x)` transformation;
2. resizing from `125 × 125` to `64 × 64`;
3. per-channel normalization;
4. clipping to `[0, 1]`; and
5. rescaling to `[-1, 1]`.

The dataset is subsequently divided into:

```text
Training:   111,444 events
Validation:  13,930 events
Test:        13,932 events
```

corresponding to an `80/10/10` split.

For model training, the stored images are first mapped back to `[0, 1]`.

An additional global normalization scale is then computed exclusively from
the nonzero pixels in the training partition:

```text
s = 0.02824259
```

corresponding to the 99.5th percentile of nonzero training-set pixels.

The same scale is kept fixed when preprocessing the validation and test
partitions.

After clipping to `[0, 1]`, VAE inputs are mapped to `[-1, 1]`.

Further preprocessing details are provided in the accompanying manuscript.

---

## Requirements

The implementation uses Python and PyTorch.

Main dependencies include:

- Python 3
- PyTorch
- NumPy
- h5py
- matplotlib
- SciPy
- Jupyter Notebook

A minimal Python environment can be created with:

```bash
python -m venv .venv
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the main dependencies with:

```bash
pip install torch numpy h5py matplotlib scipy jupyter
```

---

## Hardware

The final experiments reported in the accompanying manuscript were
performed on:

```text
MacBook Pro
Apple M4 Pro
24 GB unified memory
```

Model training was accelerated using the PyTorch Metal Performance
Shaders (`MPS`) backend.

---

## Running the Experiments

Clone the repository:

```bash
git clone https://github.com/latent-diffusion-Anon-submission/conditional-latent-diffusion.git
cd conditional-latent-diffusion
```

Download the HDF5 dataset from the dataset link provided above.

Set the local HDF5 file path in the corresponding data-loading cell of
the notebook before execution.

Start Jupyter Notebook with:

```bash
jupyter notebook
```

```markdown
The integrated latent-diffusion workflow used for the results reported
in the accompanying manuscript is located in:

```text
LDM/LDM_DIFFUSION_CONFERENCE_S2S.ipynb

Run the notebook sequentially from top to bottom.

---

## Experimental Pipeline

### 1. Dataset Loading

The HDF5 jet-image dataset is loaded and divided into training,
validation, and held-out test partitions.

### 2. Preprocessing

The provided detector images are transformed using the training-derived
normalization scale described above.

All statistics used for the additional normalization are computed from
the training partition only.

### 3. Spatial VAE Training

The spatial VAE compresses each detector image as:

```text
3 × 64 × 64
      ↓
4 × 16 × 16
```

The encoder contains residual convolutional blocks and two spatial
downsampling stages.

The decoder separately predicts:

- pixel occupancy;
- conditional pixel intensity.

This separation is designed to better model the highly sparse detector
images.

During training, the decoder uses a differentiable occupancy gate.

For inference, the soft gate is replaced by fixed channel-dependent hard
occupancy thresholds calibrated exclusively using the validation set.

The final VAE is trained for:

```text
Epochs:        12
Batch size:    16
Optimizer:     AdamW
Learning rate: 2e-4
```

---

## Occupancy Threshold Calibration

The final hard-decoding thresholds are:

```text
Tracker: 0.805437
ECAL:    0.793923
HCAL:    0.854696
```

These thresholds are selected using the validation partition and are
kept fixed during test-set reconstruction and generation.

---

## Latent Representation

After VAE training, the deterministic posterior mean is used as the
latent representation.

Each event is represented as:

```text
4 × 16 × 16
```

Latent channels are standardized using mean and standard-deviation
statistics computed exclusively from the training set.

The same statistics are then applied unchanged to validation, test, and
generated latent samples.

---

## Class-Conditional Latent Diffusion

The diffusion model is trained in the standardized latent space.

The forward diffusion process uses:

```text
Diffusion steps: 1000
Beta start:      1e-4
Beta end:        0.02
Schedule:        linear
```

The denoising model is a U-Net with:

```text
Input channels:              4
Base width:                  64
Channel multipliers:         (1, 2, 4, 4)
Timestep embedding:          256
Self-attention resolutions:  16 × 16 and 8 × 8
```

The network is conditioned on:

- diffusion timestep;
- jet class.

The class labels are:

```text
0 = Gluon
1 = Quark
```

The model uses `v`-prediction rather than direct noise prediction.

The diffusion objective combines the standard velocity-prediction loss
with an auxiliary tail-weighted clean-latent reconstruction loss.

The final diffusion model is trained for:

```text
Epochs:        30
Batch size:    64
Optimizer:     Adam
Learning rate: 1e-4
```

The best checkpoint is selected according to validation loss.

---

## Generation

At generation time:

1. Gaussian noise is sampled in standardized latent space.
2. The latent is iteratively denoised using the class-conditioned
   diffusion model.
3. The final latent is transformed back using the training-set latent
   statistics.
4. The latent representation is decoded using the frozen spatial VAE.
5. Validation-calibrated hard occupancy thresholds are applied during
   decoding.

The final evaluation reported in the manuscript uses:

```text
2,500 generated events
```

with class labels taken from held-out test events.

---

## Evaluation

Generated and reference samples are compared globally, per detector
channel, and separately for gluon and quark jets.

The evaluation includes:

- integrated normalized detector response;
- active-pixel fraction;
- maximum pixel response;
- radial response profiles;
- Wasserstein-1 distances;
- channel-wise response bias; and
- class-conditional response comparisons.

The active-pixel threshold used for the detector-level diagnostics is:

```text
1e-4
```

The main observables are evaluated independently for:

```text
Tracker
ECAL
HCAL
```

---

## Reproducibility

To avoid information leakage between dataset partitions:

- additional detector-response normalization statistics are calculated
  using the training partition only;
- latent standardization statistics are calculated using the training
  partition only;
- occupancy thresholds are calibrated using the validation partition;
- model selection is based on validation loss; and
- the test partition remains held out for final evaluation.

The same preprocessing and evaluation definitions are used for reference
and generated samples.

Additional architecture details, objective functions, hyperparameters,
and evaluation definitions are documented in the accompanying anonymous
manuscript.

---

## Approximate Training Time

On the Apple M4 Pro system described above:

### Spatial VAE

```text
Approximately 18 minutes / epoch
12 epochs
Total: approximately 3.6 hours
```

### Latent Diffusion

```text
Approximately 150 seconds / epoch
30 epochs
Total: approximately 1.25 hours
```

The combined wall-clock training time for the two final models is
approximately:

```text
4.85 hours
```

These values are approximate and may vary depending on hardware,
software versions, and system load.

---

## Notes on Repository Contents

The repository contains both final integrated components and earlier
development implementations.

The `LDM/` directory contains the integrated VAE and latent-diffusion
workflow.

The standalone `VAE/` and `Diffusion/` directories document earlier
development stages of the individual components and are retained for
transparency.

---

## Anonymity

This repository accompanies a double-blind workshop submission.

Author names, institutional affiliations, personal contact information,
and other identifying information have intentionally been omitted to
preserve anonymity during peer review.
