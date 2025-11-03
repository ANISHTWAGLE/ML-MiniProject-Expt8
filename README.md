# CelebA IMLE GAN and Visualization Toolkit
# Dataset :- https://www.kaggle.com/datasets/jessicali9530/celeba-dataset

This project provides a complete pipeline for analyzing the CelebA dataset and training a Generative Adversarial Network (GAN) using an Implicit Maximum Likelihood Estimation (IMLE) loss.

The repository is divided into three main components:
1.  **Data Visualization (`CelebA_visualization_notebook.py`):** A comprehensive Jupyter Notebook script to perform an exploratory data analysis (EDA) of the CelebA dataset. It analyzes partitions, bounding boxes, facial landmarks, and attributes.
2.  **Data Preparation (`create_celeba_hdf5.py`):** A utility script to convert the `img_align_celeba` folder of individual JPEGs into a single, efficient HDF5 file. This dramatically speeds up data loading during training.
3.  **Model Training (`train_imle_gan.py`):** A series of three iterative approaches to training a StyleGAN-like generator with a standard GAN loss supplemented by an IMLE loss to improve mode coverage and training stability.

## Project Components

* `CelebA_visualization_notebook.py`: A script intended to be run cell-by-cell in a Jupyter or Kaggle notebook. It provides functions to plot dataset partitions, show random image grids, analyze bounding box distributions, visualize landmark heatmaps, and plot attribute correlation matrices.
* `create_celeba_hdf5.py`: Contains the `HDF5Exporter` class. This script iterates through the CelebA image directory, applies a center crop to 128x128, and saves the images as `uint8` arrays in a large `.hdf5` file.
* `train_imle_gan.py`: The main training script. It includes an `H5ImageDataset` for loading the HDF5 file, a simple StyleGAN-like generator, a CNN discriminator, and the training loop combining a standard GAN loss (`BCELoss`) with the IMLE loss (`MSELoss`).

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download CelebA Dataset:**
    * Download the CelebA dataset (e.g., from Kaggle or the official site).
    * You will need the following files and directories:
        * `img_align_celeba/` (the folder of >200k images)
        * `list_attr_celeba.csv`
        * `list_bbox_celeba.csv`
        * `list_landmarks_align_celeba.csv`
        * `list_eval_partition.csv`
    * Place them in a directory, for example, `./data/celeba/`.

## Workflow

### Step 1. (Optional) Analyze the Dataset

Run the `CelebA_visualization_notebook.py` script in a Jupyter environment. Make sure to update the paths in the final `run_all(...)` function call to point to your downloaded CelebA files.

```python
# Inside your notebook, at the end of the script:
run_all(
    images_dir="/path/to/img_align_celeba",
    attr_csv="/path/to/list_attr_celeba.csv",
    bbox_csv="/path/to/list_bbox_celeba.csv",
    landmark_csv="/path/to/list_landmarks_align_celeba.csv",
    eval_csv="/path/to/list_eval_partition.csv",
    out_dir="./output",
    sample_n=2000,
    run_embeddings=False
)
```

### Step 2. Prepare HDF5 Dataset

Run the data exporter script. This will create a single `celeba_128.hdf5` file containing the pre-processed images. This step only needs to be done once.

```bash
# Example command (assuming you've merged the HDF5Exporter script into a file)
python create_celeba_hdf5.py \
    --celeba_dir /path/to/img_align_celeba \
    --hdf5_path ./data/celeba_128.hdf5 \
    --num_images 30000 # Or 0 to process all images
```

### Step 3. Train the IMLE GAN

Run the training script. Before running, edit `train_imle_gan.py` to ensure the `h5_path` variable points to the file created in Step 2.

```bash
python train_imle_gan.py
```

## Summary of IMLE GAN Approaches

This project contains three iterations of the training script, demonstrating the process of stabilizing a GAN.

### Approach 1: Baseline IMLE-GAN

* **Models:** Simple style-based Generator and a standard CNN Discriminator.
* **Loss:** `BCELoss` (GAN) + `MSELoss` (IMLE).
* **IMLE Strategy:** For each real image, `k=5` latent vectors are sampled. These are passed through the generator to create 5 fake images. The L2 distance is calculated between the real image and all 5 fakes. The IMLE loss is the L2 distance of the *closest* fake, encouraging the generator to produce at least one sample that can reconstruct the real image.
* **Result:** **Training Failed.** The discriminator loss quickly collapsed to 0.0, while the generator loss exploded to >13.5. This indicates the discriminator became too powerful, and the generator failed to learn.

### Approach 2: Stabilized (Warmup & Clipping)

* **Models:** Same as Approach 1.
* **Changes:**
    * **IMLE Warmup:** IMLE loss is only added *after* epoch 2 (`imle_start_epoch = 2`), allowing the generator to first focus on the simpler GAN objective.
    * **IMLE Weighting:** The IMLE loss contribution is reduced (`imle_weight = 0.1`).
    * **Gradient Clipping:** `torch.nn.utils.clip_grad_norm_` is applied to both G and D to prevent exploding gradients.
    * **Slower Discriminator:** The discriminator's learning rate is reduced (`1e-4`) to prevent it from overpowering the generator (`2e-4`).
* **Result:** An intermediate step (logs not shown) designed to fix the instability of Approach 1.

### Approach 3: Advanced Stabilization (Final Model)

* **Models:**
    * **Generator:** Added `BatchNorm2d` to each generator block for more stable feature outputs.
    * **Discriminator:** Added `SpectralNorm` to all convolutional and linear layers. This is a powerful technique to enforce the 1-Lipschitz constraint, which is critical for GAN stability.
* **Changes:**
    * **Label Smoothing:** Real labels for the discriminator are set to `0.9` instead of `1.0`. This prevents the discriminator from becoming over-confident.
    * **Longer Warmup:** IMLE warmup increased to 4 epochs.
    * **Hyperparameter Tuning:** IMLE weight was further reduced to `0.05`, and the D learning rate was dropped to `5e-5` to balance against the spectrally-normalized layers.
* **Result:** **Stable Training.** The logs show a successful 20-epoch run where both D and G losses remain balanced and converge. The D loss stabilizes around ~0.35, and the G loss around ~4.24, indicating neither model has collapsed.

## Results Summary

The initial baseline GAN (Approach 1) failed due to discriminator overpowering, a common GAN failure mode.

The final model (Approach 3) achieved stable training by incorporating a suite of modern stabilization techniques. The key takeaways from the iterative process are:

* **Spectral Norm (in D) and Batch Norm (in G)** are highly effective architectural changes for stabilizing training.
* **Label Smoothing** is a simple and crucial trick to prevent the discriminator loss from collapsing to zero.
* **Slowing down the discriminator** with a lower learning rate (`lr_D < lr_G`) is essential for balance.
* **IMLE loss** requires careful **weighting** and a **warmup period**. Adding it from the beginning with a high weight (as in Approach 1) destabilizes the generator.
