# Comparison of Deep Generative Models

In recent years, generative models have been very effective in learning the under-lying distribution of data sets and are able to reproduce a variety of results thatare indistinguishable from the original dataset.  In this project we will compareand discuss the generative capabilities of three different deep generative models:GAN, Wasserstein-GAN, and Variational autoencoders (VAE) on the EMNISThand-written letters data set

## Build Environment
```
conda env create -f environment.yml
conda activate gan
``` 

### Training WGAN

To train WGAN, run all the cells in:
```
WGAN.ipynb
```
To generate a few WGAN samples from a pretrained model call:
```
python wgan_generate_samples.py
```
To evaluate the reconstruction error:
```
python wgan_reconstruction_error.py
```
