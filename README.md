### Repository Overview

The repository contains the full pipeline for the project *VQ-VAE a posteriori with Geodesic Quantization*.

* **`Part1.ipynb` – Continuous VAE**
  Trains a convolutional VAE on CIFAR-10 and extracts latent means \$\mu(x)\$.

* **`Part2.ipynb` – Geodesic Quantization**
  Builds a \$k\$-NN graph, computes geodesic distances, and applies \$K\$-Medoids clustering to form the discrete codebook.

* **`Part3.ipynb` – Autoregressive Prior**
  Prepares discrete sequences and trains a Transformer prior on geodesic codes, with comparison to a baseline.

* **`Part4.ipynb` – Evaluation**
  Compares reconstruction, generative quality (FID/IS), and codebook utilization between GeoQuant and VQ-VAE baseline.

* **`vae.py`**: convolutional VAE implementation.

* **`vqvae_baseline.py`**: standard VQ-VAE with commitment loss.

* **`geodesic_kmedoids.py`**: graph-based clustering with geodesic distances.

* **`results/`**: trained models, checkpoints, histories, and codebooks.

* **`output_imgs/`**: qualitative reconstructions and sample generations.

* **`data/`**: CIFAR-10 dataset and processed splits.


