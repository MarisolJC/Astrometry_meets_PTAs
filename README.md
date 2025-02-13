# Measuring SGWB with Astrometry and PTAs

This repository contains codes[^1] to calculate and visualise the Overlap Reduction Functions (ORFs) reported in :page_facing_up: 2412.14010.

## How do we plot our Maps?

![K1K1_dipole_Random](https://github.com/user-attachments/assets/57a52a54-28e9-47e0-92a4-619edd5c1c58)

You will find this information in :file_folder: **Maps-PTA_Astro**. The folder contains 4 Jupyter notebooks:
- **Maps_Autocorrelation.ipynb** : Plots maps for the astrometry ORFs $\mathrm{Tr}[\mathbf{H}_0 \mathbf{H}_0]$ and $\mathrm{Tr}[\mathbf{H}_1 \mathbf{H}_1]$.
- **Maps_PTAxAstro.ipynb** : Plots maps for the PTA_astrometry cross-correlation ORFs $\mathbf{K}_0 \mathbf{K}_0^{T}$ and $\mathbf{K}_1 \mathbf{K}_1 ^{T}$.
- **ORFs_Visualisation.ipynb** : If you have never used **Healpy** for visualisation, this notebook contains detailed explanations for this.
- **pta_x_astro_Test.ipynb** : This is a numerical test for the ORFs. It calculates the integrals (2.10) and (2.27) of the paper.

## Forecast 

In the folder with this name you will find a notebook to plot the forecasts we found for the magnitude of the SGWB energy density with the astrometry estimators $p_0$ and $p_1$.


[^1]: This codes where runed with Python version 3.11.9, numpy 1.26.4, and jax 0.4.31


## To cite our work

```
@article{cruz2024astrometry,
  title={Astrometry meets Pulsar Timing Arrays: Synergies for Gravitational Wave Detection},
  author={Cruz, NM and Malhotra, Ameek and Tasinato, Gianmassimo and Zavala, Ivonne},
  journal={arXiv preprint arXiv:2412.14010},
  year={2024}
}
```
