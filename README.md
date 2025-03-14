# Measuring SGWB with Astrometry and PTAs

This repository contains codes[^1] to calculate and visualise the Overlap Reduction Functions (ORFs) reported in :page_facing_up: 2412.14010.

## SGWB Maps

![K1K1_dipole_Random](https://github.com/user-attachments/assets/57a52a54-28e9-47e0-92a4-619edd5c1c58)

* :file_folder: **Maps-PTA_Astro** contains 4 Jupyter notebooks:
  
  1. **Maps_Autocorrelation.ipynb** : Plots maps for the astrometry ORFs $\mathrm{Tr}[\mathbf{H}_0 \mathbf{H}_0]$ and $\mathrm{Tr}[\mathbf{H}_1 \mathbf{H}_1]$.
     
  2. **Maps_PTAxAstro.ipynb** : Plots maps for the PTA_astrometry cross-correlation ORFs $\mathbf{K}_0 \mathbf{K}_0^{T}$ and $\mathbf{K}_1 \mathbf{K}_1 ^{T}$.
  
  3. **ORFs_Visualisation.ipynb** : If you have never used **Healpy** for visualisation, this notebook contains detailed explanations for this.
     
  4. **pta_x_astro_Test.ipynb** : This is a test for the ORFs. It calculates the integrals (2.10) and (2.22) of the paper and compare them with the analytical results from (2.15) and (2.27).

## ORFs Angular dependence 

* :file_folder: **Plots** has a notebook to show the behavior of the ORFs with respect to the angles between pairs of stars or between stars and pulsars.

## Forecast 

* You will find a notebook to plot the forecasts found for the magnitude of the SGWB energy density with the astrometry estimators $p_0$ and $p_1$ in :file_folder: **Forecast**.


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
