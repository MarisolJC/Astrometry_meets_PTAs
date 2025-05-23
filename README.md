# Measuring SGWB with Astrometry and PTAs

This repository contains codes to calculate and visualise the Overlap Reduction Functions (ORFs), as well as the Fisher forecasts reported in :page_facing_up: [2412.14010](https://arxiv.org/abs/2412.14010).

## SGWB Maps

![K1K1_dipole_Random](https://github.com/user-attachments/assets/57a52a54-28e9-47e0-92a4-619edd5c1c58)

* :file_folder: **Maps-PTA_Astro** contains 4 Jupyter notebooks:
  
  1. **Maps_Autocorrelation.ipynb** : Plots maps for the astrometry ORFs $\mathrm{Tr}[\mathbf{H}_0 \mathbf{H}_0]$ and $\mathrm{Tr}[\mathbf{H}_1 \mathbf{H}_1]$.
     
  2. **Maps_PTAxAstro.ipynb** : Plots maps for the PTA_astrometry cross-correlation ORFs $\mathbf{K}_0 \mathbf{K}_0^{T}$ and $\mathbf{K}_1 \mathbf{K}_1 ^{T}$.
  
  3. **ORFs_Visualisation.ipynb** : If you have never used **Healpy** for visualisation, this notebook contains detailed explanations for this.
     
  4. **pta_x_astro_Test.ipynb** : This is a test for the ORFs. It calculates the integrals (2.10) and (2.22) of the paper and compare them with the analytical results from (2.16) and (2.28).

## ORFs Angular dependence 

* :file_folder: **Plots** has a notebook to show the behavior of the ORFs with respect to the angles between pairs of stars or between stars and pulsars.

## Astrometry forecast 

* :file_folder: **Astrometry_Forecast** contains a notebook for the predictions on the magnitude of the SGWB energy density with Astrometry.

## Fisher forecasts for PTA and Astrometry synergy

* The folder :file_folder: **Fisher_Forecast_Synergy** contains the necessary files required to reproduce the results of section 3.2 of the paper. The files helper_functions.py and response are used to define the pta, astrometry and cross-correlation response functions. The other files with can be run from the command line (see the invidual files for the command line inputs). This will require an additional library [Hasasia](https://hasasia.readthedocs.io/en/latest/index.html) to generate the PTA noise curves. The python notebook **noise_vs_signal.ipynb** shows plots of the signal vs noise for the different scenarios studied in the paper.




## To cite our work

```
@article{cruz2024astrometry,
  title={Astrometry meets Pulsar Timing Arrays: Synergies for Gravitational Wave Detection},
  author={Cruz, NM and Malhotra, Ameek and Tasinato, Gianmassimo and Zavala, Ivonne},
  journal={arXiv preprint arXiv:2412.14010},
  year={2024}
}
```
