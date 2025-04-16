import numpy as np
import matplotlib.pyplot as plt
from math import pi, sqrt

# The Fisher matrix is given by
# F_ij = Tobs/(2 * noise_sq^2) * Tr[H_0 H_0]/(4 pi)^2 * \sum_i df * [(f_i/fstar)^(2-gamma)]^2
# frequecies = fmin + df * i

year = 365*24*60*60  # seconds in a year
Tobs = 15 * year
Tcad = year / 15
fstar = 1 / year
fmin = 1 / Tobs
fmax = 1 / Tcad
print(f"fmin = {fmin:.2e}, fmax = {fmax:.2e}")
gamma = 13/3
df = 1 / Tobs
NG15_I0 = 9.1e-23

# Frequency array
nfreqs = 15
f = fmin + df*np.arange(0,nfreqs,1) #np.arange(fmin, fmax, df)
print(f"nfreqs = {len(f)}")
print(f"min freq = {f[0]:.2e}, max freq = {f[-1]:.2e}")
print(f"{f/fmin}")

# noise
theta_rms = 0.01 # mas
conv_fac = 4.84814e-6 # as to radians
theta_rms = theta_rms * conv_fac * 1e-3
noise_sq = theta_rms**2 * 2 * Tcad

def psd(I0,f,fstar=fstar,gamma=gamma):
    return I0*(f/fstar)**(2-gamma)

def fisher_f(f):
    return psd(I0=1.,f=f)**2

def fisher_I0(freqs,Nstar):
    fisher = 0.
    tr_term = 3 * Nstar**2 / (4*pi)**2 
    noise_inv = 2 * noise_sq**2 
    for f in freqs:
        fisher += fisher_f(f)
    fisher = tr_term * fisher * df * Tobs / noise_inv 
    return fisher 

def Delta_I0(freqs,Nstar):
    fisher = fisher_I0(freqs,Nstar)
    delta_I0 = 1 / sqrt(fisher)
    return delta_I0

Nstars = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]

dI0s = []
for Nstar in Nstars:
    dI0 = Delta_I0(f,Nstar)
    dI0s.append(dI0)

# convert to Omega_GW
H0 = 67.2e3 / 3.086e22  # Hubble constant in s^-1
print(f"H0 = {H0:.2e} s^-1")
def Omega_GW(I0):
    return 4 * pi**2 * fstar**3 * I0 / (3 * H0**2) 

def Delta_Omega_GW(I0, Nstar):
    delta_I0 = Delta_I0(f,Nstar)
    delta_Omega_GW = Omega_GW(delta_I0)
    return delta_Omega_GW
dOmegas = []
for Nstar in Nstars:
    dOmega = Delta_Omega_GW(NG15_I0, Nstar)
    dOmegas.append(dOmega)
NG15_Omega_GW = Omega_GW(NG15_I0)

# Plot both graphs in the same figure
plt.figure(figsize=(10, 8))

# Subplot 1: Delta I0 vs Number of stars
plt.subplot(2, 1, 1)
plt.plot(Nstars, dI0s)
plt.axhline(NG15_I0, color='r', linestyle='--', label='NG15')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of stars')
plt.ylabel(r'$\Delta I_0$')
plt.title(r'$\Delta I_0$ vs Number of Stars')
plt.legend()

# Subplot 2: Delta Omega_GW vs Number of stars

plt.subplot(2, 1, 2)
plt.plot(Nstars, dOmegas)
plt.axhline(NG15_Omega_GW, color='r', linestyle='--', label='NG15')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of stars')
plt.ylabel(r'$\Delta \Omega_{GW}$')
plt.title(r'$\Delta \Omega_{GW}$ vs Number of Stars')
# plot the scaling relation
def scaling(val,Nstar=1e6,sigma=0.01,tcad=Tcad,Tobs=Tobs,nI=2-gamma):
    return val * (1e6/Nstar) * (sigma/0.01)**2 * (tcad / (year / 15) ) * (Tobs/Tobs)**(nI)

val = dOmegas[3]  # Use the value for Nstars = 1e6
print(val)
Nstars = np.array(Nstars)
plt.plot(Nstars, scaling(val,Nstar=Nstars,sigma=0.01,tcad=Tcad,Tobs=Tobs,nI=2-gamma),color='g',linestyle=':', label='Scaling relation')
plt.legend()

plt.tight_layout()
plt.savefig('forecast_I0.png', dpi=300)
plt.show()