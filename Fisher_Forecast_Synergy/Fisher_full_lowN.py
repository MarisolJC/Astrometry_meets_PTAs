# in case of memory issues, run on cluster
# for noiseless PTA simply remove pta noise term in the Fisher matrix

from scipy.linalg import LinAlgError
from math import sqrt
import numpy as np
import scipy
from scipy.interpolate import interp1d
import healpy as hp
import sys
import os
os.environ['XLA_FLAGS'] = "--xla_force_host_platform_device_count=6" # in case running on a cluster with multiple cpus
import scipy.linalg
import jax.numpy as jnp
import jax
from jax import vmap, grad, jit
from jax import config
from jax.lax import stop_gradient
config.update("jax_enable_x64", True)
from helper_functions import HD, proj_angular, pta_x_astro
from helper_functions import pairwise_monopole_v2 as pairwise_monopole
import hasasia.sensitivity as hsen
import hasasia.sim as hsim
import time
# plot settings
import matplotlib
import matplotlib.pyplot as plt
from getdist import plots, MCSamples
year = 365*24*3600
np.set_printoptions(precision=8,suppress=False)
jnp.set_printoptions(precision=8,suppress=False)
font = {'size'   : 14, 'family':'Serif'}
axislabelfontsize='large'
matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True)


# The GWB
Tobs = 15*year
df = 1/Tobs
nfreqs = 14
freqs = 1/Tobs + df*np.arange(0,nfreqs) #  +df/2

fref = 1/year
logamp = -14.6 #np.log10(6.4e-15) 
logI = 2*logamp - np.log10(2*fref)
gamma = 13/3
nI = 2 - gamma
gwb_args = (logamp,gamma)

def gwb_model(logamp,gamma,f=1/year,fref=1/year):
    logI = 2*logamp - np.log10(2*fref)
    I_f = 10**logI * (f/fref)**(2-gamma)
    return I_f

# Pulsar only part
Tobs = 15*year
npsr =  75
np.random.seed(100000)
phi = np.random.uniform(0, 2*np.pi,size=npsr)
cos_theta =  np.random.uniform(-0.96,0.96,size=npsr)
#This ensures a uniform distribution across the sky.
theta = np.arccos(cos_theta)
psr_pos = np.array( [np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)]  ).T
hfreqs = np.logspace(np.log10(5e-10),np.log10(5e-7),500)
psrs = hsim.sim_pta(timespan=15, cad=20, sigma=1e-7,
                    phi=phi,theta=theta,A_rn=3.25e-15,alpha=-2/3,freqs=hfreqs)

sp = hsen.Spectrum(psrs[0], freqs=hfreqs)
sigmasq_hsim = interp1d(hfreqs,1/sp.NcalInv) # type: ignore

pta_cov = vmap(vmap(HD,in_axes=(0,None),out_axes=(0)),in_axes=(None,0),out_axes=(0)) (psr_pos,psr_pos)
pta_cov = np.nan_to_num(pta_cov,nan=0.) # nans come from the log term in HD
pta_cov = pta_cov - np.diag(np.diag(pta_cov)) # remove diagonal to make zeros on diagonal
pta_cov = pta_cov + 8/3 * np.eye(npsr) # diagonal terms should be 8/3
print(np.isnan(pta_cov).any())

print("Calculating PTA only Fisher")
def pta_fisher_matrix(f,gwb_args,):
    nparams = len(gwb_args)
    mat = np.zeros((nparams,nparams))
    fac = 1/ (4 * np.pi * f)**2 
    grad_fn = grad(gwb_model,argnums=list(np.arange(nparams)))
    cov = fac*pta_cov*gwb_model(*gwb_args,f=f) + sigmasq_hsim(f)*np.eye(npsr)
    inv_cov = np.linalg.inv(cov)
    # print("Cov shape",cov.shape)
    for i in range(nparams):
        grad_terms = np.transpose( fac*pta_cov[:,:,None] * np.array(grad_fn(*gwb_args,f=f))[None,:]
                                  ,axes=(2,0,1))
        for j in range(0,i+1):
            dcov_i  = grad_terms[i]
            dcov_j  = grad_terms[j]
            mi =  inv_cov @ dcov_i 
            mj = inv_cov @ dcov_j
            mat[i,j] = np.trace(mi@mj)
            mat[j,i] = mat[i,j]
    del mi,mj, grad_fn, grad_terms, dcov_i, dcov_j,cov, inv_cov
    return 0.5 * mat

pta_only_fisher = 0.
for f in freqs:
    pta_only_fisher+= pta_fisher_matrix(f,gwb_args)

param_cov_pta = np.linalg.inv(pta_only_fisher)
print("Param cov pta\n",param_cov_pta)

# pta only numbers
names = ['logA','gamma']
samples_pta_only = np.random.multivariate_normal(mean = gwb_args, cov = param_cov_pta,size=int(1e6) )
labels = [r"\log_{10} A_{\rm GW}",r"\gamma"]
gdsamples_pta = MCSamples(samples=samples_pta_only,names=names,labels=labels,)

keys = ['logA','gamma']

for name in keys:
    print(gdsamples_pta.getInlineLatex(name,limit=2,err_sig_figs=4))



# now the astro setup
nstar = int(sys.argv[1])
Tobs = 15*year

# astro noise levels
sigma_mas = 0.0002
conv_fac = 4.84814e-6 # mas to radians
def noise_astro(f,sigma=sigma_mas,dT=year/52.): # every week 52, every 2 weeks 26,...
    sigma = sigma * 1e-3 * conv_fac
    noise = 2 * sigma**2 * dT
    return noise

np.random.seed(100001)
phi = np.random.uniform(0., 2*np.pi,size=nstar)
cos_theta = np.random.uniform(-0.95,0.95,size=nstar)
#This ensures a uniform distribution across the sky.
theta =  np.arccos(cos_theta)
star_pos = np.transpose(np.array( [np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])  )
star_pos = jnp.array(star_pos)
# mesh = jax.make_mesh((2,3), ('x', 'y')) # this can be to put the arrays across devices when running on a cluster
# stars = jax.device_put(star_pos,NamedSharding(mesh,P('x','y')))
# jax.debug.visualize_array_sharding(stars)
# jax.debug.visualize_array_sharding(stars)
stars = star_pos
del star_pos, theta,cos_theta,phi

# # setup the auto cov-mat
f = lambda vec: vmap(pairwise_monopole,in_axes=(None,0),)(vec,stars)

# direct vmap method
star_cov = vmap(vmap(pairwise_monopole,in_axes=(None,0)),in_axes=(0,None))(stars,stars)

star_cov = jnp.transpose(star_cov,(0,2,1,3)) 
proj_mat = proj_angular(stars)
star_cov = jnp.einsum("imj,ijkl, knl -> imkn",proj_mat,star_cov,proj_mat)
star_cov = star_cov.reshape(2*nstar,2*nstar) / (4*np.pi)
print("star cov shape",star_cov.shape)
# jax.debug.visualize_array_sharding(star_cov)

#setup the cross-cov mat
def cross_single_pulsar(psr_pos):
    cross = vmap(pta_x_astro,in_axes=(None,0),out_axes=(0))(psr_pos,stars)
    cross_ang = jnp.einsum('ijk,ik...->ij',proj_mat,cross)
    return stop_gradient(cross_ang)

cross_cov = jax.lax.map(cross_single_pulsar,psr_pos,batch_size=5)
cross_cov = jnp.nan_to_num(cross_cov,nan=0.)
cross_cov = jnp.transpose(cross_cov,axes=(0,1,2)).reshape(npsr,2*nstar) / (4*np.pi)
print("computed astrometric auto and cross")

del proj_mat, stars
jax.clear_caches()

def astro_fisher_term_ij(f,gwb_args,grad_term_i,grad_term_j,gwb_model=gwb_model,star_cov=star_cov):
    res = 0.
    C = star_cov * gwb_model(*gwb_args,f=f) + noise_astro(f,) * jnp.eye(2*nstar)
    # inv_C = jnp.linalg.inv(C)
    dC_i = star_cov * grad_term_i
    dC_j = star_cov * grad_term_j
    cho = jax.scipy.linalg.cho_factor(C,lower=True) 
    if (jnp.isnan(cho[0]).any()):
        inv_C = scipy.linalg.pinvh(C) 
        res = jnp.einsum("ij,jk,kl,li",inv_C,dC_i,inv_C,dC_j)
    else:
        invC_dCi = jax.scipy.linalg.cho_solve(cho,dC_i)
        invC_dCj = jax.scipy.linalg.cho_solve(cho,dC_j)
        res = jnp.einsum("ij,ji",invC_dCi,invC_dCj)
    return res

def astro_fisher(freqs,gwb_args,gwb_model=gwb_model,star_cov=star_cov,):    
    nparams = len(gwb_args)
    print(f"Fisher for {nparams} parameters")
    print(f"Using frequencies\n {freqs*Tobs}")
    mat = np.zeros((nparams,nparams))
    fish = np.zeros((nparams,nparams))
    grad_fn = grad(gwb_model,argnums=list(range(nparams)))
    grad_terms = jnp.zeros(2)
    fidx=0
    for f in freqs:
        print(f"step {fidx}, frequency = {f:.2e}")
        grad_terms = jnp.array(grad_fn(*gwb_args,f=f))
        for i in range(nparams):
            for j in range(0,i+1):
                mat[i,j] =  astro_fisher_term_ij(f=f,gwb_args=gwb_args,
                                        grad_term_i=grad_terms[i],grad_term_j=grad_terms[j]
                                        ,gwb_model=gwb_model,star_cov=star_cov,)
                mat[j,i] = mat[i,j]
        fish+=mat
        fidx+=1
    return 0.5 * fish

gwb_args = (logamp,gamma)
print("Original PTA freqs\n",freqs*Tobs)
mat = astro_fisher(freqs,gwb_args,)
param_cov_astro = np.linalg.inv(mat)
print("Astro only param cov\n",param_cov_astro)

# astro only numbers
names = ['logA','gamma']
samples_astro_only = np.random.multivariate_normal(mean = gwb_args, cov = param_cov_astro,size=int(1e6) )
labels = [r"\log_{10} A_{\rm GW}",r"\gamma"]
gdsamples_astro = MCSamples(samples=samples_astro_only,names=names,labels=labels,)

keys = ['logA','gamma']

for name in keys:
    print(gdsamples_astro.getInlineLatex(name,limit=2,err_sig_figs=4))


# Full PTA+Astro Fisher calculation
def full_from_blocks(A,B,D): # to get fullcov and derivatives from blocks
    mat1 = jnp.hstack([A,B])
    mat2 = jnp.hstack([B.T,D])
    return jnp.vstack([mat1,mat2])

def fisher_term_ij(f,gwb_args,grad_term_i,grad_term_j,
                   gwb_model=gwb_model,pta_cov=pta_cov,
                   cross_cov=cross_cov,star_cov=star_cov):
    res = 0.
    fac = 1/ (4*np.pi*f)
    A = fac**2 * pta_cov * gwb_model(*gwb_args,f=f) +  sigmasq_hsim(f) * jnp.eye(npsr)
    B =  fac * cross_cov * gwb_model(*gwb_args,f=f)
    D = star_cov*gwb_model(*gwb_args,f=f) + noise_astro(f,)*jnp.eye(2*nstar)
    C = full_from_blocks(A,B,D) 
    A_i = fac**2 * pta_cov * grad_term_i
    A_j = fac**2 * pta_cov * grad_term_j
    B_i = fac * cross_cov * grad_term_i
    B_j = fac * cross_cov * grad_term_j
    D_i = star_cov * grad_term_i
    D_j = star_cov * grad_term_j
    dC_i = full_from_blocks(A_i,B_i,D_i)
    dC_j = full_from_blocks(A_j,B_j,D_j) 
    del B,D,B_i,B_j,D_i,D_j
    cho = jax.scipy.linalg.cho_factor(C,lower=True) 
    if (jnp.isnan(cho[0]).any()):
        inv_C = scipy.linalg.pinvh(C)
        res = jnp.einsum("ij,jk,kl,li",inv_C,dC_i,inv_C,dC_j)
    else:
        invC_dCi = jax.scipy.linalg.cho_solve(cho,dC_i)
        invC_dCj = jax.scipy.linalg.cho_solve(cho,dC_j)
        res = jnp.einsum("ij,ji",invC_dCi,invC_dCj)
    return res

def cross_fisher(freqs,gwb_args,gwb_model=gwb_model,pta_cov=pta_cov,star_cov=star_cov,cross_cov=cross_cov):    
    nparams = len(gwb_args)
    print(f"Fisher for {nparams} parameters")
    print(f"Using frequencies\n {freqs*Tobs}")
    mat = np.zeros((nparams,nparams))
    fish = np.zeros((nparams,nparams))
    grad_fn = grad(gwb_model,argnums=list(range(nparams)))
    grad_terms = jnp.zeros(2)
    fidx = 0
    for f in freqs:
        print(f"step {fidx}, frequency = {f:.2e}")
        grad_terms = jnp.array(grad_fn(*gwb_args,f=f))
        for i in range(nparams):
            for j in range(0,i+1):
                mat[i,j] =  fisher_term_ij(f=f,gwb_args=gwb_args,
                                        grad_term_i=grad_terms[i],grad_term_j=grad_terms[j]
                                        ,gwb_model=gwb_model,pta_cov=pta_cov,star_cov=star_cov,cross_cov=cross_cov)
                mat[j,i] = mat[i,j]
        fish+=mat
        fidx+=1
        print(np.linalg.inv(0.5*fish))
    return 0.5 * fish

print(f"Npsr = 75, Nstar = {nstar}, full Fisher matrix")
gwb_args = (logamp,gamma)
print("Original PTA freqs\n",freqs*Tobs)
mat = cross_fisher(freqs,gwb_args,)
print("PTA only param cov\n",param_cov_pta)
param_cov_cross = np.linalg.inv(mat)
print("PTA + Cross param cov\n",param_cov_cross)

names = ['logA','gamma']

samples_pta_only = np.random.multivariate_normal(mean = gwb_args, cov = param_cov_pta,size=int(1e6) )
samples_pta_astro = np.random.multivariate_normal(mean = gwb_args, cov = param_cov_cross,size=int(1e6) )
labels = [r"\log_{10} A_{\rm GW}",r"\gamma",r'\beta']
gdsamples_pta = MCSamples(samples=samples_pta_only,names=names,labels=labels,)
gdsamples_pta_astro = MCSamples(samples=samples_pta_astro,names=names,labels=labels,)

keys = ['logA','gamma']

for s in [gdsamples_pta,gdsamples_pta_astro]:
    for name in keys:
        print(s.getInlineLatex(name,err_sig_figs=4))


markers = dict(zip(keys,gwb_args))

g = plots.get_subplot_plotter(subplot_size=3.,subplot_size_ratio=1/1.4,scaling=True)
g.settings.axes_fontsize=16
g.settings.legend_fontsize = 16
g.settings.axes_labelsize = 16
g.settings.title_limit_fontsize = 14
g.triangle_plot([gdsamples_pta_astro,gdsamples_pta_astro],keys,
                legend_labels=[f"PTA, {npsr} pulsars",f"+ Astrometry, $10^{int(np.log10(nstar))}$ stars"],
                filled=[False,True],markers=markers,contour_colors=['blue','red'], # type: ignore
                contour_lws=[1.25,1.25]) 
g.export(f'fig8a.pdf')