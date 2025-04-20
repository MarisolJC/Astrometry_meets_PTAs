# Calculate the Fisher matrix for PTA + Astrometry in the weak astro signal limit using the series expansion method presented in the paper.
import numpy as np
from scipy.interpolate import interp1d
import healpy as hp
import sys
import os
import jax.numpy as jnp
import jax
from jax import vmap, grad, jit
from jax import config
from jax.lax import stop_gradient
config.update("jax_enable_x64", True)


from helper_functions import HD, proj_angular, psr_dipole, pta_x_astro, split_vmap
from helper_functions import pairwise_monopole_v2 as pairwise_monopole
from response import pairwise_dipole, pulsar_star_pair_dipole
import hasasia.sensitivity as hsen
import hasasia.sim as hsim
import time
# plot settings
import matplotlib
import matplotlib.pyplot as plt
from getdist import plots, MCSamples
year = 365*24*3600
np.set_printoptions(precision=6,suppress=False)
jnp.set_printoptions(precision=6,suppress=False)
font = {'size'   : 14, 'family':'Serif'}
axislabelfontsize='large'
matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True)


# The GWB
Tobs = 15*year
df = 1/Tobs
nfreqs = 15
freqs = 1/Tobs + df*np.arange(0,nfreqs) # +df/2 + df*np.arange(0,nfreqs)
fref = 1/year
logamp = -14.6 #np.log10(6.4e-15) 
logI = 2*logamp - np.log10(2*fref)
gamma = 13/3
nI = 2 - gamma

# dipole direction and magnitude
theta0 = np.deg2rad(48) 
phi0 = np.deg2rad(264)
hn = np.array([[np.sin(theta0)*np.cos(phi0),np.sin(theta0)*np.sin(phi0),np.cos(theta0)]])
alpha = (1-nI)
beta = 5e-2

def gwb_model(logamp,gamma,beta,f=1/year,fref=1/year):
    logI = 2*logamp - np.log10(2*fref)
    I_f = 10**logI * (f/fref)**(2-gamma)
    return I_f

def gwb_dipole(logamp,gamma,beta,f=1/year,fref=1/year):
    nI = 2 - gamma
    mono = gwb_model(logamp=logamp,gamma=gamma,beta=beta,f=f,fref=fref)
    dip = mono * (1-nI) * beta
    return dip

# Pulsar only part
npsr = int(sys.argv[1])
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

print(f"Psr pos shape {psr_pos.shape}")
pta_cov = vmap(vmap(HD,in_axes=(0,None),out_axes=(0)),in_axes=(None,0),out_axes=(0)) (psr_pos,psr_pos)
pta_cov = np.nan_to_num(pta_cov,nan=0.) # nans come from the log term in HD
pta_cov = pta_cov - np.diag(np.diag(pta_cov)) # remove diagonal to make zeros on diagonal
pta_cov = pta_cov + 8/3 * np.eye(npsr) # diagonal terms should be 8/3 according to our normalisation conventions cov -> cov*(1+delta_{pq})
print(np.isnan(pta_cov).any())

# dipole
hn_dot_hp = np.sum(psr_pos*hn,axis=1)
costheta_mat = jnp.einsum('ij,kj->ik',psr_pos,psr_pos)
pta_cov_dipole = psr_dipole(costheta_mat) 
pta_cov_dipole+= np.diag(np.diag(pta_cov_dipole)) # double the diagonal cov -> cov*(1+delta_{pq})
pta_cov_dipole = pta_cov_dipole * np.add.outer(hn_dot_hp,hn_dot_hp)
print(np.isnan(pta_cov_dipole).any())

gwb_args = (logamp,gamma,beta)

print("Calculating PTA only Fisher")
def pta_fisher_matrix(f,gwb_args,):
    """
    Calculate the Fisher matrix for the PTA, given the GWB model and dipole without any approximation
    """
    nparams = len(gwb_args)
    mat = np.zeros((nparams,nparams))
    fac = 1/ (4 * np.pi * f)**2 
    grad_fn_0 = grad(gwb_model,argnums=list(np.arange(nparams)))
    grad_fn_1 = grad(gwb_dipole,argnums=list(np.arange(nparams)))
    cov = fac*(pta_cov*gwb_model(*gwb_args,f=f) + pta_cov_dipole*gwb_dipole(*gwb_args,f=f)) + sigmasq_hsim(f)*np.eye(npsr)
    inv_cov = np.linalg.inv(cov)
    for i in range(nparams):
        grad_terms = np.transpose( fac*pta_cov[:,:,None] * np.array(grad_fn_0(*gwb_args,f=f))[None,:] + fac*pta_cov_dipole[:,:,None] * np.array(grad_fn_1(*gwb_args,f=f))[None,:]
                                  ,axes=(2,0,1))
        for j in range(0,i+1):
            dcov_i  = grad_terms[i]
            dcov_j  = grad_terms[j]
            mi =  inv_cov @ dcov_i 
            mj = inv_cov @ dcov_j
            mat[i,j] = np.trace(mi@mj)
            mat[j,i] = mat[i,j]
    del mi,mj, grad_fn_0,grad_fn_1, grad_terms, dcov_i, dcov_j,cov, inv_cov
    return 0.5 * mat

pta_only_fisher = 0.
for f in freqs:
    pta_only_fisher+= pta_fisher_matrix(f,gwb_args)

param_cov_pta = np.linalg.inv(pta_only_fisher)
print("Param cov pta\n",param_cov_pta)



# now the astro setup
nstar = int(sys.argv[2])

# astro noise levels
sigma_mas = float(sys.argv[3]) #0.01 # mas
conv_fac = 4.84814e-6 # as to radians
def noise_astro(f,sigma=sigma_mas,dT=year/15.): # every week 52, every 2 weeks 26,...
    sigma = sigma * 1e-3 * conv_fac
    noise = 2 * sigma**2 * dT
    return noise

np.random.seed(100001)
phi = np.random.uniform(0., 2*np.pi,size=nstar)
cos_theta = np.random.uniform(-0.95,0.95,size=nstar)
#This ensures a uniform distribution across the sky.
theta =  np.arccos(cos_theta)
star_pos = np.transpose(np.array( [np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])  ) #.astype(dtype=np.float32)
star_pos = jnp.array(star_pos)


stars = star_pos
del star_pos, theta,cos_theta,phi

proj_mat = proj_angular(stars)

def cross_single_pulsar(qvec):
    proj_factor = proj_angular(jnp.atleast_2d(qvec))
    cross = jnp.einsum('ijk,...ik->ij',proj_factor,vmap(pta_x_astro,in_axes=(0,None),out_axes=(0))(psr_pos,qvec))
    return stop_gradient(cross)

cross_cov = jax.lax.map(cross_single_pulsar,stars,batch_size=200)
cross_cov = jnp.nan_to_num(cross_cov,nan=0.)
cross_cov = jnp.transpose(cross_cov,axes=(1,0,2)).reshape(npsr,2*nstar)  / (4*np.pi)
print("Cross cov shape ",cross_cov.shape)

nside = 4
npix = hp.nside2npix(nside)
pix = np.arange(npix)
pvec_array = np.array(hp.pix2vec(nside,pix)).T

f1 = lambda n, q : pulsar_star_pair_dipole(nvec=n,qvec=q,pvec_array=pvec_array,nside=nside,vvec=hn)
def single_star_cross_PTA(qvec):
    cross = vmap(f1,in_axes=(0,None))(psr_pos,qvec)
    proj_factor = proj_angular(jnp.atleast_2d(qvec))
    cross = jnp.einsum('ijk,...ik->ij',proj_factor,vmap(f1,in_axes=(0,None),out_axes=(0))(psr_pos,qvec))
    return stop_gradient(cross)
cross_cov_dipole = jax.lax.map(single_star_cross_PTA,stars,batch_size=200)
print("Cross cov dipole shape ",cross_cov_dipole.shape)
print("cross cov nans",jnp.isnan(cross_cov_dipole).any())
cross_cov_dipole = jnp.nan_to_num(cross_cov_dipole,nan=0.)
cross_cov_dipole = jnp.transpose(cross_cov_dipole,axes=(1,0,2)).reshape(npsr,2*nstar) / (4*np.pi)
print("Cross cov dipole shape ",cross_cov_dipole.shape)
bb_mat = cross_cov @ jnp.transpose(cross_cov)
bb_mat_dipole = cross_cov_dipole @ jnp.transpose(cross_cov_dipole)
bb_mat_md = cross_cov @ jnp.transpose(cross_cov_dipole)
bb_mat_dm = jnp.transpose(bb_mat_md)
print(bb_mat_dm.shape)
print("computed astrometric auto and cross")


del proj_mat, stars, cross_cov, cross_cov_dipole
jax.clear_caches()


# Full PTA+Astro Fisher calculation
def full_from_blocks(A,B,D): # to get fullcov and derivatives from blocks
    mat1 = jnp.hstack([A,B])
    mat2 = jnp.hstack([B.T,D])
    return jnp.vstack([mat1,mat2])

def series_terms_ij(f,gwb_args,inv_A,A_i,A_j
                    ,grad_term_i,grad_term_j
                    ,grad_dipole_i,grad_dipole_j):
    """
    See paper for details on the series expansion terms
    """
    fac = 1/ (4*np.pi*f)**2
    mono = gwb_model(*gwb_args,f=f)
    dipole = gwb_dipole(*gwb_args,f=f)
    bbT = fac * (bb_mat*mono**2 + bb_mat_dipole*dipole**2 + (bb_mat_md+bb_mat_dm)*mono*dipole )
    bbT_i = fac * (bb_mat * mono * grad_term_i + bb_mat_dipole*dipole*grad_dipole_i + bb_mat_md*grad_term_i*dipole + bb_mat_dm*mono*grad_dipole_i )
    bbT_j = fac * (bb_mat * mono * grad_term_j + bb_mat_dipole*dipole*grad_dipole_j + bb_mat_md*grad_term_j*dipole + bb_mat_dm*mono*grad_dipole_j )
    bbT_ij =  fac * (bb_mat * grad_term_i * grad_term_j + bb_mat_dipole*grad_dipole_i*grad_dipole_j + bb_mat_md*grad_term_i*grad_dipole_j + bb_mat_dm*grad_dipole_i*grad_term_j)
    res = 0.
    res+= jnp.einsum("ij,jk,kl,lm,mn,ni",A_i,inv_A,A_j,inv_A,bbT,inv_A)
    res+= -jnp.einsum("ij,jk,kl,li",A_i,inv_A,bbT_j,inv_A)
    res+= jnp.einsum("ij,ji",bbT_ij,inv_A)
    res+= -jnp.einsum("ij,jk,kl,li",bbT_i,inv_A,A_j,inv_A)
    res+= jnp.einsum("ij,jk,kl,lm,mn,ni",A_i,inv_A,bbT,inv_A,A_j,inv_A)
    res+= -jnp.einsum("ij,jk,kl,li",bbT_i,inv_A,A_j,inv_A)
    res+= jnp.einsum("ij,ji",bbT_ij,inv_A)
    res+= -jnp.einsum("ij,jk,kl,li",A_i,inv_A,bbT_j,inv_A)
    return stop_gradient(res)


def fisher_term_ij(f,gwb_args
                   ,grad_term_i,grad_term_j
                   ,grad_dipole_i,grad_dipole_j,):
    """
    Calculate the Fisher matrix term ij for the PTA + Astro with the series expansion method
    """
    res = 0.
    fac = 1/ (4*np.pi*f)
    A = fac**2 * (pta_cov*gwb_model(*gwb_args,f=f) + pta_cov_dipole*gwb_dipole(*gwb_args,f=f)) +  sigmasq_hsim(f) * jnp.eye(npsr)
    A_i = fac**2 * (pta_cov * grad_term_i + pta_cov_dipole*grad_dipole_i)
    A_j = fac**2 * (pta_cov * grad_term_j + pta_cov_dipole*grad_dipole_j)
    invA = jax.scipy.linalg.inv(A)
    res = jnp.einsum("ij,jk,kl,li",invA,A_i,invA,A_j)
    res+= series_terms_ij(f,gwb_args,invA,A_i,A_j,
                            grad_term_i,grad_term_j,
                            grad_dipole_i,grad_dipole_j)/noise_astro(f,)

    return stop_gradient(res)

def cross_fisher(freqs,gwb_args,gwb_model=gwb_model,):    
    nparams = len(gwb_args)
    print(f"Fisher for {nparams} parameters")
    print(f"Using frequencies\n {freqs*Tobs}")
    mat = np.zeros((nparams,nparams))
    fish = np.zeros((nparams,nparams))
    grad_fn = grad(gwb_model,argnums=list(range(nparams)))
    grad_dipole = grad(gwb_dipole,argnums=list(range(nparams)))
    grad_terms = jnp.zeros(nparams)
    grad_terms_dipole = jnp.zeros(nparams)
    fidx = 0
    for f in freqs:
        grad_terms = jnp.array(grad_fn(*gwb_args,f=f))
        grad_terms_dipole = jnp.array(grad_dipole(*gwb_args,f=f))
        for i in range(nparams):
            for j in range(0,i+1):
                mat[i,j] =  fisher_term_ij(f=f,gwb_args=gwb_args,
                                        grad_term_i=grad_terms[i],grad_term_j=grad_terms[j]
                                        ,grad_dipole_i=grad_terms_dipole[i],grad_dipole_j=grad_terms_dipole[j]
                                        )
                mat[j,i] = mat[i,j]
        fish+=mat
        fidx+=1
    return 0.5 * fish

print(f"Npsr = {npsr}, Nstar = {nstar}, full Fisher matrix")
gwb_args = (logamp,gamma,beta)
print("Original PTA freqs\n",freqs*Tobs)
mat = cross_fisher(freqs,gwb_args,)
print("PTA only param cov\n",param_cov_pta)
# print("Cross fisher\n",mat)
param_cov_cross = np.linalg.inv(mat)
print("PTA + Cross param cov\n",param_cov_cross)



# PTA only vs cross monopole only

mono_cov_pta = np.linalg.inv(np.linalg.inv(param_cov_pta)[:2,:2])
mono_cov_cross = np.linalg.inv(np.linalg.inv(param_cov_cross)[:2,:2])

gwb_args_m = (logamp,gamma)
names = ['logA','gamma']
markers = dict(zip(names,gwb_args_m))
samples_pta_only = np.random.multivariate_normal(mean = gwb_args_m, cov = mono_cov_pta,size=int(1e5) )
samples_pta_astro = np.random.multivariate_normal(mean = gwb_args_m, cov = mono_cov_cross,size=int(1e5) )
labels = [r"\log_{10} A_{\rm GW}",r"\gamma"]
gdsamples_pta = MCSamples(samples=samples_pta_only,names=names,labels=labels,)
gdsamples_pta_astro = MCSamples(samples=samples_pta_astro,names=names,labels=labels,)

for s in [gdsamples_pta,gdsamples_pta_astro]:
    for name in names:
        print(s.getInlineLatex(name,limit=1,err_sig_figs=4))

keys = names
g = plots.get_subplot_plotter(subplot_size=3.25,subplot_size_ratio=1/1.4)
g.settings.axes_fontsize=16
g.settings.legend_fontsize = 16
g.settings.axes_labelsize = 16
g.settings.title_limit_fontsize = 14
g.triangle_plot([gdsamples_pta,gdsamples_pta_astro],keys,
                legend_labels=[f"PTA, {npsr} pulsars",f"+ $10^{int(np.log10(nstar))}$ stars, $ \\sigma = {sigma_mas}$ mas"],
                filled=[False,True],markers=markers,contour_colors=['blue','red'],param_limits={'beta': (0,None)}, # type: ignore
                contour_lws=[1.25,1.25]) 
noise_str = int(sigma_mas*1e3)
g.export(f'Fisher_Cross_monopole_{noise_str:d}' + '_muas.pdf')


## PTA only vs cross with dipole
print(f'amp spectral R cross = {param_cov_pta[0,1]**2 / (param_cov_pta[0,0] * param_cov_pta[1,1] )}')
print(f'amp spectral R pta = {param_cov_cross[0,1]**2 / (param_cov_cross[0,0] * param_cov_cross[1,1] )}')

names = ['logA','gamma','beta']
gwb_args = (logamp,gamma,beta)

samples_pta_only = np.random.multivariate_normal(mean = gwb_args, cov = param_cov_pta,size=int(1e6) )
samples_pta_astro = np.random.multivariate_normal(mean = gwb_args, cov = param_cov_cross,size=int(1e6) )
labels = [r"\log_{10} A_{\rm GW}",r"\gamma",r'\beta']
gdsamples_pta = MCSamples(samples=samples_pta_only,names=names,labels=labels,)
gdsamples_pta_astro = MCSamples(samples=samples_pta_astro,names=names,labels=labels,)


keys = ['logA','gamma','beta']

for s in [gdsamples_pta,gdsamples_pta_astro]:
    for name in keys:
        print(s.getInlineLatex(name,err_sig_figs=4))

markers = dict(zip(keys,gwb_args))

g = plots.get_subplot_plotter(subplot_size=3.,subplot_size_ratio=1/1.4,scaling=True)
g.settings.axes_fontsize=16
g.settings.legend_fontsize = 16
g.settings.axes_labelsize = 16
g.settings.title_limit_fontsize = 14
g.triangle_plot([gdsamples_pta,gdsamples_pta_astro],keys,
                legend_labels=[f"PTA, {npsr} pulsars",f"+ $10^{int(np.log10(nstar))}$ stars, $ \\sigma = {sigma_mas}$ mas"],
                filled=[False,True],markers=markers,contour_colors=['blue','red'],param_limits={'beta': (0,None)}, # type: ignore
                contour_lws=[1.25,1.25]) 
g.export(f'Fisher_Cross_Dipole_{noise_str:d}' + '_muas.pdf')