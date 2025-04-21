import numpy as np
import healpy as hp
from jax import config, vmap, grad, jit
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax
from jax.lax import stop_gradient

# dipole direction in galactic coordinates
theta0 = np.deg2rad(48) 
phi0 = np.deg2rad(264)
hn = np.array([[np.sin(theta0)*np.cos(phi0),np.sin(theta0)*np.sin(phi0),np.cos(theta0)]])


def HD(pvec,qvec):
    """
    Calculate the Hellings and Downs correlation function
    pvec, qvec are the two pulsar vectors, normalisation convention of arxiv: 2006.14570
    """
    x =  jnp.sum(pvec*qvec)
    y = 0.5 * (1 - x)
    val = 1 + x/3 + 4*y*jnp.log(y) 
    return val

def psr_dipole(x):
    """
    Calculate the dipole correlation function
    pvec, qvec are the two pulsar vectors
    """
    y = 0.5 * (1 - x)
    val = 8 * (1/12 + y/2 + (y/2)*jnp.log(y)/(1-y) ) # 8 comes from the normalisation convention
    val = jnp.where(jnp.abs(y)<1e-10,3/4,val)
    val = jnp.where(jnp.abs(y-1)<1e-10,3/4,val)
    return val

@jit
def pta_x_astro(pvec,nvec):
    """
    Calculate the cross correlation between pulsar and star
    pvec = pulsar vector
    nvec = star vector
    """
    fac = 16*np.pi/3
    y = (1-jnp.dot(pvec,nvec))/2
    pn = jnp.dot(pvec,nvec)
    vec = (pn*nvec - pvec)/(1-pn**2)
    yterm = (2*y - 2*y**2 + 3*y**2*jnp.log(y))
    return fac * vec * yterm


#Legendre polynomial representation of Fy
def poly_fy(x):
    coeffs = jnp.array([
        0.0561012,  # x^9
        0.0506851,  # x^8
        -0.0697414, # x^7
        -0.0428234, # x^6
        0.0650654,  # x^5
        0.067935,   # x^4
        0.089599,   # x^3
        0.280655,   # x^2
        1.4293,     # x^1
        -0.88075    # x^0
    ])

    return jnp.polyval(coeffs,x)

def pairwise_monopole(pvec,nvec):
    """
    Monpole ORF for astrometry using Legendre polynomials for Fy. Pvec is N x 3 where N is the number of stars
    """
    x = jnp.dot(pvec,nvec)
    y = 0.5*(1 - x)
    fy = poly_fy(x)
    ij_terms =  (2-2*y)*jnp.eye(3) - jnp.outer(pvec,pvec) - jnp.outer(nvec,nvec) - jnp.outer(nvec,pvec) + (1-2*y)*jnp.outer(pvec,nvec)
    return stop_gradient(fy * ij_terms)

def pairwise_monopole_v2(pvec,nvec):
    """
    Monpole ORF for astrometry using conditionals to handle some edge cases. Pvec is N x 3 where N is the number of stars
    """
    x = jnp.dot(pvec,nvec)
    y = 0.5*(1 - x)
    log_term = jnp.where(y>0,jnp.log(y)*6*y**2,0.)
    fy = jnp.where(y<1,(1 - 8*y + 7*y**2 -  log_term) * np.pi/(3*(1-y)**2),-2*np.pi/3)
    ij_terms =  (2-2*y)*jnp.eye(3) - jnp.outer(pvec,pvec) - jnp.outer(nvec,nvec) - jnp.outer(nvec,pvec) + (1-2*y)*jnp.outer(pvec,nvec)
    return fy * ij_terms

def proj_angular(star_pos):
    """
    Calculate the projection matrix to go from cartesian to angular deflections for the system of stars.
    Has shape 2N x 3N
    """
    x,y,z = star_pos.T
    r1 = [jnp.zeros_like(z),jnp.zeros_like(z),1/jnp.sqrt(1-z**2)]
    r2 = [-y/(x**2 + y**2),x/(x**2 + y**2),jnp.zeros_like(z)]
    p = jnp.array([r1,r2])
    p = jnp.transpose(p,axes = (2,0,1))
    return p

# utility function to split vmap over a batch of inputs,  minor modifications of https://github.com/martinjankowiak/saasbo/blob/main/util.py
def split_vmap(func,input_arrays,batch_size=8):
    num_inputs = input_arrays[0].shape[0]
    num_batches = (num_inputs + batch_size - 1 ) // batch_size
    batch_idxs = [jnp.arange( i*batch_size, min( (i+1)*batch_size,num_inputs  )) for i in range(num_batches)]
    res = [vmap(func)(*tuple([arr[idx] for arr in input_arrays])) for idx in batch_idxs]
    nres = len(res[0])
    # now combine results across batches and function outputs to return a tuple (num_outputs, num_inputs, ...)
    results = tuple( jnp.concatenate([x[i] for x in res]) for i in range(nres))
    return results