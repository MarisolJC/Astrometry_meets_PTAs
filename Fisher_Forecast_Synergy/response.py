import numpy as np
from jax import config, vmap
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import healpy as hp


I_3 = jnp.eye(3)

def skyvec(theta,phi): #unit vector
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    return np.array([st*cp,st*sp,ct])

theta0 = np.deg2rad(48) 
phi0 = np.deg2rad(264)
vvec = skyvec(theta0,phi0)

def dir(theta,phi):
    return np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)]).T

def projector(pvec):
    """
    Calculate the projection tensor
    """
    kd_ijkl = jnp.einsum('ik,jl->ijkl',I_3,I_3) + jnp.einsum('il,jk->ijkl',I_3,I_3) - jnp.einsum('ij,kl->ijkl',I_3,I_3) 
    proj_ijkl =  kd_ijkl + jnp.einsum('i...,j...,k...,l...->ijkl...',pvec,pvec,pvec,pvec) \
                - jnp.einsum('ik,j...,l...->ijkl...',I_3,pvec,pvec) - jnp.einsum('jl,i...,k...->ijkl...',I_3,pvec,pvec) \
                - jnp.einsum('il,j...,k...->ijkl...',I_3,pvec,pvec) - jnp.einsum('jk,i...,l...->ijkl...',I_3,pvec,pvec) \
                + jnp.einsum('ij,k...,l...->ijkl...',I_3,pvec,pvec) + jnp.einsum('kl,i...,j...->ijkl...',I_3,pvec,pvec)
    return proj_ijkl

def single_star_and_pix_response(nvec,pvec):
    """
    Calculate the response tensor for a single star and pixel
    nvec = star direction
    pvec = pixel direction
    """
    ndotp = jnp.sum(nvec*pvec)
    n_plus_pvec = (nvec+pvec)/(1+ndotp)
    R_ikl = 0.5 * (jnp.einsum('i,k,l->ikl',n_plus_pvec,nvec,nvec) - jnp.einsum('k,il->ikl',nvec,jnp.eye(3)) )
    return R_ikl


def pairwise_single_pix_response(nvec,qvec,pvec):
    """
    Calculate the response tensor for a pair of stars and pixel
    nvec = star direction
    qvec = star direction
    pvec = pixel direction
    """
    R_ikl = single_star_and_pix_response(nvec,pvec)
    R_jrs = single_star_and_pix_response(qvec,pvec)
    proj_klrs = projector(pvec)
    H_ij = jnp.einsum('ikl,jrs,klrs->ij',R_ikl,R_jrs,proj_klrs)
    return H_ij

def pairwise_monopole(nvec,qvec,pvec_array,nside):
    """
    Calculate the monopole response for a pair of stars and pixels
    nvec = star direction
    qvec = star direction
    pvec_array = pixel directions
    nsides = healpix nside
    """
    da = hp.nside2pixarea(nside)
    f = lambda pvec: pairwise_single_pix_response(nvec=nvec,qvec=qvec,pvec=pvec)
    pairwise_antenna = vmap(f,in_axes=(0),out_axes=(-1))(pvec_array)
    return da*jnp.sum(pairwise_antenna,axis=-1)

def pairwise_dipole(nvec,qvec,pvec_array,nside,vvec=vvec):
    """
    Calculate the dipole response for a pair of stars and pixels
    nvec = star direction
    qvec = star direction
    pvec_array = pixel directions
    nside = healpix nside
    vvec = dipole direction
    """
    da = hp.nside2pixarea(nside)
    f = lambda pvec: jnp.sum(vvec*pvec) * pairwise_single_pix_response(nvec=nvec,qvec=qvec,pvec=pvec)
    pairwise_antenna = vmap(f,in_axes=(0),out_axes=(-1))(pvec_array)
    return da*jnp.sum(pairwise_antenna,axis=-1)

# Puslar timing functions

def pulsar_single_pix_response(xvec,pvec):
    """
    Calculate the response tensor for a pulsar and pixel
    xvec = pulsar direction
    pvec = pixel direction
    """
    xdotp = jnp.sum(xvec*pvec)
    return jnp.einsum("i,j->ij",xvec,xvec)/(1+xdotp)

def pulsar_star_single_pix(xvec,nvec,pvec):
    """
    Calculate the cross-response tensor for a pulsar and star and pixel
    nvec = pulsar
    xvec = star
    """
    R_ikl = single_star_and_pix_response(xvec,pvec)
    proj_klrs = projector(pvec)
    P_rs = pulsar_single_pix_response(nvec,pvec)
    K_i = jnp.einsum("ikl,klrs,rs->i",R_ikl,proj_klrs,P_rs)
    return K_i

def pulsar_star_pair_monopole(nvec,xvec,pvec_array,nside):
    """
    Calculate the monopole response for a pulsar and star and pixels
    nvec = pulsar
    xvec = star
    pvec_array = pixel directions
    nside = healpix nside
    """
    da = hp.nside2pixarea(nside)# /fourpi
    f = lambda pvec: pulsar_star_single_pix(nvec=nvec,xvec=xvec,pvec=pvec)
    pairwise_antenna = vmap(f,in_axes=(0),out_axes=(-1))(pvec_array)
    return da*jnp.sum(pairwise_antenna,axis=-1)

def pulsar_star_pair_dipole(nvec,qvec,pvec_array,nside,vvec=vvec):
    """
    Caclulate the dipole response for a pulsar and star and pixels
    nvec = pulsar
    qvec = star
    pvec_array = pixel directions
    nside = healpix nside
    vvec = dipole direction
    """
    da = hp.nside2pixarea(nside)# /fourpi
    f = lambda pvec: -jnp.sum(vvec*pvec) * pulsar_star_single_pix(nvec=nvec,xvec=qvec,pvec=pvec)
    pairwise_antenna = vmap(f,in_axes=(0),out_axes=(-1))(pvec_array)
    return da*jnp.sum(pairwise_antenna,axis=-1)