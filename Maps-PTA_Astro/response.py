import numpy as np
import jax.numpy as jnp
from jax import vmap, grad
import healpy as hp
# from jax import config
# config.update("jax_enable_x64", True)


I_3 = jnp.eye(3)

def skyvec(theta,phi): #unit vector
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    return np.array([st*cp,st*sp,ct]) #hp.ang2vec(theta,phi).T 

theta0 = np.pi/2 - np.deg2rad(48) 
phi0 = np.deg2rad(264)
# vvec = np.array([0,0,1]) # ONLY FOR TEST
#vvec = np.array([0.6, 0.4, 0.69282]) # ONLY FOR TEST
vvec = skyvec(theta0,phi0)

def dir(theta,phi):
    return np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)]).T

#vvec = dir(theta0,phi0)

def projector(pvec):
    # pvec = hp.pix2vec(nside,pix)
    # print(np.shape(pvec))
    kd_ijkl = jnp.einsum('ik,jl->ijkl',I_3,I_3) + jnp.einsum('il,jk->ijkl',I_3,I_3) - jnp.einsum('ij,kl->ijkl',I_3,I_3) 
    proj_ijkl =  kd_ijkl + jnp.einsum('i...,j...,k...,l...->ijkl...',pvec,pvec,pvec,pvec) \
                - jnp.einsum('ik,j...,l...->ijkl...',I_3,pvec,pvec) - jnp.einsum('jl,i...,k...->ijkl...',I_3,pvec,pvec) \
                - jnp.einsum('il,j...,k...->ijkl...',I_3,pvec,pvec) - jnp.einsum('jk,i...,l...->ijkl...',I_3,pvec,pvec) \
                + jnp.einsum('ij,k...,l...->ijkl...',I_3,pvec,pvec) + jnp.einsum('kl,i...,j...->ijkl...',I_3,pvec,pvec)
    return proj_ijkl

def single_star_and_pix_response(nvec,pvec):
    # print(np.shape(nvec))
    # pvec = hp.pix2vec(nside,ppix)
    # print(np.shape(pvec))
    ndotp = jnp.sum(nvec*pvec)
    # print('dot prod shape = ',np.shape(ndotp))
    # delta_il = np.einsum('ij,kl',np.eye(3),np.eye(3))
    n_plus_pvec = (nvec+pvec)/(1+ndotp)
    # print('nplusp shape = ',np.shape(n_plus_pvec))
    R_ikl = 0.5 * (jnp.einsum('i,k,l->ikl',n_plus_pvec,nvec,nvec) - jnp.einsum('k,il->ikl',nvec,jnp.eye(3)) )
    return R_ikl


def pairwise_single_pix_response(nvec,qvec,pvec):
    R_ikl = single_star_and_pix_response(nvec,pvec)
    R_jrs = single_star_and_pix_response(qvec,pvec)
    proj_klrs = projector(pvec)
    # print("Proj shape ",proj_klrs.shape)
    H_ij = jnp.einsum('ikl,jrs,klrs->ij',R_ikl,R_jrs,proj_klrs)
    return H_ij

def pairwise_monopole(nvec,qvec,pvec_array,nside):
    da = hp.nside2pixarea(nside)# /fourpi
    # da * np.einsum('iklp,jrsp,klrsp->ij',R_ikl,R_jrs,proj_klrs)
    f = lambda pvec: pairwise_single_pix_response(nvec=nvec,qvec=qvec,pvec=pvec)
    pairwise_antenna = vmap(f,in_axes=(0),out_axes=(-1))(pvec_array)
    # print("Pairwise antenna shape = ",pairwise_antenna.shape)
    return da*jnp.sum(pairwise_antenna,axis=-1)

def pairwise_dipole(nvec,qvec,pvec_array,nside,vvec=vvec):
    da = hp.nside2pixarea(nside)# /fourpi
    # da * np.einsum('iklp,jrsp,klrsp->ij',R_ikl,R_jrs,proj_klrs)
    f = lambda pvec: jnp.sum(vvec*pvec) * pairwise_single_pix_response(nvec=nvec,qvec=qvec,pvec=pvec)
    pairwise_antenna = vmap(f,in_axes=(0),out_axes=(-1))(pvec_array)
    # print("Pairwise antenna shape = ",pairwise_antenna.shape)
    return da*jnp.sum(pairwise_antenna,axis=-1)

def trace_pair_pair_H0(nvec,qvec,pvec_array,nside):
    monopole = pairwise_monopole(nvec,qvec,pvec_array,nside)
    #trace = jnp.trace(monopole*monopole)
    trace = jnp.trace(monopole@monopole)
    return trace

def trace_pair_pair_H1(nvec,qvec,pvec_array,nside):
    dipole = pairwise_dipole(nvec,qvec,pvec_array,nside)
    #trace = jnp.trace(dipole*dipole)
    trace = jnp.trace(dipole@dipole)
    return trace

def trace_pair_pair_H01(nvec,qvec,pvec_array,nside):
    monopole = pairwise_monopole(nvec,qvec,pvec_array,nside)
    dipole = pairwise_dipole(nvec,qvec,pvec_array,nside)
    trace = jnp.trace(dipole*monopole)
    return trace

def trace_over_pairs(nvec_array,pvec_array,nside):
    f = lambda n,q: trace_pair_pair_H0(n,q,pvec_array=pvec_array,nside=nside)
    pair_pair_matrix_0 = vmap(vmap(f,in_axes=(0,None)),in_axes=(None,0))(nvec_array,nvec_array)     # this should have n x n elements
    trace_0 = jnp.sum(pair_pair_matrix_0)
    # print(pair_pair_matrix.shape)
    f = lambda n,q: trace_pair_pair_H1(n,q,pvec_array=pvec_array,nside=nside)
    pair_pair_matrix_1 = vmap(vmap(f,in_axes=(0,None)),in_axes=(None,0))(nvec_array,nvec_array)     # this should have n x n elements
    trace_1 = jnp.sum(pair_pair_matrix_1)
    f = lambda n,q: trace_pair_pair_H01(n,q,pvec_array=pvec_array,nside=nside)
    pair_pair_matrix_01 = vmap(vmap(f,in_axes=(0,None)),in_axes=(None,0))(nvec_array,nvec_array)     # this should have n x n elements
    trace_01 = jnp.sum(pair_pair_matrix_01)
    return trace_0, trace_1, trace_01


# Pulsar timing functions

def pulsar_single_pix_response(xvec,pvec):
    xdotp = jnp.sum(xvec*pvec)
    return jnp.einsum("i,j->ij",xvec,xvec)/(1+xdotp)

def pulsar_star_single_pix(nvec,xvec,pvec):
    R_ikl = single_star_and_pix_response(nvec,pvec)
    proj_klrs = projector(pvec)
    P_rs = pulsar_single_pix_response(xvec,pvec)
    K_i = jnp.einsum("ikl,klrs,rs->i",R_ikl,proj_klrs,P_rs)
    return K_i

def pulsar_star_pair_monopole(nvec,xvec,pvec_array,nside):
    da = hp.nside2pixarea(nside)# /fourpi
    f = lambda pvec: pulsar_star_single_pix(nvec=nvec,xvec=xvec,pvec=pvec)
    pairwise_antenna = vmap(f,in_axes=(0),out_axes=(-1))(pvec_array)
    return da*jnp.sum(pairwise_antenna,axis=-1)

def pulsar_star_pair_dipole(nvec,xvec,pvec_array,nside,vvec=vvec):
    da = hp.nside2pixarea(nside)# /fourpi
    f = lambda pvec: jnp.sum(vvec*pvec) * pulsar_star_single_pix(nvec=nvec,xvec=xvec,pvec=pvec)
    pairwise_antenna = vmap(f,in_axes=(0),out_axes=(-1))(pvec_array)
    return da*jnp.sum(pairwise_antenna,axis=-1)

def Sq_pair_pair_K0(nvec,qvec,pvec_array,nside):
    monopole = pulsar_star_pair_monopole(nvec,qvec,pvec_array,nside)
    #trace = jnp.trace(monopole*monopole)
    #trace = jnp.trace(monopole@(monopole.T))
    mon_sq = monopole@(monopole.T)
    return mon_sq

def Sq_pair_pair_K1(nvec,qvec,pvec_array,nside):
    dipole = pulsar_star_pair_dipole(nvec,qvec,pvec_array,nside)
    dip_sp = jnp.sum(dipole*dipole)
    #dip_sp = (dipole.T)@(dipole)
    return dip_sp