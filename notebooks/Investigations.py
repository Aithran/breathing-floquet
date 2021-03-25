#!/usr/bin/env python
# coding: utf-8

# In[292]:


import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import h5py
import os

import breathing_critical_floquet as critical


# In[244]:


n_sites = 17
n_floq = 4


# In[293]:


font = {'size'   : 22}
matplotlib.rc('font', **font)
outdir = '/home/zachsmith/Desktop/groupPlots/'


# In[245]:


small_jsq_op = critical.on_site_jsq(n_sites)
fig = plt.figure()
ax = fig.add_subplot(111,aspect='equal')
mesh = ax.pcolormesh(small_jsq_op, cmap='inferno')
plt.colorbar(mesh)


# In[246]:


full_panel = critical.build_floquet_jsq(small_jsq_op, n_floq, np.array([1, 0.75, 0.5, 0.25, 0.125, 0, 1]))
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
mesh = ax.pcolormesh(full_panel, cmap='inferno')
plt.colorbar(mesh)


# In[247]:


hopping_op = critical.tunneling_block(n_sites, -1)
fig = plt.figure()
ax = fig.add_subplot(111,aspect='equal')
mesh = ax.pcolormesh(hopping_op, cmap='inferno')
plt.colorbar(mesh)


# In[248]:


blank = np.zeros_like(full_panel)
full_tunneling = critical.add_tunneling_blocks(blank, n_floq, hopping_op, [1, -1, 0.5])
fig = plt.figure()
ax = fig.add_subplot(111,aspect='equal')
mesh = ax.pcolormesh(full_tunneling, cmap='RdBu')
plt.colorbar(mesh)


# In[249]:


diag = critical.add_floquet_diag(blank, n_floq, 0.5)
fig = plt.figure()
ax = fig.add_subplot(111,aspect='equal')
mesh = ax.pcolormesh(diag, cmap='RdBu')
plt.colorbar(mesh)


# In[270]:


# Compute needed coefficients
floquet_m_radius = 3
n_floquet = 2*floquet_m_radius + 1
site_spacing_um = 1.3
trap_freq = 20
gamma = 0.01
####
n_floquet = 2*floquet_m_radius + 1
lattice_tunneling = critical.tunneling_J(1.5e3, critical.lattice_recoil_energy(site_spacing_um))
print(lattice_tunneling)
jsq_scale = critical.gen_jsq_coeffs(site_spacing_um, trap_freq, gamma, n_floquet)
h0_scale = critical.dc.all_chis(critical.dc.y(gamma))
crit_drive = critical.dc.crit_drive_freq(critical.dc.y(gamma), trap_freq)
print(crit_drive)
# Build up floquet problem
jsq_block = critical.on_site_jsq(n_sites)
tun_block = critical.tunneling_block(n_sites, lattice_tunneling)
floquet_h = critical.build_floquet_jsq(jsq_block, floquet_m_radius, jsq_scale)
floquet_jsq = floquet_h.copy()
floquet_h = critical.add_tunneling_blocks(floquet_h, floquet_m_radius, tun_block, h0_scale)
floquet_tun = floquet_h.copy()
floquet_h = critical.add_floquet_diag(floquet_h, floquet_m_radius, crit_drive)


# In[257]:


#plt.semilogy(np.abs(floquet_h[::n_sites, floquet_m_radius*n_sites])/lattice_tunneling, '.')
#plt.semilogy(np.abs(floquet_h[n_sites-1::n_sites,(floquet_m_radius+1)*n_sites-1])/lattice_tunneling, '.')
plt.semilogy(np.abs(floquet_h[:, floquet_m_radius*n_sites+n_sites//2])/lattice_tunneling, '.')
plt.semilogy(np.abs(floquet_h[:, floquet_m_radius*n_sites])/lattice_tunneling, '.')
plt.semilogy(np.abs(floquet_h[:,(floquet_m_radius+1)*n_sites-1])/lattice_tunneling, '.')

plt.axhline(1, ls=':', c='k')


# In[258]:


jsq_scale


# In[259]:


plt.pcolormesh(floquet_jsq)
plt.colorbar()


# In[281]:


def plot_flqouet_matrix(data, ax=None, max_abs=None, m_grid=True, draw_cbar=True, **kwargs):
    if ax is None:
        ax = plt.figure().add_subplot(111, aspect='equal')
    if max_abs is None:
        max_abs = np.abs(data).max()
    default_kwargs = {'cmap':"RdBu", 'vmax':max_abs, 'vmin':-max_abs}
    default_kwargs.update(kwargs)
    quad = plt.pcolormesh(np.flipud(data), **default_kwargs)
    if m_grid:
        for m in np.arange(2*floquet_m_radius+2):
            ax.axvline(m*n_sites, ls=':', c='grey')
            ax.axhline(m*n_sites, ls=':', c='grey')
    ax.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,
    left=False, right=False, labelleft=False, # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
    if draw_cbar:
        plt.colorbar(quad, ax=ax)


# In[261]:


get_ipython().run_cell_magic('time', '', 'maxmag = np.abs(floquet_tun).max()\nmaxmag = 20\nax = plt.figure().add_subplot(111, aspect=\'equal\')\nplt.pcolormesh(floquet_tun, cmap="RdBu", vmax=maxmag, vmin=-maxmag)\nfor m in np.arange(2*floquet_m_radius+2):\n    ax.axvline(m*n_sites, ls=\':\', c=\'k\')\n    ax.axhline(m*n_sites, ls=\':\', c=\'k\')\nplt.colorbar()')


# In[262]:


get_ipython().run_cell_magic('time', '', "plot_flqouet_matrix(floquet_h[3*n_sites:4*n_sites:,::], max_abs=1, m_grid=False)\nplt.savefig('/home/zachsmith/Desktop/testOut.png', figsize=(8,8), dpi=300)")


# In[263]:


maxmag = np.abs(floquet_tun).max()
#fig = plt.figure(aspe)
plt.pcolormesh(floquet_h, cmap="RdBu", vmax=maxmag, vmin=-maxmag)
plt.colorbar()


# In[264]:


critical.dc.keff_factor(0.9995, cancel_keff=False)


# In[265]:


floquet_radius =3
np.abs(np.arange(-floquet_radius, floquet_radius + 1)[:, np.newaxis]
                     - np.arange(-floquet_radius, floquet_radius + 1)[np.newaxis, :])


# In[294]:


fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(221, aspect='equal')
plot_flqouet_matrix(floquet_jsq, ax=ax, max_abs=None, m_grid=True, draw_cbar=False)

ax = fig.add_subplot(222, aspect='equal')
plot_flqouet_matrix(floquet_tun-floquet_jsq, ax=ax, max_abs=None, m_grid=True, draw_cbar=False)

ax = fig.add_subplot(223, aspect='equal')
plot_flqouet_matrix(floquet_h - floquet_tun, ax=ax, max_abs=None, m_grid=True, draw_cbar=False)

ax = fig.add_subplot(224, aspect='equal')
plot_flqouet_matrix(floquet_h, ax=ax, max_abs=None, m_grid=True, draw_cbar=False)
fig.tight_layout()
fig.tight_layout()
fig.savefig(outdir + 'hfloquet_panels.png', dpi=150)


# In[295]:


from matplotlib import colors
symlog = colors.SymLogNorm(linthresh=0.03, linscale=0.01, vmin=-10.0, vmax=10.0)
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(221, aspect='equal')
plot_flqouet_matrix(floquet_jsq, ax=ax, max_abs=None, m_grid=True, draw_cbar=False, norm=symlog)

ax = fig.add_subplot(222, aspect='equal')
plot_flqouet_matrix(floquet_tun-floquet_jsq, ax=ax, max_abs=None, m_grid=True, draw_cbar=False, norm=symlog)

ax = fig.add_subplot(223, aspect='equal')
plot_flqouet_matrix(floquet_h - floquet_tun, ax=ax, max_abs=None, m_grid=True, draw_cbar=False, norm=symlog)

ax = fig.add_subplot(224, aspect='equal')
plot_flqouet_matrix(floquet_h, ax=ax, max_abs=None, m_grid=True, draw_cbar=False, norm=symlog)
fig.tight_layout()
fig.savefig(outdir + 'hfloquet_panels_log.png', dpi=150)


# In[297]:


plot_data = [floquet_jsq, floquet_tun-floquet_jsq, floquet_h - floquet_tun, floquet_h]
filestem = ['jsq', 'tunneling', 'diag', 'total']
for idx, data in enumerate(plot_data):
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    plot_flqouet_matrix(data, ax=ax, max_abs=None, m_grid=True, draw_cbar=True)
    fig.tight_layout()
    fig.savefig(outdir + filestem[idx] + '_zoom.png', dpi=150)


# In[298]:


plot_data = [floquet_jsq, floquet_tun-floquet_jsq, floquet_h - floquet_tun, floquet_h]
filestem = ['jsq', 'tunneling', 'diag', 'total']
for idx, data in enumerate(plot_data):
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    plot_flqouet_matrix(data, ax=ax, max_abs=None, m_grid=True, draw_cbar=True, norm=symlog)
    fig.tight_layout()
    fig.savefig(outdir + filestem[idx] + '_zoom_log.png', dpi=150)


# In[ ]:




