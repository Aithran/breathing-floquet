#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import h5py
import os

from matplotlib import pyplot as plt

from zss_progbar import log_progress as progbar


# In[2]:


base_dir = os.path.join('/media', 'simulationData', 'BreathingLattice', 'floquet')


# In[3]:


def gen_energy_plot(ax, x_array, y_matrix, e_min, e_max, modulus=None, **kwargs):
    """ Plot into ax points from rows of y_matrix whose x-coordinate comes from matching x_array, e_min<y<e_max
       
    """
    for idx, this_x in enumerate(x_array):
        if modulus is None:
            y_row = y_matrix[idx]
        elif hasattr(modulus, "__len__"):
            y_row = np.mod(y_matrix[idx] + modulus[idx]/2, modulus[idx]) - modulus[idx]/2
        else:
            y_row = np.mod(y_matrix[idx] + modulus/2, modulus) - modulus/2
        y_in_range = y_row[np.logical_and(y_row >= e_min, y_row <= e_max)]
        ax.plot(np.repeat(this_x, len(y_in_range)), y_in_range, **kwargs)
    ax.set_xlim(2*x_array[0] - x_array[1], 2*x_array[-1] - x_array[-2])
    ax.set_ylim(e_min, e_max)


# In[4]:


get_ipython().run_cell_magic('time', '', 'file_stem = \'gamma_scan\'\nwith h5py.File(os.path.join(base_dir, file_stem, file_stem + \'_out.h5\'), \'r\', libver="latest") as f:\n    fig = plt.figure(figsize=(8,10.5))\n    ax = fig.add_subplot(111)\n    gen_energy_plot(ax, f[\'scan_values\'][:], f[\'floquet_energy\'][:], -100, 100, marker=\',\', ls=\'None\', c=\'k\', modulus=1000.)\n    fig.tight_layout()')


# In[5]:


def wrap_spectrum(energy_list, mod):
    half = 0.5*mod
    return np.mod(energy_list + half, mod) - half


# In[6]:


def unfolding(energy_array, idx=0, deg=5, mod=None):
    this_energy = energy_array[idx][0::2]
    if mod is not None:
        this_energy = wrap_spectrum(this_energy, mod)
    energy_len = len(this_energy)
    sorted_energy = np.sort(this_energy)
    return np.array((sorted_energy, np.arange(energy_len))), np.polynomial.polynomial.Polynomial.fit(sorted_energy, np.arange(energy_len)+0.5, deg)


# In[7]:


def unfold_from_file(file_stem, which_idx):
    with h5py.File(os.path.join(base_dir, file_stem, file_stem + '_out.h5'), 'r', libver="latest") as f:
        #unfolded = unfolding(f['floquet_energy'][:], idx=which_idx, deg=9, mod=f['scan_values'][which_idx])
        unfolded = unfolding(f['floquet_energy'][:], idx=which_idx, deg=9, mod=1000.0)
        unfolded_x = unfolded[0][0]
        unfolded_y = unfolded[1](unfolded_x)
        return unfolded_y


# In[8]:


def spacing_goe(s_vals):
    return 0.5*np.pi*s_vals*np.exp(-0.25*np.pi*np.square(s_vals))
def spacing_gue(s_vals):
    return 32.0/(np.pi*np.pi)*np.square(s_vals)*np.exp(-4.0/np.pi*np.square(s_vals))
def spacing_gse(s_vals):
    scaled_s = 8.0/(3.0*np.pi)*s_vals
    return 64.0/9.0*np.pi*np.power(scaled_s, 4)*np.exp(-np.pi*np.square(scaled_s))
def spacing_poisson(s_vals):
    return np.exp(-s_vals)
def plus_poisson(s_vals, prefactor, n_sigma=1):
    poisson = prefactor * spacing_poisson(s_vals)
    return poisson + n_sigma*np.sqrt(poisson)
def minus_poisson(s_vals, prefactor, n_sigma=1):
    poisson = prefactor * spacing_poisson(s_vals)
    return poisson - n_sigma*np.sqrt(poisson)


# In[21]:


unfolded_y = unfold_from_file('gamma_scan', -1)
smin, smax = 0.0, 4.0
n_bins = 100
amp_factor = (len(unfolded_y)-1)*(smax-smin)/n_bins
s_coords = np.linspace(smin, smax, 501, endpoint=True)
plt.hist(unfolded_y[1:] - unfolded_y[:-1], bins=n_bins, range=(smin, smax), color='grey')
plt.plot(s_coords, amp_factor*spacing_goe(s_coords), 'b', label="GOE/COE")
plt.plot(s_coords, amp_factor*spacing_gue(s_coords), 'g', label="GUE/CUE")
#plt.plot(s_coords, amp_factor*spacing_gse(s_coords), label="GSE/CSE")
plt.plot(s_coords, amp_factor*spacing_poisson(s_coords), 'k', label="Poisson")
plt.plot(s_coords, plus_poisson(s_coords, amp_factor), '--k', label="$\pm\sigma$")
plt.plot(s_coords, minus_poisson(s_coords, amp_factor), '--k')
plt.plot(s_coords, plus_poisson(s_coords, amp_factor,2), ':k', label="$\pm2\sigma$")
plt.plot(s_coords, minus_poisson(s_coords, amp_factor,2), ':k')
plt.legend()
plt.xlim(smin,smax)
plt.ylim(0, plus_poisson(0.0, amp_factor,2.5))
plt.savefig('/home/zachsmith/Desktop/first_spacing.png', dpi=300)


# In[10]:


plt.plot(unfolded_x, raw_y, label='raw')
plt.plot(unfolded_x, unfolded_y, label='unfolded')
plt.legend()


# In[11]:


np.sum(spacing_gse(s_coords))*10./500.


# In[12]:


256*9


# In[13]:


get_ipython().run_cell_magic('time', '', "all_unfoldings = np.empty((901, 1152))\nfor this_idx in np.arange(901):\n    all_unfoldings[this_idx] = unfold_from_file('gamma_scan', this_idx)\nall_spacings = all_unfoldings[:,1:] - all_unfoldings[:,:-1]")


# In[14]:


spacing_hist = np.empty((901, 100))
for this_idx in np.arange(901):
    spacing_hist[this_idx], spacing_edges = np.histogram(all_spacings[this_idx], bins=100, range=(0.0, 4.0))
spacing_centers = (spacing_edges[1:] + spacing_edges[:-1])/2.0


# In[15]:


plt.pcolormesh(spacing_hist, cmap='inferno',vmax=60)
plt.colorbar()
plt.savefig('/home/zachsmith/Desktop/all_spacings.png', dpi=300)


# In[16]:


plt.plot(spacing_hist[:,0])
plt.plot(spacing_hist[:,1])
plt.plot(spacing_hist[:,10])


# In[30]:


import matplotlib
font = {'size'   : 22}
matplotlib.rc('font', **font)
outdir = '/home/zachsmith/Desktop/groupPlots/'
fig = plt.figure(figsize=(15,8))

ax = fig.add_subplot(121)
unfolded_y = unfold_from_file('gamma_scan', -1)
smin, smax = 0.0, 4.0
n_bins = 100
amp_factor = (len(unfolded_y)-1)*(smax-smin)/n_bins
s_coords = np.linspace(smin, smax, 501, endpoint=True)
ax.hist(unfolded_y[1:] - unfolded_y[:-1], bins=n_bins, range=(smin, smax), color='grey')
ax.plot(s_coords, amp_factor*spacing_goe(s_coords), 'b', label="GOE/COE")
ax.plot(s_coords, amp_factor*spacing_gue(s_coords), 'g', label="GUE/CUE")
#plt.plot(s_coords, amp_factor*spacing_gse(s_coords), label="GSE/CSE")
ax.plot(s_coords, amp_factor*spacing_poisson(s_coords), 'k', label="Poisson")
ax.plot(s_coords, plus_poisson(s_coords, amp_factor), '--k', label="$\pm\sigma$")
ax.plot(s_coords, minus_poisson(s_coords, amp_factor), '--k')
ax.plot(s_coords, plus_poisson(s_coords, amp_factor,2), ':k', label="$\pm2\sigma$")
ax.plot(s_coords, minus_poisson(s_coords, amp_factor,2), ':k')
ax.legend()
ax.set_xlim(smin,smax)
ax.set_ylim(0, plus_poisson(0.0, amp_factor,2.5))
ax.set_xlabel('s')
ax.set_ylabel('N')

ax = fig.add_subplot(122)
quad = ax.pcolormesh(spacing_hist, cmap='inferno',vmax=60)
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
plt.colorbar(quad, ax=ax)

fig.tight_layout()
fig.savefig(outdir + 'our_spacings.png', dpi=150)


# In[33]:


import matplotlib
font = {'size'   : 22}
matplotlib.rc('font', **font)
outdir = '/home/zachsmith/Desktop/groupPlots/'
fig = plt.figure(figsize=(8,8))

ax = fig.add_subplot(111)
unfolded_y = unfold_from_file('gamma_scan', -1)
smin, smax = 0.0, 4.0
n_bins = 100
amp_factor = (len(unfolded_y)-1)*(smax-smin)/n_bins
s_coords = np.linspace(smin, smax, 501, endpoint=True)
#ax.hist(unfolded_y[1:] - unfolded_y[:-1], bins=n_bins, range=(smin, smax), color='grey')
ax.plot(s_coords, amp_factor*spacing_goe(s_coords), 'b', label="GOE/COE")
ax.plot(s_coords, amp_factor*spacing_gue(s_coords), 'g', label="GUE/CUE")
#plt.plot(s_coords, amp_factor*spacing_gse(s_coords), label="GSE/CSE")
ax.plot(s_coords, amp_factor*spacing_poisson(s_coords), 'k', label="Poisson")
#ax.plot(s_coords, plus_poisson(s_coords, amp_factor), '--k', label="$\pm\sigma$")
#ax.plot(s_coords, minus_poisson(s_coords, amp_factor), '--k')
#ax.plot(s_coords, plus_poisson(s_coords, amp_factor,2), ':k', label="$\pm2\sigma$")
#ax.plot(s_coords, minus_poisson(s_coords, amp_factor,2), ':k')
ax.legend()
ax.set_xlim(smin,smax)
ax.set_ylim(0, plus_poisson(0.0, amp_factor,2.5))
ax.set_xlabel('s')
ax.set_ylabel('N')

#ax = fig.add_subplot(122)
#quad = ax.pcolormesh(spacing_hist, cmap='inferno',vmax=60)
#ax.get_xaxis().set_ticks([])
#ax.get_yaxis().set_ticks([])
#plt.colorbar(quad, ax=ax)

fig.tight_layout()
fig.savefig(outdir + 'blank_spacings.png', dpi=150)


# In[ ]:




