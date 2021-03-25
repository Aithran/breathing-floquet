#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
import h5py
import os
import glob
from zss_progbar import log_progress as progbar
import re
import time


# In[2]:


m_re = re.compile(r'.*_m(\d+)_out.h5')
def get_m(file_name):
    return int(m_re.search(file_name).group(1))
base_path = '/home/zachsmith/gitlab/breathing-floquet/notebooks/data/mscan/'
data_dirs = [os.path.basename(dirpath[:-1]) for dirpath in glob.glob(base_path + "direct_gamma*/")]
data_paths = sorted([os.path.join(base_path, this_dir, this_dir + '_out.h5') for this_dir in data_dirs], key=get_m)
for idx, path in enumerate(data_paths):
    print('{}:\t{}'.format(idx, os.path.basename(path)))
largest_m_path = data_paths[-1]
print(largest_m_path)


# In[3]:


from cycler import cycler
kelly_cycler = (cycler(color=['#FFB300', '#803E75', '#FF6800',
              '#A6BDD7', '#C10020', '#CEA262',
              '#817066', '#007D34']) + cycler(linestyle=['-', '--', ':', '-.', '-', '--', ':', '-.']))


# In[4]:


get_ipython().run_cell_magic('time', '', 'n_sites = 129\nn_time_samples = 1024\nmin_samples = 10 # Warn if there\'s a component who won\'t be sampled faster than this\ntwo_pi_i = np.complex(0, np.pi*2.0)\ntry:\n    len(first_set_residuals_dict)\nexcept NameError:\n    first_set_residuals_dict = dict()\ntry:\n    len(last_set_residuals_dict)\nexcept NameError:\n    last_set_residuals_dict = dict()\nskip_if_present = True\nfor this_path in progbar(data_paths):\n    with h5py.File(this_path, \'r\', libver="latest") as f:\n        print(os.path.basename(this_path))\n        states_matrix = f[\'floquet_state\']\n        first_states = states_matrix[0]\n        last_states = states_matrix[-1]\n        n_states = states_matrix.shape[-1]\n        first_resid_pops = np.empty((n_states, n_time_samples))\n        last_resid_pops = np.empty((n_states, n_time_samples))\n        m_rad = ((states_matrix.shape[-1]//n_sites) - 1)//2\n        if skip_if_present and (m_rad in first_set_residuals_dict.keys()):\n            continue\n        if (n_time_samples/(2*m_rad) < min_samples):\n            print(\'Minimum sample violation {} < {}\'.format(n_time_samples/(2*m_rad), min_samples))\n        phasor = np.zeros((n_sites, (2*m_rad+1)*n_sites), dtype=np.complex)\n        where_mat = np.tile(np.diag(np.full(n_sites, True)), 2*m_rad+1)\n        m_block = np.kron(np.arange(-m_rad, m_rad+1), np.identity(n_sites)).astype(np.complex)\n        for idx, t in progbar(enumerate(np.linspace(0.0, 1.0, n_time_samples)), size=n_time_samples):\n            np.exp(-two_pi_i * m_block * t, out=phasor, where=where_mat)\n            now_amps = np.matmul(phasor, first_states)\n            first_resid_pops[:, idx] = 1.0 - np.sum(np.abs(np.conjugate(now_amps) * now_amps), axis=0)\n            now_amps = np.matmul(phasor, last_states)\n            last_resid_pops[:, idx] = 1.0 - np.sum(np.abs(np.conjugate(now_amps) * now_amps), axis=0)\n        first_set_residuals_dict[m_rad] = first_resid_pops.copy()\n        last_set_residuals_dict[m_rad] = last_resid_pops.copy()\ndel now_amps, phasor, where_mat, first_states, last_states, m_block, first_resid_pops, last_resid_pops')


# In[5]:


ax = plt.subplot(111)
ax.set_prop_cycle(kelly_cycler)
for m_count, resids in first_set_residuals_dict.items():
    plt.plot(np.sort(np.sum(np.abs(resids), axis=1))/n_time_samples, label='{}-'.format(m_count))
for m_count, resids in last_set_residuals_dict.items():
    plt.plot(np.sort(np.sum(np.abs(resids), axis=1))/n_time_samples, label='{}+'.format(m_count))
plt.ylim(0,1)
plt.legend()


# In[6]:


colors = kelly_cycler.by_key()['color']
for idx, m_count in enumerate(np.sort(list(first_set_residuals_dict.keys()))[-len(colors):]):
    resids = first_set_residuals_dict[m_count]
    plt.plot(np.sort(np.sum(np.abs(resids), axis=1))/n_time_samples, label=m_count, c=colors[idx])
    resids = last_set_residuals_dict[m_count]
    plt.plot(np.sort(np.sum(np.abs(resids), axis=1))/n_time_samples, c=colors[idx], ls='--')
plt.legend()
plt.xlim(0,n_sites)
plt.ylim(0,0.01)


# In[7]:


colors = kelly_cycler.by_key()['color']
for idx, m_count in enumerate(np.sort(list(first_set_residuals_dict.keys()))[-len(colors):]):
    resids = first_set_residuals_dict[m_count]
    plt.semilogy(np.sort(np.sum(np.abs(resids), axis=1))/n_time_samples, label=m_count, c=colors[idx])
    resids = last_set_residuals_dict[m_count]
    plt.semilogy(np.sort(np.sum(np.abs(resids), axis=1))/n_time_samples, c=colors[idx], ls='--')
plt.legend()
plt.xlim(0,3*n_sites)
plt.ylim(1e-9,1e-2)


# In[8]:


m_rad = 16
plt.semilogy(np.sum(np.abs(last_set_residuals_dict[m_rad]), axis=1)/n_time_samples, '.')
plt.semilogy(np.sum(np.abs(first_set_residuals_dict[m_rad]), axis=1)/n_time_samples, '.')
for i in np.arange(2*(m_rad+1)):
    plt.axvline(i*n_sites, c='k', ls=':')
plt.axvline(m_rad*n_sites, c='k', ls='--')
plt.axvline((m_rad+1)*n_sites, c='k', ls='--')
plt.xlim(0,(2*m_rad+1)*n_sites)


# In[9]:


m_out = 8
plt.semilogy(np.sum(np.abs(last_set_residuals_dict[m_rad]), axis=1)/n_time_samples, '.')
plt.semilogy(np.sum(np.abs(first_set_residuals_dict[m_rad]), axis=1)/n_time_samples, '.')
plt.ylim(None,0.01)
for i in np.arange(2*(m_rad+1)):
    plt.axvline(i*n_sites, c='k', ls=':')
plt.axvline(m_rad*n_sites, c='k', ls='--')
plt.axvline((m_rad+1)*n_sites, c='k', ls='--')
plt.xlim((m_rad-m_out)*n_sites,(m_rad+1+m_out)*n_sites)


# In[10]:


## Old matrix-based method
#%%time
#n_sites = 129
#n_sets = 256
#set_stride = 64
#n_time_samples = 1024
#min_samples = 10 # used to compare n_time_samples to fastest oscillation timescale...
#two_pi_i = np.complex(0, np.pi*2.0)
# Results, 2D, one set per column
#this_path = largest_m_path
#
# Check quasienergy vs. residual, make sure we have enough unique states
#with h5py.File(this_path, 'r', libver="latest") as f:
#    print(os.path.basename(this_path))
#    states_matrix = f['floquet_state']
#    m_rad = ((states_matrix.shape[-1]//n_sites) - 1)//2
#    n_states = states_matrix.shape[-1]
#    all_raw_residuals = np.empty((n_sets//set_stride, n_states))
#    resid_pops = np.empty((n_states, n_time_samples))
#    phasor = np.zeros((n_sites, (2*m_rad+1)*n_sites), dtype=np.complex)
#    where_mat = np.tile(np.diag(np.full(n_sites, True)), 2*m_rad+1)
#    m_block = np.kron(np.arange(-m_rad, m_rad+1), np.diag(np.full(n_sites, -two_pi_i))).astype(np.complex)
#    site_amps = np.empty((n_sites, n_states), dtype=np.complex)
#    #one_site_m_block = (-two_pi_i * np.arange(-m_rad, m_rad+1)).reshape((1,-1,1))
#    #one_site_phasor = np.empty_like(one_site_m_block)
#    for out_idx, this_set in progbar(enumerate(np.arange(0, n_sets, set_stride)), every=1, size=n_sets//set_stride):
#        these_states = states_matrix[this_set,:]
#        for idx, t in enumerate(np.linspace(0.0, 1.0, n_time_samples)):
#            np.exp(m_block * t, out=phasor, where=where_mat)
#            #np.exp(one_site_m_block * t, out=one_site_phasor)
#            np.matmul(phasor, these_states, out=site_amps)
#            #%time np.sum(one_site_phasor * these_states.reshape(n_sites, -1, n_states), axis=1, out=site_amps)
#            resid_pops[:, idx] = 1.0-np.sum(np.real(np.conjugate(site_amps) * site_amps), axis=0)
#        all_raw_residuals[out_idx] = np.sum(np.abs(resid_pops), axis=1)/n_time_samples


# In[10]:


get_ipython().run_cell_magic('time', '', 'n_sites = 129\nn_sets = 256\nset_stride = 2\nn_time_samples = 1024\nmin_samples = 10 # used to compare n_time_samples to fastest oscillation timescale...\ntwo_pi_i = np.complex(0, np.pi*2.0)\n# Results, 2D, one set per column\nthis_path = largest_m_path\n# Check quasienergy vs. residual, make sure we have enough unique states\nwith h5py.File(this_path, \'r\', libver="latest") as f:\n    print(os.path.basename(this_path))\n    states_matrix = f[\'floquet_state\']\n    m_rad = ((states_matrix.shape[-1]//n_sites) - 1)//2\n    n_states = states_matrix.shape[-1]\n    all_raw_residuals = np.empty((n_sets//set_stride, n_states))\n    resid_pops = np.empty((n_states, n_time_samples))\n    m_block = (np.arange(-m_rad, m_rad+1) * -two_pi_i).reshape(-1,1,1)\n    site_amps = np.empty((n_sites, n_states), dtype=np.complex)\n    expand_states = np.empty((2*m_rad + 1, n_sites, n_states), dtype=np.complex)\n    for out_idx, this_set in progbar(enumerate(np.arange(0, n_sets, set_stride)), every=1, size=n_sets//set_stride):\n        these_states = states_matrix[this_set,:].reshape(2*m_rad + 1, n_sites, -1)\n        for idx, t in enumerate(np.linspace(0.0, 1.0, n_time_samples)):\n            phasor = np.exp(m_block * t)\n            np.multiply(phasor, these_states, out=expand_states)\n            np.sum(expand_states, axis=0, out=site_amps)\n            resid_pops[:, idx] = 1.0-np.sum(np.real(np.conjugate(site_amps) * site_amps), axis=0)\n        all_raw_residuals[out_idx] = np.sum(np.abs(resid_pops), axis=1)/n_time_samples\ndel expand_states, phasor, site_amps, resid_pops, m_block')


# In[11]:


#raise RuntimeException("Don't overwrite things yet")
result_file_path = os.path.join(base_path, 'residual_pops_gamma.h5')
print(result_file_path)
with h5py.File(result_file_path, 'w', libver="latest") as sf:
    full_scan = sf.create_dataset('total_residuals', data=all_raw_residuals)
    full_scan.attrs['m_rad'] = m_rad
    full_scan.attrs['source_path'] = str(largest_m_path)
    full_scan.attrs['n_time_samples'] = n_time_samples
    grp = sf.create_group('first_scan_residuals')
    grp.attrs['n_time_samples'] = n_time_samples
    grp.attrs['n_sites'] = n_sites
    for m_index, residuals in first_set_residuals_dict.items():
        dset_name = 'first_scan_residuals/m{:d}'.format(m_index)
        grp.create_dataset('m{:d}'.format(m_index), data=residuals)
        sf[dset_name].attrs['source_path'] = next((s for s in data_paths if 'm{:d}_out'.format(m_index) in s), "Not Found")
        #print(m_index, sf[dset_name].name, sf[dset_name].attrs['source_path'])
    grpl = sf.create_group('last_scan_residuals')
    grpl.attrs['n_time_samples'] = n_time_samples
    grpl.attrs['n_sites'] = n_sites
    for m_index, residuals in last_set_residuals_dict.items():
        dset_name = 'last_scan_residuals/m{:d}'.format(m_index)
        grpl.create_dataset('m{:d}'.format(m_index), data=residuals)
        sf[dset_name].attrs['source_path'] = next((s for s in data_paths if 'm{:d}_out'.format(m_index) in s), "Not Found")
print('{} MB'.format(os.path.getsize(result_file_path)/(1024.*1024.)))


# In[28]:


clear_first_last = True
if clear_first_last:
    last_set_residuals_dict.clear()
    first_set_residuals_dict.clear()
    del last_set_residuals_dict, first_set_residuals_dict, residuals


# In[29]:


with h5py.File(largest_m_path, 'r', libver="latest") as f:
    gamma_list = f['scan_values'][:]
thresh_list = np.power(10., np.arange(-6,-1))
count_below_thresh = np.empty((len(thresh_list), all_raw_residuals.shape[0]))
for idx, this_thresh in progbar(enumerate(thresh_list), every=1):
    count_below_thresh[idx] = np.sum(all_raw_residuals < this_thresh, axis=1)


# In[13]:


for idx, this_thresh in enumerate(thresh_list):
    plt.plot(gamma_list[::set_stride], count_below_thresh[idx], '.', label=this_thresh)
plt.ylim(0,2.5*n_sites)
plt.axhline( n_sites, c='k', ls=':')
plt.axhline(2*n_sites, c='k', ls=':')
#plt.legend(thresh_list)
plt.legend()


# In[14]:


resid_file_path = os.path.join(base_path, 'gamma_m12_all_residuals.h5')
print(resid_file_path)
#raise RuntimeException("Don't overwrite things yet")
with h5py.File(resid_file_path, 'w', libver="latest") as sf:
    full_scan = sf.create_dataset('total_residuals', data=all_raw_residuals)
    full_scan.attrs['m_rad'] = m_rad
    full_scan.attrs['source_path'] = str(this_path)
    full_scan.attrs['n_time_samples'] = n_time_samples


# In[15]:


all_raw_residuals.shape


# In[16]:


plt.plot(all_raw_residuals.T, ',k')
plt.xlim(0,4515)


# In[17]:


with h5py.File(this_path, 'r', libver='latest') as f:
    print(os.path.basename(this_path))
    matching_quasi = f['floquet_energy'][::set_stride]
    matching_drive = 1e3
    matching_gamma = f['scan_values'][::set_stride]


# In[18]:


print(all_raw_residuals.shape,
     #matching_drive.shape,
     matching_gamma.shape,
     matching_quasi.shape)


# In[19]:


fig = plt.figure(figsize=(8,6))
ax = plt.subplot(111)
for idx, gamma in enumerate(matching_gamma):
    stable_states = np.where(all_raw_residuals[idx] < 0.005)
    stable_quasi = matching_quasi[idx, stable_states].flatten()
    fbz_quasi = np.mod(stable_quasi + 0.5*matching_drive, matching_drive) - 0.5*matching_drive
    ax.plot(np.full(len(fbz_quasi), gamma), fbz_quasi, ',k')
#ax.plot(matching_gamma,  0.5*matching_drive, ':r')
#ax.plot(matching_gamma, -0.5*matching_drive, ':r')
plt.ylim(-500, 500)
plt.xlim(0,.01)


# # Shift matrix to weed out similar physical states?
# The plan:
# For a state |n> with quasi-energy e_nm
# Find FBZ quasienergy e_n0 via e_n0 = mod(e_nm + w/2) - w/2,
# Find also the shift amount m via (e_nm + w/2)//w
# Compute |n'> by doing S_(m*n_sites) |n>
# Once these have been built, do <l|n> and see what states overlap significantly

# In[20]:


def shift_by_m_blocks(m_shift, n_sites, n_states):
    return np.eye( n_states, k=m_shift*n_sites)


# In[21]:


#%%time
#shifted_scans = dict()
#with h5py.File(this_path, 'r', libver="latest") as f:
#    print(os.path.basename(this_path))
#    states_matrix = f['floquet_state']
#    for out_idx, this_set in progbar(enumerate(np.arange(0, n_sets, set_stride)),
#                                     every=1, size=n_sets//set_stride):
#        which_states = np.where(all_raw_residuals[out_idx] < 0.01)[0]
#        stable_states = states_matrix[this_set,:,which_states]
#        stable_quasi  = f['floquet_energy'][this_set, which_states]
#        this_drive = f['scan_values'][this_set]
#        shifted_states = np.empty_like(stable_states)
#        mshift, fqe = np.divmod(stable_quasi + 0.5*this_drive, this_drive)
#        mshift = mshift.astype(np.int)
#        fqe -= 0.5*this_drive
#        for idx in progbar(np.arange(len(which_states))):
#            shifted_states[:, idx] = np.matmul(shift_by_m_blocks(mshift[idx], n_sites, n_states),
#                                              stable_states[:,idx])
#        shifted_scans[out_idx] = {'fqe':fqe.copy(),
#                                  'mshift':mshift.copy(),
#                                  'shifted_states':shifted_states.copy()}
#        #break


# In[22]:


get_ipython().run_cell_magic('time', '', 'shifted_scans = dict()\nwith h5py.File(this_path, \'r\', libver="latest") as f:\n    print(os.path.basename(this_path))\n    states_matrix = f[\'floquet_state\']\n    for out_idx, this_set in progbar(enumerate(np.arange(0, n_sets, set_stride)),\n                                     size=n_sets//set_stride):\n        which_states = np.where(all_raw_residuals[out_idx] < 0.001)[0]\n        stable_states = states_matrix[this_set,:,which_states]\n        stable_quasi  = f[\'floquet_energy\'][this_set, which_states]\n        #this_drive = f[\'scan_values\'][this_set]\n        this_drive = 1e3\n        shifted_states = np.zeros_like(stable_states)\n        mshift, fqe = np.divmod(stable_quasi + 0.5*this_drive, this_drive)\n        mshift = (mshift).astype(np.int)\n        fqe -= 0.5*this_drive\n        zero_counter = 0\n        for idx in np.arange(len(which_states)):\n            if np.abs(mshift[idx]) > 2*m_rad:\n                zero_counter += 1\n                continue\n            dest_imin = 0 if mshift[idx] >= 0 else -mshift[idx] * n_sites\n            dest_imax = n_states if mshift[idx] <= 0 else n_states - (mshift[idx] * n_sites)\n            unshifted_imin = 0 if mshift[idx] <= 0 else mshift[idx] * n_sites \n            unshifted_imax = n_states if mshift[idx] >= 0 else n_states + (mshift[idx] * n_sites)\n            shifted_states[dest_imin:dest_imax, idx] = stable_states[unshifted_imin:unshifted_imax, idx]\n        shifted_scans[out_idx] = {\'fqe\':fqe.copy(),\n                                  \'mshift\':mshift.copy(),\n                                  \'shifted_states\':shifted_states.copy(),\n                                  \'zero_states\':zero_counter}\ndel stable_states')


# In[47]:


which_scan = 127
fqe = shifted_scans[which_scan]['fqe']
mshift = shifted_scans[which_scan]['mshift']
shifted_states = shifted_scans[which_scan]['shifted_states']


# In[48]:


for this_idx in list(shifted_scans.keys()):
    plt.plot(this_idx, shifted_scans[this_idx]['zero_states'],'.k')


# In[49]:


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(221)
ax.plot(fqe, '.')
ax = fig.add_subplot(223)
ax.plot(np.sort(fqe), '.')
ax = fig.add_subplot(222)
ax.plot(np.abs(shifted_states), ',k')
ax.set_xlim(0, n_states)
ax = fig.add_subplot(224)
overlaps = np.matmul(np.conjugate(shifted_states.T), shifted_states)
imref = ax.imshow(np.abs(overlaps), vmax=1.0, vmin=0)
plt.colorbar(imref)


# In[50]:


plt.figure(figsize=(6,6))
already_matched = set()
new_mat = np.zeros_like(overlaps)
last_i = 0
n_unique = 0
for ov_idx in np.arange(overlaps.shape[0]):
    these_matches = np.where(np.abs(overlaps[ov_idx]) > 0.65)[0]
    new_matches = set(these_matches) - already_matched
    if len(new_matches) == 0:
        continue
    n_unique += 1
    already_matched |= new_matches
    new_matches = np.array(list(new_matches))
    ordering = np.argsort(np.abs(overlaps[ov_idx, new_matches]))
    for i, j in enumerate(new_matches[ordering], start=last_i):
        new_mat[i, j] = 1
        last_i = i+1
print('{:d} unique physical states found'.format(n_unique))
plt.figure(figsize=(6,6))
resorted = np.matmul(new_mat, shifted_states.T).T
new_overlaps = np.matmul(np.conjugate(resorted.T), resorted)
plt.imshow(np.abs(new_overlaps), vmax=1.0, vmin=0)
plt.colorbar()
if len(set(np.arange(overlaps.shape[0])) - already_matched) > 0:
    print("Some states are not included")
    #raise ValueError("Some states are not included")
if not np.allclose(np.diag(np.abs(new_overlaps)), 1.0):
    print("Missing main diagonal somehow")
    #raise ValueError("Missing main diagonal somehow")


# In[44]:


np.allclose(new_mat.T @ new_mat, np.identity(new_mat.shape[0], dtype=np.complex))


# In[51]:


plt.imshow(np.real(resorted[2075:2175]).T, vmin=-1, vmax=1, cmap='RdBu', aspect=0.05)


# In[46]:


plt.plot(np.real(resorted[:,0]),label='0')
plt.plot(np.real(resorted[:,60]))


# In[ ]:




