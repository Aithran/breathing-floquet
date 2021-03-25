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


base_dir = os.path.join('/media', 'simulationData', 'BreathingLattice', 'floquet', 'small_lattice', 'critical')
done_flag_path = os.path.join(base_dir, 'done_flag.lck')
while not os.path.exists(done_flag_path):
    time.sleep(60)


# In[3]:


m_re = re.compile(r'.*_m(\d+)_out.h5')
def get_m(file_name):
    return int(m_re.search(file_name).group(1))
base_path = base_dir
data_dirs = [os.path.basename(dirpath[:-1]) for dirpath in glob.glob(base_path + "/critical*n33*/")]
data_paths = sorted([os.path.join(base_path, this_dir, this_dir + '_out.h5') for this_dir in data_dirs], key=get_m)
for idx, path in enumerate(data_paths):
    print('{}:\t{}'.format(idx, os.path.basename(path)))
largest_m_path = data_paths[-1]


# In[4]:


from cycler import cycler
kelly_cycler = (cycler(color=['#FFB300', '#803E75', '#FF6800',
              '#A6BDD7', '#C10020', '#CEA262',
              '#817066', '#007D34']) + cycler(linestyle=['-', '--', ':', '-.', '-', '--', ':', '-.']))


# In[30]:


get_ipython().run_cell_magic('time', '', 'skip_if_present = False\nn_sites = 33\nn_time_samples = 3*1024\nmin_samples = 10 # Warn if there\'s a component who won\'t be sampled faster than this\ntwo_pi_i = np.complex(0, np.pi*2.0)\ntry:\n    len(first_set_residuals_dict)\nexcept NameError:\n    first_set_residuals_dict = dict()\ntry:\n    len(last_set_residuals_dict)\nexcept NameError:\n    last_set_residuals_dict = dict()\nfor this_path in progbar(data_paths):\n    with h5py.File(this_path, \'r\', libver="latest") as f:\n        print(os.path.basename(this_path))\n        states_matrix = f[\'floquet_state\']\n        first_states = states_matrix[0]\n        last_states = states_matrix[-1]\n        n_states = states_matrix.shape[-1]\n        first_resid_pops = np.empty((n_states, n_time_samples))\n        last_resid_pops = np.empty((n_states, n_time_samples))\n        m_rad = ((states_matrix.shape[-1]//n_sites) - 1)//2\n        if skip_if_present and (m_rad in first_set_residuals_dict.keys()):\n            continue\n        if (n_time_samples/(2*m_rad) < min_samples):\n            print(\'Minimum sample violation {} < {}\'.format(n_time_samples/(2*m_rad), min_samples))\n        ##\n        #phasor = np.zeros((n_sites, (2*m_rad+1)*n_sites), dtype=np.complex)\n        #where_mat = np.tile(np.diag(np.full(n_sites, True)), 2*m_rad+1)\n        #m_block = np.kron(np.arange(-m_rad, m_rad+1), np.identity(n_sites)).astype(np.complex)\n        #for idx, t in progbar(enumerate(np.linspace(0.0, 1.0, n_time_samples)), size=n_time_samples, every=1):\n        #    np.exp(-two_pi_i * m_block * t, out=phasor, where=where_mat)\n        #    now_amps = np.matmul(phasor, first_states)\n        #    first_resid_pops[:, idx] = 1.0 - np.sum(np.abs(np.conjugate(now_amps) * now_amps), axis=0)\n        #    now_amps = np.matmul(phasor, last_states)\n        #    last_resid_pops[:, idx] = 1.0 - np.sum(np.abs(np.conjugate(now_amps) * now_amps), axis=0)\n        #first_set_residuals_dict[m_rad] = first_resid_pops.copy()\n        #last_set_residuals_dict[m_rad] = last_resid_pops.copy()\n        ##\n        first_states = first_states.reshape(2*m_rad+1, n_sites, -1)\n        last_states = last_states.reshape(2*m_rad+1, n_sites, -1)\n        m_block = (np.arange(-m_rad, m_rad+1) * -two_pi_i).reshape(-1,1,1)\n        site_amps = np.empty((n_sites, n_states), dtype=np.complex)\n        expand_states = np.empty((2*m_rad + 1, n_sites, n_states), dtype=np.complex)\n        for idx, t in progbar(enumerate(np.linspace(0.0, 1.0, n_time_samples)), size=n_time_samples):\n            phasor = np.exp(m_block * t)\n            np.multiply(phasor, first_states, out=expand_states)\n            np.sum(expand_states, axis=0, out=site_amps)\n            first_resid_pops[:, idx] = 1.0-np.sum(np.real(np.conjugate(site_amps) * site_amps), axis=0)\n            np.multiply(phasor, last_states, out=expand_states)\n            np.sum(expand_states, axis=0, out=site_amps)\n            last_resid_pops[:, idx] = 1.0-np.sum(np.real(np.conjugate(site_amps) * site_amps), axis=0)\n        first_set_residuals_dict[m_rad] = first_resid_pops.copy()\n        last_set_residuals_dict[m_rad] = last_resid_pops.copy()\n        ##\ndel first_states, last_states, m_block, site_amps, expand_states, phasor, first_resid_pops, last_resid_pops')


# In[31]:


ax = plt.subplot(111)
ax.set_prop_cycle(kelly_cycler)
for m_count, resids in first_set_residuals_dict.items():
    plt.plot(np.sort(np.sum(np.abs(resids), axis=1))/n_time_samples, label='{}-'.format(m_count))
for m_count, resids in last_set_residuals_dict.items():
    plt.plot(np.sort(np.sum(np.abs(resids), axis=1))/n_time_samples, label='{}+'.format(m_count))
plt.ylim(0,1)
plt.legend()


# In[32]:


colors = kelly_cycler.by_key()['color']
for idx, m_count in enumerate(np.sort(list(first_set_residuals_dict.keys()))[-len(colors):]):
    resids = first_set_residuals_dict[m_count]
    plt.plot(np.sort(np.sum(np.abs(resids), axis=1))/n_time_samples, label=m_count, c=colors[idx])
    resids = last_set_residuals_dict[m_count]
    plt.plot(np.sort(np.sum(np.abs(resids), axis=1))/n_time_samples, c=colors[idx], ls='--')
plt.legend()
plt.xlim(0,n_sites)
plt.ylim(0,0.01)


# In[33]:


colors = kelly_cycler.by_key()['color']
for idx, m_count in enumerate(np.sort(list(first_set_residuals_dict.keys()))[-len(colors):]):
    resids = first_set_residuals_dict[m_count]
    plt.semilogy(np.sort(np.sum(np.abs(resids), axis=1))/n_time_samples, label=m_count, c=colors[idx])
    resids = last_set_residuals_dict[m_count]
    plt.semilogy(np.sort(np.sum(np.abs(resids), axis=1))/n_time_samples, c=colors[idx], ls='--')
plt.legend()
plt.xlim(0,2*n_sites)
plt.axvline(n_sites, ls=':', c='k')
plt.ylim(1e-9,1e-2)


# In[34]:


m_rad = 68
plt.semilogy(np.sum(np.abs(last_set_residuals_dict[m_rad]), axis=1)/n_time_samples, '.', label='last')
plt.semilogy(np.sum(np.abs(first_set_residuals_dict[m_rad]), axis=1)/n_time_samples, '.', label='first')
for i in np.arange(2*(m_rad+1)):
    plt.axvline(i*n_sites, c='k', ls=':')
plt.axvline(m_rad*n_sites, c='k', ls='--')
plt.axvline((m_rad+1)*n_sites, c='k', ls='--')
plt.xlim(0,(2*m_rad+1)*n_sites)
plt.legend()


# In[35]:


m_out = 6
plt.semilogy(np.sum(np.abs(last_set_residuals_dict[m_rad]), axis=1)/n_time_samples, '.')
plt.semilogy(np.sum(np.abs(first_set_residuals_dict[m_rad]), axis=1)/n_time_samples, '.')
plt.ylim(None,0.01)
for i in np.arange(2*(m_rad+1)):
    plt.axvline(i*n_sites, c='k', ls=':')
plt.axvline(m_rad*n_sites, c='k', ls='--')
plt.axvline((m_rad+1)*n_sites, c='k', ls='--')
plt.xlim((m_rad-m_out)*n_sites,(m_rad+1+m_out)*n_sites)


# In[36]:


get_ipython().run_cell_magic('time', '', 'skip_if_present = False\nn_sites = 33\nn_time_samples = 3*1024\nmin_samples = 10 # Warn if there\'s a component who won\'t be sampled faster than this\ntwo_pi_i = np.complex(0, np.pi*2.0)\ntry:\n    len(first_set_residuals_dict)\nexcept NameError:\n    first_set_residuals_dict = dict()\ntry:\n    len(last_set_residuals_dict)\nexcept NameError:\n    last_set_residuals_dict = dict()\nfor this_path in progbar(data_paths):\n    with h5py.File(this_path, \'r\', libver="latest") as f:\n        print(os.path.basename(this_path))\n        states_matrix = f[\'floquet_state\']\n        first_states = states_matrix[0]\n        last_states = states_matrix[-1]\n        n_states = states_matrix.shape[-1]\n        first_resid_pops = np.empty((n_states, n_time_samples))\n        last_resid_pops = np.empty((n_states, n_time_samples))\n        m_rad = ((states_matrix.shape[-1]//n_sites) - 1)//2\n        if skip_if_present and (m_rad in first_set_residuals_dict.keys()):\n            continue\n        if (n_time_samples/(2*m_rad) < min_samples):\n            print(\'Minimum sample violation {} < {}\'.format(n_time_samples/(2*m_rad), min_samples))\n        ##\n        phasor = np.zeros((n_sites, (2*m_rad+1)*n_sites), dtype=np.complex)\n        where_mat = np.tile(np.diag(np.full(n_sites, True)), 2*m_rad+1)\n        m_block = np.kron(np.arange(-m_rad, m_rad+1), np.identity(n_sites)).astype(np.complex)\n        for idx, t in progbar(enumerate(np.linspace(0.0, 1.0, n_time_samples)), size=n_time_samples, every=1):\n            np.exp(-two_pi_i * m_block * t, out=phasor, where=where_mat)\n            now_amps = np.matmul(phasor, first_states)\n            first_resid_pops[:, idx] = 1.0 - np.sum(np.abs(np.conjugate(now_amps) * now_amps), axis=0)\n            now_amps = np.matmul(phasor, last_states)\n            last_resid_pops[:, idx] = 1.0 - np.sum(np.abs(np.conjugate(now_amps) * now_amps), axis=0)\n        first_set_residuals_dict[m_rad] = first_resid_pops.copy()\n        last_set_residuals_dict[m_rad] = last_resid_pops.copy()\n        ##\n        ##\n        #first_states = first_states.reshape(2*m_rad+1, n_sites, -1)\n        #last_states = last_states.reshape(2*m_rad+1, n_sites, -1)\n        #m_block = (np.arange(-m_rad, m_rad+1) * -two_pi_i).reshape(-1,1,1)\n        #site_amps = np.empty((n_sites, n_states), dtype=np.complex)\n        #expand_states = np.empty((2*m_rad + 1, n_sites, n_states), dtype=np.complex)\n        #for idx, t in progbar(enumerate(np.linspace(0.0, 1.0, n_time_samples)), size=n_time_samples):\n        #    phasor = np.exp(m_block * t)\n        #    np.multiply(phasor, first_states, out=expand_states)\n        #    np.sum(expand_states, axis=0, out=site_amps)\n        #    first_resid_pops[:, idx] = 1.0-np.sum(np.real(np.conjugate(site_amps) * site_amps), axis=0)\n        #    np.multiply(phasor, last_states, out=expand_states)\n        #    np.sum(expand_states, axis=0, out=site_amps)\n        #    last_resid_pops[:, idx] = 1.0-np.sum(np.real(np.conjugate(site_amps) * site_amps), axis=0)\n        #first_set_residuals_dict[m_rad] = first_resid_pops.copy()\n        #last_set_residuals_dict[m_rad] = last_resid_pops.copy()\n        ##\ndel first_states, last_states, m_block, site_amps, expand_states, phasor, first_resid_pops, last_resid_pops')


# In[39]:


get_ipython().run_cell_magic('time', '', 'n_sites = 33\nn_sets = 256\nset_stride = 2\nn_time_samples = 3*1024\nmin_samples = 10 # used to compare n_time_samples to fastest oscillation timescale...\ntwo_pi_i = np.complex(0, np.pi*2.0)\n# Results, 2D, one set per column\nthis_path = largest_m_path\n# Check quasienergy vs. residual, make sure we have enough unique states\nwith h5py.File(this_path, \'r\', libver="latest") as f:\n    print(os.path.basename(this_path))\n    states_matrix = f[\'floquet_state\']\n    m_rad = ((states_matrix.shape[-1]//n_sites) - 1)//2\n    n_states = states_matrix.shape[-1]\n    all_raw_residuals = np.empty((n_sets//set_stride, n_states))\n    resid_pops = np.empty((n_states, n_time_samples))\n    ##\n    phasor = np.zeros((n_sites, (2*m_rad+1)*n_sites), dtype=np.complex)\n    where_mat = np.tile(np.diag(np.full(n_sites, True)), 2*m_rad+1)\n    m_block = np.kron(np.arange(-m_rad, m_rad+1), np.identity(n_sites)).astype(np.complex)\n    for out_idx, this_set in progbar(enumerate(np.arange(0, n_sets, set_stride)), every=1, size=n_sets//set_stride):\n        these_states = states_matrix[this_set,:]\n        for idx, t in enumerate(np.linspace(0.0, 1.0, n_time_samples)):\n            np.exp(-two_pi_i * m_block * t, out=phasor, where=where_mat)\n            now_amps = np.matmul(phasor, these_states)\n            resid_pops[:, idx] = 1.0 - np.sum(np.abs(np.conjugate(now_amps) * now_amps), axis=0)\n        all_raw_residuals[out_idx] = np.sum(np.abs(resid_pops), axis=1)/n_time_samples\n    ##\n    ##\n    #m_block = (np.arange(-m_rad, m_rad+1) * -two_pi_i).reshape(-1,1,1)\n    #site_amps = np.empty((n_sites, n_states), dtype=np.complex)\n    #expand_states = np.empty((2*m_rad + 1, n_sites, n_states), dtype=np.complex)\n    #for out_idx, this_set in progbar(enumerate(np.arange(0, n_sets, set_stride)), every=1, size=n_sets//set_stride):\n    #    these_states = states_matrix[this_set,:].reshape(2*m_rad + 1, n_sites, -1)\n    #    for idx, t in enumerate(np.linspace(0.0, 1.0, n_time_samples)):\n    #        phasor = np.exp(m_block * t)\n    #        np.multiply(phasor, these_states, out=expand_states)\n    #        np.sum(expand_states, axis=0, out=site_amps)\n    #        resid_pops[:, idx] = 1.0-np.sum(np.real(np.conjugate(site_amps) * site_amps), axis=0)\n    #    all_raw_residuals[out_idx] = np.sum(np.abs(resid_pops), axis=1)/n_time_samples\n    ##\ndel expand_states, phasor, site_amps, resid_pops, m_block')


# In[40]:


#raise RuntimeException("Don't overwrite things yet")
result_file_path = os.path.join(base_path, 'residual_pops_critical.h5')
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


# In[41]:


clear_first_last = True
if clear_first_last:
    last_set_residuals_dict.clear()
    first_set_residuals_dict.clear()
    del last_set_residuals_dict, first_set_residuals_dict, residuals


# In[42]:


with h5py.File(largest_m_path, 'r', libver="latest") as f:
    gamma_list = f['scan_values'][:]
thresh_list = np.power(10., np.arange(-6,-1))
count_below_thresh = np.empty((len(thresh_list), all_raw_residuals.shape[0]))
for idx, this_thresh in progbar(enumerate(thresh_list), every=1):
    count_below_thresh[idx] = np.sum(all_raw_residuals < this_thresh, axis=1)


# In[45]:


for idx, this_thresh in enumerate(thresh_list):
    plt.plot(gamma_list[::set_stride], count_below_thresh[idx], '.', label=this_thresh)
#plt.ylim(0,2.5*n_sites)
plt.axhline( n_sites, c='k', ls=':')
plt.axhline(2*n_sites, c='k', ls=':')
#plt.legend(thresh_list)
plt.legend()


# In[44]:


resid_file_path = os.path.join(base_path, 'critical_m68_all_residuals.h5')
print(resid_file_path)
#raise RuntimeException("Don't overwrite things yet")
with h5py.File(resid_file_path, 'w', libver="latest") as sf:
    full_scan = sf.create_dataset('total_residuals', data=all_raw_residuals)
    full_scan.attrs['m_rad'] = m_rad
    full_scan.attrs['source_path'] = str(this_path)
    full_scan.attrs['n_time_samples'] = n_time_samples


# In[51]:


resid_file_path = os.path.join(base_path, 'critical_m68_all_residuals.h5')
print(resid_file_path)
#raise RuntimeException("Don't overwrite things yet")
with h5py.File(resid_file_path, 'r', libver="latest") as sf:
    full_scan         = sf['total_residuals']
    all_raw_residuals = full_scan[...]
    m_rad             = full_scan.attrs['m_rad']
    this_path         = full_scan.attrs['source_path']
    n_time_samples    = full_scan.attrs['n_time_samples']


# In[6]:


n_sets = 256
set_stride = 2
n_sites=33
n_states = n_sites * (2*68 + 1)
print(n_states)
all_raw_residuals.shape


# In[7]:


plt.plot(all_raw_residuals.T, ',k')
plt.xlim(0,4515)


# In[8]:


with h5py.File(this_path, 'r', libver='latest') as f:
    print(os.path.basename(this_path))
    matching_quasi = f['floquet_energy'][::set_stride]
    matching_drive = f['critcal_drive'][::set_stride]
    matching_gamma = f['scan_values'][::set_stride]


# In[9]:


print(all_raw_residuals.shape,
     matching_drive.shape,
     matching_gamma.shape,
     matching_quasi.shape)


# In[10]:


fig = plt.figure(figsize=(8,6))
ax = plt.subplot(111)
for idx, gamma in enumerate(matching_gamma):
    stable_states = np.where(all_raw_residuals[idx] < 0.001)
    stable_quasi = matching_quasi[idx, stable_states].flatten()
    fbz_quasi = np.mod(stable_quasi + 0.5*matching_drive[idx], matching_drive[idx]) - 0.5*matching_drive[idx]
    ax.semilogx(np.full(len(stable_states[0]), gamma), fbz_quasi, ',k')
ax.plot(matching_gamma,  0.5*matching_drive, ':r')
ax.plot(matching_gamma, -0.5*matching_drive, ':r')
plt.ylim(-10, 10)
plt.xlim(1e-5, 0.1)


# # Shift matrix to weed out similar physical states?
# The plan:
# For a state |n> with quasi-energy e_nm
# Find FBZ quasienergy e_n0 via e_n0 = mod(e_nm + w/2) - w/2,
# Find also the shift amount m via (e_nm + w/2)//w
# Compute |n'> by doing S_(m*n_sites) |n>
# Once these have been built, do <l|n> and see what states overlap significantly

# In[11]:


def shift_by_m_blocks(m_shift, n_sites, n_states):
    return np.eye( n_states, k=m_shift*n_sites)


# In[12]:


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


# In[52]:


get_ipython().run_cell_magic('time', '', 'shifted_scans = dict()\nwith h5py.File(this_path, \'r\', libver="latest") as f:\n    print(os.path.basename(this_path))\n    states_matrix = f[\'floquet_state\']\n    for out_idx, this_set in progbar(enumerate(np.arange(0, n_sets, set_stride)),\n                                     size=n_sets//set_stride):\n        which_states = np.where(all_raw_residuals[out_idx] < 5e-5)[0]\n        stable_states = states_matrix[this_set,:,which_states]\n        stable_quasi  = f[\'floquet_energy\'][this_set, which_states]\n        this_drive = f[\'critcal_drive\'][this_set]\n        shifted_states = np.zeros_like(stable_states)\n        mshift, fqe = np.divmod(stable_quasi + 0.5*this_drive, this_drive)\n        mshift = (mshift).astype(np.int)\n        fqe -= 0.5*this_drive\n        zero_counter = 0\n        for idx in np.arange(len(which_states)):\n            if np.abs(mshift[idx]) > 2*m_rad:\n                zero_counter += 1\n                continue\n            dest_imin = 0 if mshift[idx] >= 0 else -mshift[idx] * n_sites\n            dest_imax = n_states if mshift[idx] <= 0 else n_states - (mshift[idx] * n_sites)\n            unshifted_imin = 0 if mshift[idx] <= 0 else mshift[idx] * n_sites \n            unshifted_imax = n_states if mshift[idx] >= 0 else n_states + (mshift[idx] * n_sites)\n            shifted_states[dest_imin:dest_imax, idx] = stable_states[unshifted_imin:unshifted_imax, idx]\n        shifted_scans[out_idx] = {\'fqe\':fqe.copy(),\n                                  \'mshift\':mshift.copy(),\n                                  \'shifted_states\':shifted_states.copy(),\n                                  \'zero_states\':zero_counter,\n                                  \'which_states\':which_states.copy(),\n                                  \'matching_resids\':all_raw_residuals[out_idx, which_states]\n                                 }\ndel stable_states, stable_quasi, all_raw_residuals, shifted_states, which_states')


# In[53]:


which_scan = 0
fqe = shifted_scans[which_scan]['fqe']
mshift = shifted_scans[which_scan]['mshift']
shifted_states = shifted_scans[which_scan]['shifted_states']


# In[54]:


for this_idx in list(shifted_scans.keys()):
    plt.plot(this_idx, shifted_scans[this_idx]['zero_states'],'.k')


# In[55]:


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(221)
ax.plot(fqe, ',')
ax = fig.add_subplot(223)
ax.plot(np.sort(fqe), ',')
ax = fig.add_subplot(222)
ax.plot(np.abs(shifted_states), ',k')
ax.set_xlim(0, n_states)
ax = fig.add_subplot(224)
overlaps = np.matmul(np.conjugate(shifted_states.T), shifted_states)
imref = ax.imshow(np.abs(overlaps), vmax=1.0, vmin=0)
plt.colorbar(imref)


# In[78]:


plt.figure(figsize=(6,6))
already_matched = set()
new_mat = np.zeros_like(overlaps)
last_i = 0
n_unique = 0
state_groups = list()
group_starts = list()
for ov_idx in np.arange(overlaps.shape[0]):
    these_matches = np.where(np.abs(overlaps[ov_idx]) > 0.8)[0]
    new_matches = set(these_matches) - already_matched
    if len(new_matches) == 0:
        continue
    state_groups.append(np.sort(list(new_matches)))
    n_unique += 1
    already_matched |= new_matches
    new_matches = np.array(list(new_matches))
    ordering = np.argsort(np.abs(overlaps[ov_idx, new_matches]))
    group_starts.append(last_i)
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


# In[66]:


plt.plot(1-np.diag(np.abs(new_overlaps)))


# In[68]:


print(group_starts)
import itertools
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)
print(new_overlaps.shape)


# In[69]:


fig, ax = plt.subplots(1)
img = ax.imshow(np.abs(new_overlaps[1200:,1200:]), vmax=1.0, vmin=0, extent=(1700.5,1806.5,1806.5,1700.5))
plt.colorbar(img)
from matplotlib import patches
for x, y in pairwise(group_starts):
    ax.add_patch(patches.Rectangle((x, x), y-x, y-x, fill=False, ec='m', ls='--', lw=2))


# In[70]:


print(state_groups[0])


# In[71]:


get_ipython().run_cell_magic('time', '', 'which_check=33\norig_idx = (shifted_scans[which_scan][\'which_states\'])[state_groups[which_check]]\nwith h5py.File(this_path, \'r\', libver="latest") as f:\n    print(os.path.basename(this_path))\n    states_matrix = f[\'floquet_state\']\n    check_equiv = states_matrix[set_stride*which_scan,:,orig_idx]\nfig, ax_ar = plt.subplots(4)\nax_ar[0].plot(\n    np.sum(\n        np.repeat(np.arange(-m_rad,m_rad+1), n_sites)[:,np.newaxis]*\n               np.square(np.abs(check_equiv)), axis=0)/np.sum(np.square(np.abs(check_equiv)), axis=0), \'.\')\nax_ar[0].plot(\n    np.sum(\n        np.repeat(np.abs(np.arange(-m_rad,m_rad+1)), n_sites)[:,np.newaxis]*\n               np.square(np.abs(check_equiv)), axis=0), \'.\')\nax_ar[1].plot(shifted_scans[which_scan][\'mshift\'][state_groups[which_check]], \'.\')\nax_ar[2].semilogy(shifted_scans[which_scan][\'matching_resids\'][state_groups[which_check]], \'.\')\nmin_idx = np.argmin(shifted_scans[which_scan][\'matching_resids\'][state_groups[which_check]])\nax_ar[3].plot(np.tile(np.arange(-n_sites//2+1,n_sites//2+1), 2*m_rad+1), np.square(np.abs(check_equiv[:,min_idx])), \'.\')\nprint(check_equiv.shape, min_idx)')


# In[72]:


plt.plot(np.repeat(np.arange(-m_rad,m_rad+1), n_sites)[:,np.newaxis])


# In[73]:


get_ipython().run_cell_magic('time', '', "fig, ax_ar = plt.subplots(2)\nax_ar[1].plot(np.real(shifted_scans[which_scan]['shifted_states'][:,state_groups[which_check]]), ',')\nax_ar[0].set_ylim(ax_ar[1].get_ylim())\nax_ar[0].plot(np.real(check_equiv), '.')")


# In[77]:


get_ipython().run_cell_magic('time', '', "fig, ax_ar = plt.subplots(2)\nax_ar[0].plot(np.real(check_equiv[:,min_idx]), '.')\nax_ar[1].plot(np.real(check_equiv[:,0]), '.')")


# In[ ]:


np.allclose(new_mat.T @ new_mat, np.identity(new_mat.shape[0], dtype=np.complex))


# In[285]:


which_scan


# In[ ]:


get_ipython().run_line_magic('reset', 'in out')


# In[19]:


plt.imshow(np.real(resorted[2500:2800]).T, vmin=-1, vmax=1, cmap='RdBu')


# In[21]:


plt.imshow(np.real(resorted[2245:2275]), vmin=-0.5, vmax=0.5, cmap='RdBu', aspect=4)


# In[ ]:




