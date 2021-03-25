#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from matplotlib import pyplot as plt
import h5py
import os
import glob
from zss_progbar import log_progress as progbar
import re
import time


# In[ ]:


base_dir = os.path.join('/media', 'simulationData', 'BreathingLattice', 'floquet', 'small_lattice')
base_dir = os.path.join('/media', 'simulationData', 'FromKhronos')
#done_flag_path = os.path.join(base_dir, 'done_flag.lck')
#while not os.path.exists(done_flag_path):
#    time.sleep(60)


# In[ ]:


m_re = re.compile(r'.*_m(\d+)_out.h5')
def get_m(file_name):
    return int(m_re.search(file_name).group(1))
base_path = base_dir
data_dirs = [os.path.basename(dirpath[:-1]) for dirpath in glob.glob(base_path + "/shallow*/")]
print(data_dirs)
#data_paths = [os.path.join(base_path, this_dir, this_dir + '_out.h5') for this_dir in data_dirs]
data_paths = sorted([os.path.join(base_path, this_dir, this_dir + '_out.h5') for this_dir in data_dirs], key=get_m)
for idx, path in enumerate(data_paths):
    print('{}:\t{}'.format(idx, os.path.basename(path)))
largest_m_path = data_paths[-1]


# In[ ]:


from cycler import cycler
kelly_cycler = (cycler(color=['#FFB300', '#803E75', '#FF6800',
              '#A6BDD7', '#C10020', '#CEA262',
              '#817066', '#007D34']) + cycler(linestyle=['-', '--', ':', '-.', '-', '--', ':', '-.']))


# In[ ]:


get_ipython().run_cell_magic('time', '', 'skip_if_present = False\nn_sites = 33\nn_time_samples = 5*1024\nmin_samples = 10 # Warn if there\'s a component who won\'t be sampled faster than this\ntwo_pi_i = np.complex(0, np.pi*2.0)\ntry:\n    len(first_set_residuals_dict)\nexcept NameError:\n    first_set_residuals_dict = dict()\ntry:\n    len(last_set_residuals_dict)\nexcept NameError:\n    last_set_residuals_dict = dict()\nfor this_path in progbar(data_paths):\n    with h5py.File(this_path, \'r\', libver="latest") as f:\n        print(os.path.basename(this_path))\n        states_matrix = f[\'floquet_state\']\n        first_states = states_matrix[0]\n        last_states = states_matrix[-1]\n        n_states = states_matrix.shape[-1]\n        first_resid_pops = np.zeros((n_states))\n        last_resid_pops = np.zeros((n_states))\n        m_rad = ((states_matrix.shape[-1]//n_sites) - 1)//2\n        if skip_if_present and (m_rad in first_set_residuals_dict.keys()):\n            continue\n        if (n_time_samples/(2*m_rad) < min_samples):\n            print(\'Minimum sample violation {} < {}\'.format(n_time_samples/(2*m_rad), min_samples))\n        ##\n        #phasor = np.zeros((n_sites, (2*m_rad+1)*n_sites), dtype=np.complex)\n        #where_mat = np.tile(np.diag(np.full(n_sites, True)), 2*m_rad+1)\n        #m_block = np.kron(np.arange(-m_rad, m_rad+1), np.identity(n_sites)).astype(np.complex)\n        #for idx, t in progbar(enumerate(np.linspace(0.0, 1.0, n_time_samples)), size=n_time_samples, every=1):\n        #    np.exp(-two_pi_i * m_block * t, out=phasor, where=where_mat)\n        #    now_amps = np.matmul(phasor, first_states)\n        #    first_resid_pops[:, idx] = 1.0 - np.sum(np.abs(np.conjugate(now_amps) * now_amps), axis=0)\n        #    now_amps = np.matmul(phasor, last_states)\n        #    last_resid_pops[:, idx] = 1.0 - np.sum(np.abs(np.conjugate(now_amps) * now_amps), axis=0)\n        #first_set_residuals_dict[m_rad] = first_resid_pops.copy()\n        #last_set_residuals_dict[m_rad] = last_resid_pops.copy()\n        ##\n        first_states = first_states.reshape(2*m_rad+1, n_sites, -1)\n        last_states = last_states.reshape(2*m_rad+1, n_sites, -1)\n        m_block = (np.arange(-m_rad, m_rad+1) * -two_pi_i).reshape(-1,1,1)\n        site_amps = np.empty((n_sites, n_states), dtype=np.complex)\n        expand_states = np.empty((2*m_rad + 1, n_sites, n_states), dtype=np.complex)\n        for idx, t in progbar(enumerate(np.linspace(0.0, 1.0, n_time_samples)), size=n_time_samples):\n            phasor = np.exp(m_block * t)\n            np.multiply(phasor, first_states, out=expand_states)\n            np.sum(expand_states, axis=0, out=site_amps)\n            first_resid_pops[:] += np.abs(1.0-np.sum(np.real(np.conjugate(site_amps) * site_amps), axis=0))\n            np.multiply(phasor, last_states, out=expand_states)\n            np.sum(expand_states, axis=0, out=site_amps)\n            last_resid_pops[:] += np.abs(1.0-np.sum(np.real(np.conjugate(site_amps) * site_amps), axis=0))\n        first_set_residuals_dict[m_rad] = first_resid_pops.copy()\n        last_set_residuals_dict[m_rad] = last_resid_pops.copy()\n        ##\ndel first_states, last_states, m_block, site_amps, expand_states, phasor, first_resid_pops, last_resid_pops')


# In[ ]:


first_set_residuals_dict[100].shape


# In[ ]:


ax = plt.subplot(111)
ax.set_prop_cycle(kelly_cycler)
#for m_count, resids in last_set_residuals_dict.items():
plt.plot(np.sort(np.abs(first_set_residuals_dict[100]))/n_time_samples, label='First')
plt.plot(np.sort(np.abs(last_set_residuals_dict[100]))/n_time_samples, label='Last')

#    plt.plot(np.sort(np.sum(np.abs(resids), axis=1))/n_time_samples, label='{}+'.format(m_count))
plt.ylim(0,1)
plt.legend()


# In[ ]:


first_set_residuals_dict.keys()


# In[ ]:


colors = kelly_cycler.by_key()['color']
for idx, m_count in enumerate(np.sort(list(first_set_residuals_dict.keys()))[-len(colors):]):
    resids = first_set_residuals_dict[m_count]
    plt.plot(np.sort((np.abs(resids)))/n_time_samples, label=m_count, c=colors[idx])
    resids = last_set_residuals_dict[m_count]
    plt.plot(np.sort((np.abs(resids)))/n_time_samples, c=colors[idx], ls='--')
plt.legend()
plt.xlim(0,n_sites)
plt.ylim(0,0.01)


# In[ ]:


colors = kelly_cycler.by_key()['color']
for idx, m_count in enumerate(np.sort(list(first_set_residuals_dict.keys()))[-len(colors):]):
    resids = first_set_residuals_dict[m_count]
    plt.semilogy(np.sort(np.abs(resids))/n_time_samples, label=m_count, c=colors[idx])
    resids = last_set_residuals_dict[m_count]
    plt.semilogy(np.sort(np.abs(resids))/n_time_samples, c=colors[idx], ls='--')
plt.legend()
plt.xlim(0,128*n_sites)
plt.axvline(n_sites, ls=':', c='k')
plt.ylim(1e-16,1)


# In[ ]:


m_rad = 100
plt.semilogy(np.abs(last_set_residuals_dict[m_rad])/n_time_samples, ',', label='last')
plt.semilogy(np.abs(first_set_residuals_dict[m_rad])/n_time_samples, ',', label='first')
#for i in np.arange(2*(m_rad+1)):
#    plt.axvline(i*n_sites, c='k', ls=':')
#plt.axvline(m_rad*n_sites, c='k', ls='--')
#plt.axvline((m_rad+1)*n_sites, c='k', ls='--')
plt.xlim(0,(2*m_rad+1)*n_sites)
plt.legend()


# In[ ]:


n_sites = 33
n_sets = 256
set_stride = 16
n_time_samples = 5*1024
min_samples = 10 # used to compare n_time_samples to fastest oscillation timescale...
two_pi_i = np.complex(0, np.pi*2.0)
# Results, 2D, one set per column
this_path = largest_m_path
# Check quasienergy vs. residual, make sure we have enough unique states
with h5py.File(this_path, 'r', libver="latest") as f:
    print(os.path.basename(this_path))
    states_matrix = f['floquet_state']
    m_rad = ((states_matrix.shape[-1]//n_sites) - 1)//2
    n_states = states_matrix.shape[-1]
    all_raw_residuals = np.empty((n_sets//set_stride, n_states))
    resid_pops = np.empty((n_states, n_time_samples))
    ##
    phasor = np.zeros((n_sites, (2*m_rad+1)*n_sites), dtype=np.complex)
    where_mat = np.tile(np.diag(np.full(n_sites, True)), 2*m_rad+1)
    m_block = np.kron(np.arange(-m_rad, m_rad+1), np.identity(n_sites)).astype(np.complex)
    for out_idx, this_set in progbar(enumerate(np.arange(0, n_sets, set_stride)), every=1, size=n_sets//set_stride):
        these_states = states_matrix[this_set,:]
        for idx, t in progbar(enumerate(np.linspace(0.0, 1.0, n_time_samples)), size=n_time_samples, every=32):
            np.exp(-two_pi_i * m_block * t, out=phasor, where=where_mat)
            now_amps = np.matmul(phasor, these_states)
            resid_pops[:, idx] = 1.0 - np.sum(np.abs(np.conjugate(now_amps) * now_amps), axis=0)
        all_raw_residuals[out_idx] = np.sum(np.abs(resid_pops), axis=1)/n_time_samples
    ##
    ##
    #m_block = (np.arange(-m_rad, m_rad+1) * -two_pi_i).reshape(-1,1,1)
    #site_amps = np.empty((n_sites, n_states), dtype=np.complex)
    #expand_states = np.empty((2*m_rad + 1, n_sites, n_states), dtype=np.complex)
    #for out_idx, this_set in progbar(enumerate(np.arange(0, n_sets, set_stride)), every=1, size=n_sets//set_stride):
    #    these_states = states_matrix[this_set,:].reshape(2*m_rad + 1, n_sites, -1)
    #    for idx, t in enumerate(np.linspace(0.0, 1.0, n_time_samples)):
    #        phasor = np.exp(m_block * t)
    #        np.multiply(phasor, these_states, out=expand_states)
    #        np.sum(expand_states, axis=0, out=site_amps)
    #        resid_pops[:, idx] = 1.0-np.sum(np.real(np.conjugate(site_amps) * site_amps), axis=0)
    #    all_raw_residuals[out_idx] = np.sum(np.abs(resid_pops), axis=1)/n_time_samples
    ##
del phasor, site_amps, resid_pops, m_block


# In[ ]:


#raise RuntimeException("Don't overwrite things yet")
result_file_path = os.path.join(base_path, 'residual_pops_critical_m100.h5')
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


# In[ ]:


clear_first_last = True
if clear_first_last:
    last_set_residuals_dict.clear()
    first_set_residuals_dict.clear()
    del last_set_residuals_dict, first_set_residuals_dict, residuals


# In[ ]:


with h5py.File(largest_m_path, 'r', libver="latest") as f:
    gamma_list = f['scan_values'][:]
thresh_list = np.power(10., np.arange(-6,-1))
count_below_thresh = np.empty((len(thresh_list), all_raw_residuals.shape[0]))
for idx, this_thresh in progbar(enumerate(thresh_list), every=1):
    count_below_thresh[idx] = np.sum(all_raw_residuals < this_thresh, axis=1)


# In[ ]:


for idx, this_thresh in enumerate(thresh_list):
    plt.plot(gamma_list[::set_stride], count_below_thresh[idx], '.', label=this_thresh)
#plt.ylim(0,2.5*n_sites)
plt.axhline( n_sites, c='k', ls=':')
plt.axhline(2*n_sites, c='k', ls=':')
#plt.legend(thresh_list)
plt.legend()


# In[ ]:


resid_file_path = os.path.join(base_path, 'critical_m100_all_residuals.h5')
print(resid_file_path)
#raise RuntimeException("Don't overwrite things yet")
with h5py.File(resid_file_path, 'w', libver="latest") as sf:
    full_scan = sf.create_dataset('total_residuals', data=all_raw_residuals)
    full_scan.attrs['m_rad'] = m_rad
    full_scan.attrs['source_path'] = str(this_path)
    full_scan.attrs['n_time_samples'] = n_time_samples


# In[ ]:


resid_file_path = os.path.join(base_path, 'residual_pops_critical_100.h5')
print(resid_file_path)
#raise RuntimeException("Don't overwrite things yet")
with h5py.File(resid_file_path, 'r', libver="latest") as sf:
    full_scan         = sf['total_residuals']
    all_raw_residuals = full_scan[...]
    m_rad             = full_scan.attrs['m_rad']
    this_path         = full_scan.attrs['source_path']
    n_time_samples    = full_scan.attrs['n_time_samples']


# In[ ]:


n_sets = 256
set_stride = 16
n_sites=33
n_states = n_sites * (2*100 + 1)
print(n_states)
all_raw_residuals.shape


# In[ ]:


plt.plot(all_raw_residuals[0], ',k')
plt.plot(all_raw_residuals[-1], ',r')
plt.xlim(0,4515)


# In[ ]:


with h5py.File(this_path, 'r', libver='latest') as f:
    print(os.path.basename(this_path))
    matching_quasi = f['floquet_energy'][::set_stride]
    matching_drive = f['critcal_drive'][::set_stride]
    matching_gamma = f['scan_values'][::set_stride]


# In[ ]:


print(all_raw_residuals.shape,
     matching_drive.shape,
     matching_gamma.shape,
     matching_quasi.shape)


# In[ ]:


fig = plt.figure(figsize=(2,6))
ax = plt.subplot(111)
for idx, gamma in enumerate(matching_gamma):
    stable_states = np.where(all_raw_residuals[idx] < 1e-10)
    stable_quasi = matching_quasi[idx, stable_states].flatten()
    fbz_quasi = np.mod(stable_quasi + 0.5*matching_drive[idx], matching_drive[idx]) - 0.5*matching_drive[idx]
    ax.semilogx(np.full(len(stable_states[0]), gamma), fbz_quasi, '.k')
ax.plot(matching_gamma,  0.5*matching_drive, ':r')
ax.plot(matching_gamma, -0.5*matching_drive, ':r')
plt.ylim(-25, 25)
plt.xlim(1e-5, 0.1)


# In[ ]:


plt.semilogy(matching_drive)


# # Shift matrix to weed out similar physical states?
# The plan:
# For a state |n> with quasi-energy e_nm
# Find FBZ quasienergy e_n0 via e_n0 = mod(e_nm + w/2) - w/2,
# Find also the shift amount m via (e_nm + w/2)//w
# Compute |n'> by doing S_(m*n_sites) |n>
# Once these have been built, do <l|n> and see what states overlap significantly

# In[ ]:


def shift_by_m_blocks(m_shift, n_sites, n_states):
    return np.eye( n_states, k=m_shift*n_sites)


# In[ ]:


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


# In[ ]:


get_ipython().run_cell_magic('time', '', 'shifted_scans = dict()\nwith h5py.File(this_path, \'r\', libver="latest") as f:\n    print(os.path.basename(this_path))\n    states_matrix = f[\'floquet_state\']\n    for out_idx, this_set in progbar(enumerate(np.arange(0, n_sets, set_stride)),\n                                     size=n_sets//set_stride):\n        which_states = np.where(all_raw_residuals[out_idx] < 1e-13)[0]\n        stable_states = states_matrix[this_set,:,which_states]\n        stable_quasi  = f[\'floquet_energy\'][this_set, which_states]\n        this_drive = f[\'critcal_drive\'][this_set]\n        shifted_states = np.zeros_like(stable_states)\n        mshift, fqe = np.divmod(stable_quasi + 0.5*this_drive, this_drive)\n        mshift = (mshift).astype(np.int)\n        fqe -= 0.5*this_drive\n        zero_counter = 0\n        for idx in np.arange(len(which_states)):\n            if np.abs(mshift[idx]) > 2*m_rad:\n                zero_counter += 1\n                continue\n            dest_imin = 0 if mshift[idx] >= 0 else -mshift[idx] * n_sites\n            dest_imax = n_states if mshift[idx] <= 0 else n_states - (mshift[idx] * n_sites)\n            unshifted_imin = 0 if mshift[idx] <= 0 else mshift[idx] * n_sites \n            unshifted_imax = n_states if mshift[idx] >= 0 else n_states + (mshift[idx] * n_sites)\n            shifted_states[dest_imin:dest_imax, idx] = stable_states[unshifted_imin:unshifted_imax, idx]\n        shifted_scans[out_idx] = {\'fqe\':fqe.copy(),\n                                  \'mshift\':mshift.copy(),\n                                  \'shifted_states\':shifted_states.copy(),\n                                  \'zero_states\':zero_counter,\n                                  \'which_states\':which_states.copy(),\n                                  \'matching_resids\':all_raw_residuals[out_idx, which_states]\n                                 }\n#del stable_states, stable_quasi, all_raw_residuals, shifted_states, which_states')


# In[ ]:


which_scan = 0
fqe = shifted_scans[which_scan]['fqe']
mshift = shifted_scans[which_scan]['mshift']
shifted_states = shifted_scans[which_scan]['shifted_states']


# In[ ]:


for this_idx in list(shifted_scans.keys()):
    plt.plot(this_idx, shifted_scans[this_idx]['zero_states'],'.k')


# In[ ]:


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(221)
ax.plot(fqe, ',')
ax = fig.add_subplot(223)
ax.plot(np.sort(fqe), ',')
ax = fig.add_subplot(222)
ax.plot(np.abs(shifted_states), ',k')
ax.set_xlim(0, n_states)
ax = fig.add_subplot(224)
overlaps = np.empty((shifted_states.shape[-1], shifted_states.shape[-1]), dtype=np.complex)
np.matmul(np.conjugate(shifted_states.T), shifted_states, out=overlaps)
imref = ax.imshow(np.abs(overlaps), vmax=1.0, vmin=0)
plt.colorbar(imref)


# In[ ]:


plt.figure(figsize=(6,6))
already_matched = set()
new_mat = np.zeros_like(overlaps)
last_i = 0
n_unique = 0
state_groups = list()
group_starts = list()
grouped_fqe=np.empty((overlaps.shape[0],2))
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
        grouped_fqe[i] = [n_unique-1, fqe[j]]
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


# In[ ]:


plt.plot(np.argsort(grouped_fqe[:,1]))


# In[ ]:


mean_fqe = np.empty(33)
stddev_fqe = np.empty(33)
count_fqe = np.empty(33)
for q in np.arange(33):
    these = grouped_fqe[:,1][grouped_fqe[:,0] == q]
    mean_fqe[q] = np.mean(these)
    stddev_fqe[q] = np.std(these)
    count_fqe[q] = these.shape[0]
fqe_ord = np.argsort(mean_fqe)
plt.plot(mean_fqe[fqe_ord], '.', label='mean')


# In[ ]:


plt.plot(count_fqe,'.')


# In[ ]:


plt.semilogy(np.diff(mean_fqe[fqe_ord]), label='Diff')
plt.semilogy(stddev_fqe[fqe_ord][:-2])
plt.semilogy(stddev_fqe[fqe_ord][1:])
plt.legend()


# In[ ]:


plt.plot(1-np.diag(np.abs(new_overlaps)))


# In[ ]:


print(group_starts)
import itertools
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)
print(new_overlaps.shape)


# In[ ]:


group_sizes = np.diff(np.append(group_starts, new_overlaps.shape[0]))
sizes_order = np.argsort(group_sizes)[::-1]
plt.plot(group_sizes[sizes_order], '.')
print(group_sizes[group_sizes > 64].shape, group_sizes[group_sizes > 64].min())


# In[ ]:


fig, ax = plt.subplots(1)
img = ax.imshow(np.abs(new_overlaps), vmax=1.0, vmin=0)
plt.colorbar(img)
from matplotlib import patches
for x, y in pairwise(np.append(group_starts, new_overlaps.shape[0])):
    ax.add_patch(patches.Rectangle((x, x), y-x, y-x, fill=False, ec='m', ls='--', lw=2))


# In[ ]:


get_ipython().run_cell_magic('time', '', 'which_check = 1\norig_idx = (shifted_scans[which_scan][\'which_states\'])[state_groups[which_check]]\nwith h5py.File(this_path, \'r\', libver="latest") as f:\n    print(os.path.basename(this_path))\n    states_matrix = f[\'floquet_state\']\n    check_equiv = states_matrix[set_stride*which_scan,:,orig_idx]\nfig, ax_ar = plt.subplots(4)\nax_ar[0].plot(\n    np.sum(\n        np.repeat(np.arange(-m_rad,m_rad+1), n_sites)[:,np.newaxis]*\n               np.square(np.abs(check_equiv)), axis=0)/np.sum(np.square(np.abs(check_equiv)), axis=0), \'.\')\nax_ar[0].plot(\n    np.sum(\n        np.repeat(np.abs(np.arange(-m_rad,m_rad+1)), n_sites)[:,np.newaxis]*\n               np.square(np.abs(check_equiv)), axis=0), \'.\')\nax_ar[1].plot(shifted_scans[which_scan][\'mshift\'][state_groups[which_check]], \'.\')\nax_ar[2].semilogy(shifted_scans[which_scan][\'matching_resids\'][state_groups[which_check]], \'.\')\nmin_idx = np.argmin(shifted_scans[which_scan][\'matching_resids\'][state_groups[which_check]])\nax_ar[3].plot(np.tile(np.arange(-n_sites//2+1,n_sites//2+1), 2*m_rad+1), np.square(np.abs(check_equiv[:,min_idx])), \'.\')\nprint(check_equiv.shape, min_idx)')


# In[ ]:


get_ipython().run_cell_magic('time', '', "fig, ax_ar = plt.subplots(2)\nax_ar[1].plot(np.real(shifted_scans[which_scan]['shifted_states'][:,state_groups[which_check]]), ',')\nax_ar[0].set_ylim(ax_ar[1].get_ylim())\nax_ar[0].plot(np.real(check_equiv), '.')")


# In[ ]:


get_ipython().run_cell_magic('time', '', "fig, ax_ar = plt.subplots(2)\nax_ar[0].plot(np.real(check_equiv[:,min_idx]), '.')\nax_ar[1].plot(np.real(check_equiv[:,0]), '.')")


# In[ ]:


get_ipython().run_line_magic('reset', 'in out')


# In[ ]:


### Look at squared resid instead of abs


# In[ ]:


n_sites = 33
n_sets = 256
set_stride = 128
n_time_samples = 5*1024
min_samples = 10 # used to compare n_time_samples to fastest oscillation timescale...
two_pi_i = np.complex(0, np.pi*2.0)
# Results, 2D, one set per column
this_path = largest_m_path
# Check quasienergy vs. residual, make sure we have enough unique states
with h5py.File(this_path, 'r', libver="latest") as f:
    print(os.path.basename(this_path))
    states_matrix = f['floquet_state']
    m_rad = ((states_matrix.shape[-1]//n_sites) - 1)//2
    n_states = states_matrix.shape[-1]
    all_sq_residuals = np.empty((n_sets//set_stride, n_states))
    resid_pops = np.empty((n_states, n_time_samples))
    ##
    phasor = np.zeros((n_sites, (2*m_rad+1)*n_sites), dtype=np.complex)
    where_mat = np.tile(np.diag(np.full(n_sites, True)), 2*m_rad+1)
    m_block = np.kron(np.arange(-m_rad, m_rad+1), np.identity(n_sites)).astype(np.complex)
    for out_idx, this_set in progbar(enumerate(np.arange(0, n_sets, set_stride)), every=1, size=n_sets//set_stride):
        these_states = states_matrix[this_set,:]
        for idx, t in progbar(enumerate(np.linspace(0.0, 1.0, n_time_samples)), size=n_time_samples, every=32):
            np.exp(-two_pi_i * m_block * t, out=phasor, where=where_mat)
            now_amps = np.matmul(phasor, these_states)
            resid_pops[:, idx] = 1.0 - np.sum(np.abs(np.conjugate(now_amps) * now_amps), axis=0)
        all_sq_residuals[out_idx] = np.sqrt(np.sum(np.square(np.abs(resid_pops)), axis=1)/n_time_samples)
    ##
    ##
    #m_block = (np.arange(-m_rad, m_rad+1) * -two_pi_i).reshape(-1,1,1)
    #site_amps = np.empty((n_sites, n_states), dtype=np.complex)
    #expand_states = np.empty((2*m_rad + 1, n_sites, n_states), dtype=np.complex)
    #for out_idx, this_set in progbar(enumerate(np.arange(0, n_sets, set_stride)), every=1, size=n_sets//set_stride):
    #    these_states = states_matrix[this_set,:].reshape(2*m_rad + 1, n_sites, -1)
    #    for idx, t in enumerate(np.linspace(0.0, 1.0, n_time_samples)):
    #        phasor = np.exp(m_block * t)
    #        np.multiply(phasor, these_states, out=expand_states)
    #        np.sum(expand_states, axis=0, out=site_amps)
    #        resid_pops[:, idx] = 1.0-np.sum(np.real(np.conjugate(site_amps) * site_amps), axis=0)
    #    all_raw_residuals[out_idx] = np.sum(np.abs(resid_pops), axis=1)/n_time_samples
    ##
del phasor, now_amps, resid_pops, m_block, these_states, where_mat


# In[ ]:


plt.plot(all_raw_residuals[0], ',k')
plt.plot(all_raw_residuals[-1], ',r')
plt.plot(all_sq_residuals[0], ',b')
plt.plot(all_sq_residuals[-1], ',g')
plt.xlim(0, n_sites*(2*m_rad+1))


# In[ ]:


get_ipython().run_cell_magic('time', '', 'with h5py.File(this_path, \'r\', libver="latest") as f:\n    print(os.path.basename(this_path))\n    states_matrix = f[\'floquet_state\']\n    m_rad = ((states_matrix.shape[-1]//n_sites) - 1)//2\n    n_states = states_matrix.shape[-1]\n    nm_s = states_matrix.shape[-1]//n_sites\n    test_sq_residuals = np.empty((n_sets//set_stride, n_states))\n    resid_pops = np.empty((n_states, n_time_samples))\n    ##\n    phasor = np.zeros((n_sites, (2*m_rad+1)*n_sites), dtype=np.complex)\n    where_mat = np.tile(np.diag(np.full(n_sites, True)), 2*m_rad+1)\n    m_block = np.kron(np.arange(-m_rad, m_rad+1), np.identity(n_sites)).astype(np.complex)\n    for out_idx, this_set in progbar(enumerate(np.arange(0, n_sets, set_stride)), every=1, size=n_sets//set_stride):\n        for group_idx in progbar(np.arange(n_sites)):\n            these_states = states_matrix[this_set,:,group_idx*nm_s:(group_idx+1)*nm_s]\n            #accum = - 2.0 * np.sum(np.square(np.abs(these_states[...])), axis=0, dtype=np.complex)\n            c = np.transpose(these_states.reshape(2*m_rad+1, n_sites, -1), axes=(2,1,0))\n            c_adj = np.conjugate(np.transpose(c, axes=(0,2,1)))\n            c_nm = np.matmul(c_adj, c)\n            del c, c_adj, these_states\n            accum = np.sum(np.diagonal(c_nm, 0, 1, 2)[:,None,:] * np.diagonal(c_nm, 0, 1, 2)[:,:,None], axis=(1,2))\n            for k in np.arange(1, 2*m_rad + 1):\n                accum += 2.0 * np.sum(np.diagonal(c_nm, k, 1, 2)[:,None,:] * np.diagonal(c_nm, -k, 1, 2)[:,:,None], axis=(1,2))\n            accum = np.sqrt(np.abs(accum - 1.0))\n            test_sq_residuals[out_idx, group_idx*nm_s:(group_idx+1)*nm_s] = accum\ndel c_nm')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'with h5py.File(this_path, \'r\', libver="latest") as f:\n    print(os.path.basename(this_path))\n    states_matrix = f[\'floquet_state\']\n    m_rad = ((states_matrix.shape[-1]//n_sites) - 1)//2\n    n_states = states_matrix.shape[-1]\n    nm_s = states_matrix.shape[-1]//n_sites\n    alllong_sq_residuals = np.empty((n_sets//set_stride, n_states))\n    resid_pops = np.empty((n_states, n_time_samples))\n    ##\n    phasor = np.zeros((n_sites, (2*m_rad+1)*n_sites), dtype=np.complex)\n    where_mat = np.tile(np.diag(np.full(n_sites, True)), 2*m_rad+1)\n    m_block = np.kron(np.arange(-m_rad, m_rad+1), np.identity(n_sites)).astype(np.complex)\n    for out_idx, this_set in progbar(enumerate(np.arange(0, n_sets, set_stride)), every=1, size=n_sets//set_stride):\n        for group_idx in progbar(np.arange(n_sites)):\n            these_states = states_matrix[this_set,:,group_idx*nm_s:(group_idx+1)*nm_s]\n            #accum = - 2.0 * np.sum(np.square(np.abs(these_states[...])), axis=0, dtype=np.complex)\n            c = np.transpose(these_states.reshape(2*m_rad+1, n_sites, -1), axes=(2,1,0)).astype(np.clongdouble)\n            c_adj = np.conjugate(np.transpose(c, axes=(0,2,1)))\n            c_nm = np.matmul(c_adj, c)\n            del c, c_adj, these_states\n            accum = np.sum(np.diagonal(c_nm, 0, 1, 2)[:,None,:] * np.diagonal(c_nm, 0, 1, 2)[:,:,None], axis=(1,2))\n            for k in np.arange(1, 2*m_rad + 1):\n                accum += 2.0 * np.sum(np.diagonal(c_nm, k, 1, 2)[:,None,:] * np.diagonal(c_nm, -k, 1, 2)[:,:,None], axis=(1,2))\n            accum = np.sqrt(np.abs(accum - 1.0))\n            alllong_sq_residuals[out_idx, group_idx*nm_s:(group_idx+1)*nm_s] = accum\ndel c_nm')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'with h5py.File(this_path, \'r\', libver="latest") as f:\n    print(os.path.basename(this_path))\n    states_matrix = f[\'floquet_state\']\n    m_rad = ((states_matrix.shape[-1]//n_sites) - 1)//2\n    n_states = states_matrix.shape[-1]\n    nm_s = states_matrix.shape[-1]//n_sites\n    justnm_sq_residuals = np.empty((n_sets//set_stride, n_states))\n    resid_pops = np.empty((n_states, n_time_samples))\n    ##\n    phasor = np.zeros((n_sites, (2*m_rad+1)*n_sites), dtype=np.complex)\n    where_mat = np.tile(np.diag(np.full(n_sites, True)), 2*m_rad+1)\n    m_block = np.kron(np.arange(-m_rad, m_rad+1), np.identity(n_sites)).astype(np.complex)\n    for out_idx, this_set in progbar(enumerate(np.arange(0, n_sets, set_stride)), every=1, size=n_sets//set_stride):\n        for group_idx in progbar(np.arange(n_sites)):\n            these_states = states_matrix[this_set,:,group_idx*nm_s:(group_idx+1)*nm_s]\n            #accum = - 2.0 * np.sum(np.square(np.abs(these_states[...])), axis=0, dtype=np.complex)\n            c = np.transpose(these_states.reshape(2*m_rad+1, n_sites, -1), axes=(2,1,0))\n            c_adj = np.conjugate(np.transpose(c, axes=(0,2,1)))\n            c_nm = np.matmul(c_adj, c).astype(np.clongdouble)\n            print(np.)\n            del c, c_adj, these_states\n            accum = np.sum(np.diagonal(c_nm, 0, 1, 2)[:,None,:] * np.diagonal(c_nm, 0, 1, 2)[:,:,None], axis=(1,2))\n            for k in np.arange(1, 2*m_rad + 1):\n                accum += 2.0 * np.sum(np.diagonal(c_nm, k, 1, 2)[:,None,:] * np.diagonal(c_nm, -k, 1, 2)[:,:,None], axis=(1,2))\n            accum = np.sqrt(np.abs(accum - 1.0))\n            justnm_sq_residuals[out_idx, group_idx*nm_s:(group_idx+1)*nm_s] = accum\ndel c_nm')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'with h5py.File(this_path, \'r\', libver="latest") as f:\n    print(os.path.basename(this_path))\n    states_matrix = f[\'floquet_state\']\n    m_rad = ((states_matrix.shape[-1]//n_sites) - 1)//2\n    n_states = states_matrix.shape[-1]\n    nm_s = states_matrix.shape[-1]//n_sites\n    real_sq_residuals = np.empty((n_sets//set_stride, n_states))\n    resid_pops = np.empty((n_states, n_time_samples))\n    ##\n    phasor = np.zeros((n_sites, (2*m_rad+1)*n_sites), dtype=np.complex)\n    where_mat = np.tile(np.diag(np.full(n_sites, True)), 2*m_rad+1)\n    m_block = np.kron(np.arange(-m_rad, m_rad+1), np.identity(n_sites)).astype(np.complex)\n    for out_idx, this_set in progbar(enumerate(np.arange(0, n_sets, set_stride)), every=1, size=n_sets//set_stride):\n        for group_idx in progbar(np.arange(n_sites)):\n            these_states = np.real(states_matrix[this_set,:,group_idx*nm_s:(group_idx+1)*nm_s]).astype(np.longdouble)\n            #accum = - 2.0 * np.sum(np.square(np.abs(these_states[...])), axis=0, dtype=np.complex)\n            c = np.transpose(these_states.reshape(2*m_rad+1, n_sites, -1), axes=(2,1,0))\n            c_adj = np.transpose(c, axes=(0,2,1))\n            c_nm = np.matmul(c_adj, c)\n            del c, c_adj, these_states\n            accum = np.sum(np.diagonal(c_nm, 0, 1, 2)[:,None,:] * np.diagonal(c_nm, 0, 1, 2)[:,:,None], axis=(1,2))\n            for k in np.arange(1, 2*m_rad + 1):\n                accum += 2.0 * np.sum(np.diagonal(c_nm, k, 1, 2)[:,None,:] * np.diagonal(c_nm, -k, 1, 2)[:,:,None], axis=(1,2))\n            accum = np.sqrt(np.abs(accum - 1.0))\n            real_sq_residuals[out_idx, group_idx*nm_s:(group_idx+1)*nm_s] = accum\ndel c_nm')


# In[ ]:


#%%time
with h5py.File(this_path, 'r', libver="latest") as f:
    print(os.path.basename(this_path))
    states_matrix = f['floquet_state']
    m_rad = ((states_matrix.shape[-1]//n_sites) - 1)//2
    n_states = states_matrix.shape[-1]
    nm_s = states_matrix.shape[-1]//n_sites
    mone_in_pieces = np.full((nm_s, 256*256), np.ldexp(-1, -16, dtype=np.longdouble))
    presub_sq_residuals = np.empty((n_sets//set_stride, n_states))
    resid_pops = np.empty((n_states, n_time_samples))
    c_nm = np.empty((nm_s, nm_s, nm_s), dtype=np.longdouble)
    ##
    phasor = np.zeros((n_sites, (2*m_rad+1)*n_sites), dtype=np.complex)
    where_mat = np.tile(np.diag(np.full(n_sites, True)), 2*m_rad+1)
    m_block = np.kron(np.arange(-m_rad, m_rad+1), np.identity(n_sites)).astype(np.complex)
    accum_slots = np.empty((nm_s,nm_s-1), dtype=np.longdouble)
    for out_idx, this_set in progbar(enumerate(np.arange(0, n_sets, set_stride)), every=1, size=n_sets//set_stride):
        for group_idx in progbar(np.arange(n_sites)):
            these_states = states_matrix[this_set,:,group_idx*nm_s:(group_idx+1)*nm_s].real
            #accum = - 2.0 * np.sum(np.square(np.abs(these_states[...])), axis=0, dtype=np.complex)
            c = np.transpose(these_states.reshape(2*m_rad+1, n_sites, -1), axes=(2,1,0)).astype(np.longdouble)
            c_adj = np.transpose(c, axes=(0,2,1))
            np.matmul(c_adj, c, out=c_nm)
            #del c, c_adj, these_states
            #main_diag_prog = (np.diagonal(c_nm, 0, 1, 2)[:,None,:] * np.diagonal(c_nm, 0, 1, 2)[:,:,None]).reshape(-1,nm_s*nm_s)
            #accum_slots[:, 0] = np.sum(np.append(main_diag_prog, mone_in_pieces, axis=1), axis=1)
            #accum_slots[:,0] = 0
            for k in np.arange(1, 2*m_rad + 1):
                accum_slots[:, k - 1] = np.sum(np.diagonal(c_nm, k, 1, 2)[:,None,:] * np.diagonal(c_nm, -k, 1, 2)[:,:,None], axis=(1,2))
            presub_sq_residuals[out_idx, group_idx*nm_s:(group_idx+1)*nm_s] = np.sqrt(2.0 * np.sum(accum_slots, axis=1))
del c_nm


# In[ ]:


#%%time
with h5py.File(this_path, 'r', libver="latest") as f:
    print(os.path.basename(this_path))
    states_matrix = f['floquet_state']
    m_rad = ((states_matrix.shape[-1]//n_sites) - 1)//2
    n_states = states_matrix.shape[-1]
    nm_s = states_matrix.shape[-1]//n_sites
    presub_sq_residuals = np.empty((n_sets//set_stride, n_states))
    accum_slots = np.empty((nm_s,nm_s), dtype=np.longdouble)
    c_nm = np.empty((nm_s, nm_s, nm_s), dtype=np.longdouble)
    for out_idx, this_set in progbar(enumerate(np.arange(0, n_sets, set_stride)), every=1, size=n_sets//set_stride):
        for group_idx in progbar(np.arange(n_sites)):
            these_states = states_matrix[this_set,:,group_idx*nm_s:(group_idx+1)*nm_s].real
            c = np.transpose(these_states.reshape(2*m_rad+1, n_sites, -1), axes=(2,1,0)).astype(np.longdouble)
            c_adj = np.transpose(c, axes=(0,2,1))
            np.matmul(c_adj, c, out=c_nm)
            accum_slots[:,0] = 0
            for k in np.arange(1, 2*m_rad + 1):
                accum_slots[:, k] = 2.0 * np.sum(np.matmul(np.diagonal(c_nm, k, 1, 2)[:,:,None], np.diagonal(c_nm, -k, 1, 2)[:,None,:]), axis=(1,2))
            accum = np.sqrt(np.abs(np.sum(accum_slots, axis=1)))
            presub_sq_residuals[out_idx, group_idx*nm_s:(group_idx+1)*nm_s] = accum
del c_nm, c, c_adj, these_states, accum_slots


# In[ ]:


#%%time
def sum_thing(n):
    return ((n+1)*(2*n*n + n))//6

with h5py.File(this_path, 'r', libver="latest") as f:
    print(os.path.basename(this_path))
    states_matrix = f['floquet_state']
    m_rad = ((states_matrix.shape[-1]//n_sites) - 1)//2
    n_states = states_matrix.shape[-1]
    nm_s = states_matrix.shape[-1]//n_sites
    wait_sq_residuals = np.empty((n_sets//set_stride, n_states))
    ##
    accum_slots = np.empty((n_sites, sum_thing(nm_s-1)), dtype=np.longdouble)
    c_nm = np.empty((n_sites, nm_s, nm_s), dtype=np.longdouble)
    for out_idx, this_set in progbar(enumerate(np.arange(0, n_sets, set_stride)), every=1, size=n_sets//set_stride):
        for group_idx in progbar(np.arange(nm_s)):
            these_states = states_matrix[this_set,:,group_idx*n_sites:(group_idx+1)*n_sites].real
            c = np.transpose(these_states.reshape(nm_s, n_sites, -1), axes=(2,1,0)).astype(np.longdouble)
            c_adj = np.transpose(c, axes=(0,2,1))
            np.matmul(c_adj, c, out=c_nm)
            for k in np.arange(1, 2*m_rad + 1):
                accum_slots[:, sum_thing(nm_s-1-(k)):sum_thing(nm_s-1-(k-1))] = np.matmul(np.diagonal(c_nm, k, 1, 2)[:,:,None], np.diagonal(c_nm, -k, 1, 2)[:,None,:]).reshape(-1, (nm_s-k)*(nm_s-k))
            accum = np.sqrt(np.abs(np.sum(accum_slots, axis=1)))
            wait_sq_residuals[out_idx, group_idx*n_sites:(group_idx+1)*n_sites] = np.sqrt(2.0 * np.abs(np.sum(accum_slots, axis=1)))
del c_nm, c, c_adj, these_states, accum_slots


# In[453]:


#%%time
def sum_thing(n):
    return ((n+1)*(2*n*n + n))//6

with h5py.File(this_path, 'r', libver="latest") as f:
    print(os.path.basename(this_path))
    states_matrix = f['floquet_state']
    m_rad = ((states_matrix.shape[-1]//n_sites) - 1)//2
    n_states = states_matrix.shape[-1]
    nm_s = states_matrix.shape[-1]//n_sites
    waitlong_sq_residuals = np.empty((n_sets//set_stride, n_states))
    ##
    accum_slots = np.empty((n_sites, sum_thing(nm_s-1)), dtype=np.longdouble)
    c_nm = np.empty((n_sites, nm_s, nm_s), dtype=np.longdouble)
    for out_idx, this_set in progbar(enumerate(np.arange(0, n_sets, set_stride)), every=1, size=n_sets//set_stride):
        for group_idx in progbar(np.arange(nm_s)):
            these_states = states_matrix[this_set,:,group_idx*n_sites:(group_idx+1)*n_sites].real
            c = np.transpose(these_states.reshape(nm_s, n_sites, -1), axes=(2,1,0)).astype(np.longdouble)
            c_adj = np.transpose(c, axes=(0,2,1))
            np.matmul(c_adj, c, out=c_nm)
            for k in np.arange(1, 2*m_rad + 1):
                accum_slots[:, sum_thing(nm_s-1-(k)):sum_thing(nm_s-1-(k-1))] = np.matmul(np.diagonal(c_nm, k, 1, 2)[:,:,None], np.diagonal(c_nm, -k, 1, 2)[:,None,:]).reshape(-1, (nm_s-k)*(nm_s-k))
            accum = np.sqrt(np.abs(np.sum(accum_slots, axis=1)))
            waitlong_sq_residuals[out_idx, group_idx*n_sites:(group_idx+1)*n_sites] = np.sqrt(2.0 * np.abs(np.sum(accum_slots, axis=1)))
del c_nm, c, c_adj, these_states, accum_slots


# In[454]:


#%%time
def sum_thing(n):
    return ((n+1)*(2*n*n + n))//6

with h5py.File(this_path, 'r', libver="latest") as f:
    print(os.path.basename(this_path))
    states_matrix = f['floquet_state']
    m_rad = ((states_matrix.shape[-1]//n_sites) - 1)//2
    n_states = states_matrix.shape[-1]
    nm_s = states_matrix.shape[-1]//n_sites
    wait_sq_residuals = np.empty((n_sets//set_stride, n_states))
    ##
    accum_slots = np.empty((n_sites, sum_thing(nm_s-1)))
    c_nm = np.empty((n_sites, nm_s, nm_s))
    for out_idx, this_set in progbar(enumerate(np.arange(0, n_sets, set_stride)), every=1, size=n_sets//set_stride):
        for group_idx in progbar(np.arange(nm_s)):
            these_states = states_matrix[this_set,:,group_idx*n_sites:(group_idx+1)*n_sites].real
            c = np.transpose(these_states.reshape(nm_s, n_sites, -1), axes=(2,1,0))
            c_adj = np.transpose(c, axes=(0,2,1))
            np.matmul(c_adj, c, out=c_nm)
            for k in np.arange(1, 2*m_rad + 1):
                accum_slots[:, sum_thing(nm_s-1-(k)):sum_thing(nm_s-1-(k-1))] = np.matmul(np.diagonal(c_nm, k, 1, 2)[:,:,None], np.diagonal(c_nm, -k, 1, 2)[:,None,:]).reshape(-1, (nm_s-k)*(nm_s-k))
            accum = np.sqrt(np.abs(np.sum(accum_slots, axis=1)))
            wait_sq_residuals[out_idx, group_idx*n_sites:(group_idx+1)*n_sites] = np.sqrt(2.0 * np.abs(np.sum(accum_slots, axis=1)))
del c_nm, c, c_adj, these_states, accum_slots


# In[ ]:


plt.plot(np.abs((all_sq_residuals[0,:]-test_sq_residuals[0,:])))
plt.plot(np.abs((all_sq_residuals[1,:]-test_sq_residuals[1,:])))


# In[ ]:


plt.semilogy(all_sq_residuals[1,:], ',', label="Sq, summed")
plt.semilogy(test_sq_residuals[1,:], ',', label="Sq, comps")
#plt.semilogy(all_raw_residuals[1,:], label="Abs, summed")
plt.semilogy(all_sq_residuals[0,:], ',', label="Sq, summed")
plt.semilogy(test_sq_residuals[0,:], ',', label="Sq, comps")
#plt.semilogy(all_raw_residuals[0,:], label="Abs, summed")
plt.legend()
plt.ylim(None, 1e-6)


# In[484]:


#plt.loglog(all_sq_residuals[0,:], test_sq_residuals[0,:], ',', label="std")
#plt.loglog(all_sq_residuals[0,:], alllong_sq_residuals[0,:], ',', label="all")
#plt.loglog(all_sq_residuals[0,:], justnm_sq_residuals[0,:], ',', label='nm')
plt.loglog(all_sq_residuals[0,:], all_sq_residuals[0,:], ',', label="ref")
#plt.loglog(all_sq_residuals[0,:], real_sq_residuals[0,:], '.', label='reallong')
plt.loglog(all_sq_residuals[0,:], waitlong_sq_residuals[0,:], ',', label='waitlong0')
plt.loglog(all_sq_residuals[1,:], waitlong_sq_residuals[1,:], ',', label='waitlong1')
plt.loglog(all_sq_residuals[0,:], wait_sq_residuals[0,:], ',', label='wait0')
plt.loglog(all_sq_residuals[1,:], wait_sq_residuals[1,:], ',', label='wait1')
#plt.loglog(wait_sq_residuals[0,:], waitlong_sq_residuals[0,:], ',', label='comp0')
#plt.loglog(wait_sq_residuals[1,:], waitlong_sq_residuals[1,:], ',', label='comp1')
#plt.loglog(all_sq_residuals[1,:], test_sq_residuals[1,:], '.', label="std")
#plt.loglog(all_sq_residuals[1,:], alllong_sq_residuals[1,:], '.', label="all")
#plt.loglog(all_sq_residuals[1,:], justnm_sq_residuals[1,:], '.', label='nm')
plt.axvline(1.2e-8, c='k', ls=':')
plt.axhline(1.2e-8, c='k', ls=':')
plt.ylim(None,1e-6)
plt.xlim(None,1e-6)
plt.legend()


# In[ ]:


sub_amount


# In[ ]:


np.sum(real_sq_residuals < 1e-6)


# In[ ]:


(3*60+41) + np.sum(real_sq_residuals < 1e-6)/np.sum(real_sq_residuals < 1000) * (10*60)


# In[ ]:


(10*60)


# In[456]:


((3*60+52) + np.sum(np.abs(waitlong_sq_residuals) < 5e-10)/np.sum(np.abs(waitlong_sq_residuals) < 1000) * (9.5*60))/60


# In[459]:


((1*60+43) + np.sum(np.abs(wait_sq_residuals) < 5e-8)/np.sum(np.abs(wait_sq_residuals) < 1000) * (9.5*60))/60


# In[479]:


((1*60+43) + np.sum(np.abs(wait_sq_residuals) < 1.5e-8)/np.sum(np.abs(wait_sq_residuals) < 1000) * (9.5*60))/60


# In[485]:


resid_file_path = os.path.join(base_path, 'critical_m100_all_sq_residuals.h5')
print(resid_file_path)
#raise RuntimeException("Don't overwrite things yet")
with h5py.File(resid_file_path, 'w', libver="latest") as sf:
    full_scan = sf.create_dataset('total_residuals', data=all_sq_residuals)
    full_scan.attrs['m_rad'] = m_rad
    full_scan.attrs['source_path'] = str(this_path)


# In[486]:


resid_file_path = os.path.join(base_path, 'critical_m100_all_fast_residuals.h5')
print(resid_file_path)
#raise RuntimeException("Don't overwrite things yet")
with h5py.File(resid_file_path, 'w', libver="latest") as sf:
    full_scan = sf.create_dataset('total_residuals', data=wait_sq_residuals)
    full_scan.attrs['m_rad'] = m_rad
    full_scan.attrs['source_path'] = str(this_path)


# In[483]:


largest_m_path

