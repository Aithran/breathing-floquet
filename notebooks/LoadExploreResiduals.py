#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import h5py
from matplotlib import pyplot as plt


# In[2]:


from zss_progbar import log_progress as progbar


# In[3]:


import os
import glob
import re


# In[4]:


base_dir = os.path.join('/media', 'simulationData', 'FromKhronos')


# In[5]:


m_re = re.compile(r'.*_m(\d+)_resid.h5')
def get_m(file_name):
    return int(m_re.search(file_name).group(1))
base_path = base_dir
data_dirs = [os.path.basename(dirpath[:-1]) for dirpath in glob.glob(base_path + "/*shallow*/")]
print(data_dirs)
#data_paths = [os.path.join(base_path, this_dir, this_dir + '_out.h5') for this_dir in data_dirs]
data_paths = sorted([os.path.join(base_path, this_dir, this_dir + '_resid.h5') for this_dir in data_dirs], key=get_m)
for idx, path in enumerate(data_paths):
    print('{}:\t{}'.format(idx, os.path.basename(path)))
largest_m_path = data_paths[3]


# In[6]:


from cycler import cycler
kelly_cycler = (cycler(color=['#FFB300', '#803E75', '#FF6800',
              '#A6BDD7', '#C10020', '#CEA262',
              '#817066', '#007D34']) + cycler(linestyle=['-', '--', ':', '-.', '-', '--', ':', '-.']))


# In[7]:


def load_resid_file(path):
    with h5py.File(path, 'r', libver='latest') as f:
        rough = f['rough_residuals'][:]
        refined = f['refined_residuals'][:]
        return rough, refined, f.attrs['source_path']


# In[14]:


#roughLess, refinedLess, sourceLess = load_resid_file(data_paths[0])
#print(sourceLess)
roughMore, refinedMore, sourceMore = load_resid_file(data_paths[3])
print(sourceMore)


# In[15]:


for i in np.arange(roughMore.shape[0]):
    plt.loglog(refinedMore[i], roughMore[i], ',')
plt.plot([1e-16,1], [1e-16,1], ':k')
plt.axhline(1.5e-8, ls=':', c='k')
plt.axvline(1.5e-8, ls=':', c='k')


# In[16]:


print(roughMore.shape)


# In[18]:


plt.plot(np.sum(refinedMore < 1e-13, axis=1)/45)
plt.axhline(90.44444)


# In[25]:


n_sites = m_rad = n_ms = n_states = 0
def do_shift(resids, source_path, resid_thresh=1e-13):
    global n_sites, m_rad, n_ms, n_states
    results_dict = dict()
    with h5py.File(source_path, 'r', libver='latest') as f:
        print(os.path.basename(source_path))
        n_sites = f.attrs['n_sites']
        m_rad = f.attrs['floquet_m_radius']
        n_ms = 2*m_rad + 1
        n_states = n_sites * n_ms
        state_matrix = f['floquet_state']
        energy_matrix = f['floquet_energy']
        all_drive_freqs = f['critical_drive'][:]
        n_sets = all_drive_freqs.size//2
        #print(np.arange(60,n_sets))
        #for set_idx in np.arange(60,n_sets):
        for set_idx in progbar(np.arange(60,n_sets)):
            print(set_idx)
            below_thresh = np.where(resids[set_idx] < resid_thresh)[0]
            stable_states = state_matrix[set_idx*2, :, below_thresh]
            shifted_states = np.zeros_like(stable_states)
            stable_quasi_en = energy_matrix[2*set_idx, below_thresh]
            this_drive = all_drive_freqs[2*set_idx]
            half_drive = 0.5*this_drive
            m, fbz_en = np.divmod(stable_quasi_en + 100, this_drive)
            #fbz_en = stable_quasi_en
            m = m.astype(np.int)
            fbz_en -= 100
            shifted_too_far_count = 0
            for state_idx in np.arange(len(below_thresh)):
                this_m = m[state_idx]
                m_x_n = this_m * n_sites
                if np.abs(this_m) > 2*m_rad:
                    # No pops can survive if it shifts this far
                    shifted_too_far_count += 1
                    continue
                dest_min_idx = 0 if this_m >= 0 else -m_x_n
                dest_max_idx = n_states if this_m <= 0 else n_states - m_x_n
                dest_slice = np.s_[dest_min_idx:dest_max_idx, state_idx]
                src_min_idx = 0 if this_m <= 0 else m_x_n
                src_max_idx = n_states if this_m >= 0 else n_states + m_x_n
                src_slice = np.s_[src_min_idx:src_max_idx, state_idx]
                shifted_states[dest_slice] = stable_states[src_slice]
            results_dict[set_idx] = {'fbz_en':fbz_en.copy(),
                                     'm':m.copy()*this_drive,
                                     'shifted_state':shifted_states.copy(),
                                     'too_far':shifted_too_far_count,
                                     'below_thresh_idx':below_thresh,
                                     'matching_resids':resids[set_idx, below_thresh]
                                    }
    #del shifted_states, resids
    return results_dict


# In[26]:


sourceMore = os.path.join('/media/simulationData/FromKhronos/', 'shallow_n45_lattice_m160', 'shallow_n45_lattice_m160_out.h5')
moreResults = do_shift(refinedMore, sourceMore, resid_thresh=1e-13)


# In[27]:


for this_idx in list(moreResults.keys()):
    plt.plot(this_idx, moreResults[this_idx]['too_far'],'.k')


# In[28]:


which_scan = 61
fqe = moreResults[which_scan]['fbz_en']
mshift = moreResults[which_scan]['m']
shifted_states = moreResults[which_scan]['shifted_state']


# In[29]:


print(n_states, n_sites)
with h5py.File(sourceMore, 'r', libver='latest') as f:
    gammas = f['scan_values'][:]
for set_idx in np.arange(len(list(moreResults.keys()))):
    fbzs = moreResults[set_idx]['fbz_en']
    shifts = moreResults[set_idx]['m']
    #plt.plot(np.full_like(fbzs, set_idx), fbzs-shifts, ',k')
    plt.semilogx(np.full_like(fbzs, gammas[set_idx]), fbzs, ',k')
    #plt.plot(np.full_like(fbzs, set_idx), fbzs+shifts, ',k')
plt.ylim(-50,50)


# In[30]:


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


def get_fbz_ens(shifted_results, overlap_thresh=0.9):
    all_keys = np.sort(list(shifted_results.keys()))
    state_size = shifted_results[all_keys[0]]['shifed_state'].shape[-1]
    overlaps = np.empty((state_size, state_size), dtype=np.complex)
    already_matched = set()
    last_i = n_unique = 0
    state_groups=list()
    group_starts=list()
    grouped_fqe = np.empty((state_size, 2)
    for this_set in all_keys:
        np.matmul(np.conjugate(shifted_states.T), shifted_states, out=overlaps)
        for ov_idx in np.arange(state_size):
            these_matches = np.where(np.abs(overlaps[ov_idx]) > overlap_thresh)[0]
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
                last_i = i+1


# In[31]:


try:
    del new_mat, new_overlaps, resorted, grouped_fqe
except NameError:
    pass
plt.figure(figsize=(6,6))
already_matched = set()
new_mat = np.zeros_like(overlaps)
last_i = 0
n_unique = 0
state_groups = list()
group_starts = list()
grouped_fqe=np.empty((overlaps.shape[0],2))
for ov_idx in np.arange(overlaps.shape[0]):
    these_matches = np.where(np.abs(overlaps[ov_idx]) > 0.9)[0]
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


# In[32]:


plt.plot(np.argsort(grouped_fqe[:,1]))


# In[33]:


mean_fqe = np.empty(n_unique)
stddev_fqe = np.empty(n_unique)
count_fqe = np.empty(n_unique)
for q in np.arange(n_unique):
    these = grouped_fqe[:,1][grouped_fqe[:,0] == q]
    mean_fqe[q] = np.mean(these)
    stddev_fqe[q] = np.std(these)
    count_fqe[q] = these.shape[0]
fqe_ord = np.argsort(mean_fqe)
plt.plot(mean_fqe[fqe_ord], '.', label='mean')
print(np.diff(mean_fqe[fqe_ord]))


# In[36]:


plt.plot(count_fqe,'.')
plt.ylim(0,None)
print(count_fqe.min(), np.sum(count_fqe)/45)


# In[38]:


plt.semilogy(np.diff(mean_fqe[fqe_ord]), label='Diff')
plt.semilogy(stddev_fqe[fqe_ord][:-2])
plt.semilogy(stddev_fqe[fqe_ord][1:])
plt.legend()


# In[39]:


plt.plot(1-np.diag(np.abs(new_overlaps)))


# In[40]:


print(group_starts)
import itertools
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)
print(new_overlaps.shape)
fig, ax = plt.subplots(1)
img = ax.imshow(np.abs(new_overlaps), vmax=1.0, vmin=0)
plt.colorbar(img)
from matplotlib import patches
for x, y in pairwise(np.append(group_starts, new_overlaps.shape[0])):
    ax.add_patch(patches.Rectangle((x-0.5, x-0.5), y-x, y-x, fill=False, ec='c', ls='--', lw=2))


# In[47]:


get_ipython().run_cell_magic('time', '', 'which_check = 37\nset_stride = 2\nthis_path = sourceMore\norig_idx = (moreResults[which_scan][\'below_thresh_idx\'])[state_groups[which_check]]\nwith h5py.File(this_path, \'r\', libver="latest") as f:\n    print(os.path.basename(this_path))\n    states_matrix = f[\'floquet_state\']\n    check_equiv = states_matrix[set_stride*which_scan,:,orig_idx]\nfig, ax_ar = plt.subplots(4)\nax_ar[0].plot(\n    np.sum(\n        np.repeat(np.arange(-m_rad,m_rad+1), n_sites)[:,np.newaxis]*\n               np.square(np.abs(check_equiv)), axis=0)/np.sum(np.square(np.abs(check_equiv)), axis=0), \'.\')\nax_ar[0].plot(\n    np.sum(\n        np.repeat(np.abs(np.arange(-m_rad,m_rad+1)), n_sites)[:,np.newaxis]*\n               np.square(np.abs(check_equiv)), axis=0), \'.\')\nax_ar[1].plot(moreResults[which_scan][\'m\'][state_groups[which_check]], \'.\')\nax_ar[2].semilogy(moreResults[which_scan][\'matching_resids\'][state_groups[which_check]], \'.\')\nmin_idx = np.argmin(moreResults[which_scan][\'matching_resids\'][state_groups[which_check]])\n#ax_ar[3].plot(np.tile(np.arange(-n_sites//2+1,n_sites//2+1), 2*m_rad+1), np.square(np.abs(np.sum(check_equiv[:,min_idx].reshape(n_sites, -1),axis=0))), \'.\')\nax_ar[3].plot(np.arange(-n_sites//2+1,n_sites//2+1),\n              np.square(np.abs(np.sum(check_equiv[:,min_idx].reshape(-1, n_sites), axis=0))),\n              \'.\')\nprint(check_equiv.shape, min_idx, moreResults[which_scan][\'matching_resids\'][state_groups[which_check]][min_idx])')


# In[ ]:




