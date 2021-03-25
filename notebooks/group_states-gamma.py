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
def kronos_to_local(kronos_path):
    return kronos_path.replace('/home/zach/floquet_simulations/remote_dropbox/','/media/simulationData/FromKhronos/')


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
largest_m_path = data_paths[0]


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
        return rough, refined, kronos_to_local(f.attrs['source_path'])


# In[8]:


roughLess, refinedLess, sourceLess = load_resid_file(data_paths[0])
print(sourceLess)
#roughMore, refinedMore, sourceMore = load_resid_file(data_paths[3])
#print(sourceMore)


# In[9]:


fig, ax_arr = plt.subplots(1,2, sharex=True,sharey=True)
#for i in np.arange(roughMore.shape[0]):
#    ax_arr[1].loglog(refinedMore[i], roughMore[i], ',r')
for i in np.arange(roughLess.shape[0]):
    ax_arr[0].loglog(refinedLess[i], roughLess[i], ',r')
for ax in ax_arr:
    ax.plot([1e-15,1.5e-8], [1e-15,1.5e-8], ':k')
    ax.axhline(1.5e-8, ls=':', c='k')
    ax.axvline(1.5e-8, ls=':', c='k')
ax_arr[0].set_xlim(1e-18,1)


# In[10]:


print(roughMore.shape, roughLess.shape)


# In[11]:


#plt.plot(np.sum(refinedMore < 1e-13, axis=1)/45)
plt.plot(np.sum(refinedLess < 1e-13, axis=1)/33)


# In[12]:


def check_overlaps(states, ms, n_sites):
    shifted = np.empty_like(states)
    for state_idx in np.arange(ms.size):
        this_m = ms[state_idx]
        m_x_n = this_m * n_sites
        if np.abs(this_m) > 2*m_rad:
            # No pops can survive if it shifts this far
            shifted_too_far_count += 1
            continue
        if this_m >= 0:
            shifted[:n_states-m_x_n, state_idx] = states[m_x_n:n_states, state_idx]
            shifted[n_states-m_x_n:, state_idx] = 0
        else:
            shifted[-m_x_n:n_states, state_idx] = states[:n_states+m_x_n, state_idx]
            shifted[:-m_x_n, state_idx] = 0
        #dest_min_idx = 0 if this_m >= 0 else -m_x_n
        #dest_max_idx = n_states if this_m <= 0 else n_states - m_x_n
        #dest_slice = np.s_[dest_min_idx:dest_max_idx, state_idx]
        #src_min_idx = 0 if this_m <= 0 else m_x_n
        #src_max_idx = n_states if this_m >= 0 else n_states + m_x_n
        #src_slice = np.s_[src_min_idx:src_max_idx, state_idx]
        #shifted[dest_slice] = states[src_slice]
    overlap = np.abs(np.matmul(np.conj(shifted.T), shifted[:,0]))
    return (overlap > 0.5)


# In[13]:


def find_fbz_qes(resids, source_path, resid_thresh=5e-14, diff_split=1e-6,
                 stride=1, offset=0):
    global n_sites, m_rad, n_ms, n_states
    with h5py.File(source_path, 'r', libver='latest') as f:
        print(os.path.basename(source_path))
        n_sites = f.attrs['n_sites']
        m_rad = f.attrs['floquet_m_radius']
        n_ms = 2*m_rad + 1
        n_states = n_sites * n_ms
        state_matrix = f['floquet_state']
        energy_matrix = f['floquet_energy']
        drive_freq = f.attrs['drive_freq']
        #try:
        #    all_drive_freqs = f['critical_drive'][:]
        #except KeyError:
        #    all_drive_freqs = f['critcal_drive'][:]
        n_sets = 49
        group_idx = np.full((n_sets, n_states), -1, dtype=np.int8)
        all_fqe_means = np.empty((n_sets, n_sites))
        all_fqe_stdev = np.empty_like(all_fqe_means)
        all_fqe_count = np.empty_like(all_fqe_means)
        all_fqe_minmax = np.empty((n_sets, n_sites, 2))
        for set_idx in progbar(np.arange(0,n_sets)):
            below_thresh = np.where(resids[set_idx] < resid_thresh)[0]
            #stable_states = state_matrix[stride*set_idx+offset, :, below_thresh]
            stable_quasi_en = energy_matrix[stride*set_idx+offset, below_thresh]
            this_drive = drive_freq
            half_drive = 0.5*this_drive
            m, fbz_en = np.divmod(stable_quasi_en + half_drive, this_drive)
            m = m.astype(np.int)
            fbz_en -= half_drive
            order = np.argsort(fbz_en)
            sorted_en = fbz_en[order]
            split_spots = np.argwhere(np.diff(sorted_en) > diff_split)[:,0]+1
            group_list = np.split(sorted_en, split_spots)
            order_lists = np.split(order, split_spots)
            if set_idx == 0:
                plt.semilogy(np.diff(sorted_en), '.r')
                plt.axhline(diff_split, c='grey', ls=':', lw=1)
            n_deg = 0
            for idx in np.arange(min(n_sites, len(order_lists))):
                try:
                    resorted_idx = np.sort(order_lists[idx])
                    reorder_idx = np.argsort(order_lists[idx])
                    out_idx = idx + n_deg
                    if (np.unique(m[order_lists[idx]]).size != order_lists[idx].size):
                        # Assume only two states for now
                        #print("Likely degeneracy")
                        stable_states = state_matrix[stride*set_idx+offset, :, below_thresh[resorted_idx]]
                        two_sets = check_overlaps(stable_states, m[resorted_idx], n_sites)
                        # First set
                        deg_set = resorted_idx[two_sets]
                        deg_group = group_list[idx][reorder_idx][two_sets]
                        group_idx[set_idx, below_thresh[deg_set]] = out_idx
                        all_fqe_means[set_idx, out_idx] = deg_group.mean()
                        all_fqe_stdev[set_idx, out_idx] = deg_group.std()
                        all_fqe_minmax[set_idx, out_idx, 0] = deg_group.min()
                        all_fqe_minmax[set_idx, out_idx, 1] = deg_group.max()
                        all_fqe_count[set_idx, out_idx] = deg_group.shape[0]
                        # Second set
                        n_deg += 1
                        out_idx = idx + n_deg
                        deg_set = resorted_idx[np.logical_not(two_sets)]
                        deg_group = group_list[idx][reorder_idx][np.logical_not(two_sets)]
                        group_idx[set_idx, below_thresh[deg_set]] = out_idx
                        all_fqe_means[set_idx, out_idx] = deg_group.mean()
                        all_fqe_stdev[set_idx, out_idx] = deg_group.std()
                        all_fqe_minmax[set_idx, out_idx, 0] = deg_group.min()
                        all_fqe_minmax[set_idx, out_idx, 1] = deg_group.max()
                        all_fqe_count[set_idx, out_idx] = deg_group.shape[0]
                    else:
                        group_idx[set_idx, below_thresh[order_lists[idx]]] = out_idx
                        all_fqe_means[set_idx, out_idx] = group_list[idx].mean()
                        all_fqe_stdev[set_idx, out_idx] = group_list[idx].std()
                        all_fqe_minmax[set_idx, out_idx, 0] = group_list[idx].min()
                        all_fqe_minmax[set_idx, out_idx, 1] = group_list[idx].max()
                        all_fqe_count[set_idx, out_idx] = group_list[idx].shape[0]
                except:
                    print(idx, set_idx, out_idx, n_deg)
                    plt.semilogy(np.diff(sorted_en), '.k')
                    raise
            print("Found {} states with {} pairs of degenerate states".format(out_idx+1, n_deg))
        plt.semilogy(np.diff(sorted_en), '.k')
    return all_fqe_means, all_fqe_stdev, all_fqe_minmax, all_fqe_count, group_idx


# In[14]:


plt.semilogy(np.diff(sorted_en), '.k')


# In[15]:


fbz_n33_mean, fbz_n33_dev, fbz_n33_minmax, fbz_n33_count, n33_groups = find_fbz_qes(refinedLess, sourceLess, resid_thresh=1e-13, diff_split=10, stride=1)
plt.ylim(None,1e-4)


# In[293]:


(offset, stride) = 0, 2
with h5py.File(sourceMore, 'r', libver='latest') as f:
    print(os.path.basename(sourceMore))
    gammas = f['scan_values'][offset::stride]
    try:
        omegas = f['critical_drive'][offset::stride]
    except KeyError:
        omegas = f['critcal_drive'][offset::stride]
(offset, stride) = 0, 1
with h5py.File(sourceLess, 'r', libver='latest') as f:
    print(os.path.basename(sourceLess))
    n33_gammas = f['scan_values'][offset::stride]
    try:
        n33_omegas = f['critical_drive'][offset::stride]
    except KeyError:
        n33_omegas = f['critcal_drive'][offset::stride]


# In[16]:


plt.semilogy(fbz_dev[:,:])
#plt.semilogy(np.abs(fbz_minmax[:,:,0]-fbz_mean[:,:]), ls="--")
#plt.semilogy(np.abs(fbz_minmax[:,:,1]-fbz_mean[:,:]), ls="--")
plt.plot(np.diff(fbz_mean[:,:],axis=1), ls='--')
plt.axhline(2e-6, c='k')
plt.ylim(None,None)


# In[17]:


plt.semilogy(fbz_n33_dev[:,:])
#plt.semilogy(np.abs(fbz_minmax[:,:,0]-fbz_mean[:,:]), ls="--")
#plt.semilogy(np.abs(fbz_minmax[:,:,1]-fbz_mean[:,:]), ls="--")
plt.plot(np.diff(fbz_n33_mean[:,:],axis=1), ls='--')
plt.axhline(2e-6, c='k')
plt.ylim(None,None)


# In[18]:


plt.imshow(groups.T[5500:8500].T, aspect=20, vmin=0, vmax=45)
#plt.colorbar()


# In[19]:


plt.imshow(n33_groups.T, aspect=1/25, vmin=0, vmax=33)
plt.colorbar()


# In[20]:


np.all([np.equal(np.unique(groups[i]), np.arange(-1, 45)) for i in np.arange(64)]), np.all([np.equal(np.unique(n33_groups[i]), np.arange(-1, 33)) for i in np.arange(256)])


# In[21]:


plt.plot(fbz_count[:,:], 'k', lw=1);
plt.plot(fbz_n33_count[:,:], 'r', lw=1);


# In[22]:


fig, ax_arr = plt.subplots(1,2, sharey=True)
ax_arr[1].semilogx(gammas[::], fbz_mean[:,:], 'k', lw=1);
ax_arr[0].semilogx(n33_gammas[::], fbz_n33_mean[:,:], 'k', lw=1);


# In[23]:


plt.semilogx(gammas[::], fbz_mean[:,:], 'b', lw=1);
plt.semilogx(n33_gammas[::], fbz_n33_mean[:,:], 'k', lw=1);


# In[24]:


fig, ax_arr = plt.subplots(1,2, sharey=True)
ax_arr[1].semilogx(omegas[::], fbz_mean[:,:], 'k', lw=1);
ax_arr[0].semilogx(n33_omegas[::], fbz_n33_mean[:,:], 'k', lw=1);
plt.ylim(-20,20)


# In[25]:


plt.semilogx(omegas[::], fbz_mean[:,:], 'b', lw=1);
plt.semilogx(n33_omegas[::], fbz_n33_mean[:,:], 'k', lw=1);


# In[26]:


fig, ax_arr = plt.subplots(1,2, sharey=True)
ax_arr[1].semilogx(gammas[::], fbz_mean[:,:], 'k', lw=1);
ax_arr[0].semilogx(n33_gammas[::], fbz_n33_mean[:,:], 'k', lw=1);
ax_arr[0].set_ylim(-20, 20)


# In[27]:


plt.semilogx(gammas[::], fbz_mean[:,:], 'b', lw=1);
plt.semilogx(n33_gammas[::], fbz_n33_mean[:,:], 'k', lw=1);
plt.ylim(-20, 20)


# In[28]:


plt.semilogx(gammas[::], fbz_mean[:,:], 'b', lw=1);
plt.semilogx(n33_gammas[::], fbz_n33_mean[:,:], 'k', lw=1);
plt.ylim(-8, 8)


# ## End of energy-level diagrams
# 
# Now we'll try to do some states stuff.
# Let's do stroboscopic first, 'cause it is easier

# # n=33

# In[29]:


idx_choice = np.empty(33, dtype=np.int)
for state_idx in np.arange(33):
    zz = np.nonzero(n33_groups[0] == state_idx)[0]
    minchoice = np.argmin(refinedLess[0, zz])
    idx_choice[state_idx] = zz[minchoice]
refinedLess[0, idx_choice]


# In[32]:


#%%time
which_set = 15
sorted_choices = np.sort(idx_choice)
print(sorted_choices)
with h5py.File(sourceLess, 'r', libver='latest') as f:
    print(os.path.basename(sourceLess))
    n_sites = f.attrs['n_sites']
    m_rad = f.attrs['floquet_m_radius']
    n_ms = 2*m_rad + 1
    n_states = n_sites * n_ms
    chosen_states = f['floquet_state'][which_set, :, sorted_choices]
    chosen_quasi = f['floquet_energy'][which_set, sorted_choices]
    chosen_gamma = f['scan_values'][which_set]
    try:
        chosen_drive = f['critical_drive'][which_set]
    except KeyError:
        chosen_drive = f['critcal_drive'][which_set]
#plt.semilogx(gammas[::], fbz_mean[:,:], 'r', lw=1);
sub_mean_quasi = np.sort(fbz_n33_mean[which_set])
sorted_wrapped_quasi = np.sort(np.mod(chosen_quasi + 0.5*chosen_drive, chosen_drive)-0.5*chosen_drive)
plt.plot(sub_mean_quasi-sorted_wrapped_quasi, '.')


# In[33]:


fig, (ax0, ax1) = plt.subplots(1,2)
quasi_order = np.argsort(np.mod(chosen_quasi + 0.5*chosen_drive, chosen_drive)-0.5*chosen_drive)
mags = np.real_if_close(np.sum(chosen_states.reshape(-1,33,33), axis=0))
max_mag = np.amax(np.abs(mags))
mag = ax0.imshow(mags[:,quasi_order].T, cmap='RdBu', vmin=-max_mag, vmax=max_mag, origin='bottom')
abssq = ax1.imshow(np.square(mags)[:,quasi_order].T, cmap='inferno', origin='bottom')
cbmag = fig.add_axes([0.05, 0.05, 0.42, 0.05])
cbabssq = fig.add_axes([0.55, 0.05, 0.42, 0.05])
plt.colorbar(abssq, cbabssq, orientation='horizontal')
plt.colorbar(mag, cbmag, orientation='horizontal')
fig.tight_layout()


# In[960]:


plt.imshow(np.log(np.matmul(mags.T, mags)), vmin=-16)
plt.colorbar()


# In[961]:


ordered_quasi = (np.mod(chosen_quasi + 0.5*chosen_drive, chosen_drive)-0.5*chosen_drive)[quasi_order]
for i in np.arange(33):
    plt.plot(ordered_quasi[i] + 4*mags[:,quasi_order][:,i])


# # Wannier States

# In[962]:


er_scale = 573.320792 # Hz*um^2, needs spacing^2 divided through
def get_er_and_spacing(depth, gamma, spacing):
    v0 = depth*(1+0.5*gamma*gamma)
    a_eff = spacing/(1+0.5*gamma*gamma)
    return a_eff*a_eff*v0/er_scale, a_eff
    
def wannier_state(depth, gamma, spacing):
    v0 = depth*(1+0.5*gamma*gamma)
    a_eff = spacing/(1+0.5*gamma*gamma)
    k_l_eff = (1+0.5*gamma*gamma)*np.pi/spacing
    print(v0, k_l_eff, (spacing, a_eff), spacing*spacing*v0/er_scale)


# In[963]:


with h5py.File(sourceLess, 'r', libver='latest') as f:
    ver, a = get_er_and_spacing(f.attrs['lattice_depth'],
                  f['scan_values'][which_set],
                  f.attrs['site_spacing_um'])


# In[964]:


n_qs = 1024 + 1
k_rad =  50
n_ks = 2*k_rad + 1
all_qs = np.linspace(-1, 1, n_qs, endpoint=True)
v0_vals = np.empty((n_qs, n_ks))
v0_vecs = np.empty((n_qs, n_ks, n_ks))
v0_qs = np.empty((n_qs*n_ks))
v0_wp = np.empty_like(v0_qs, dtype=np.complex)
for idx, q in enumerate(all_qs):
    these_kparts = (2*np.arange(-k_rad, k_rad+1)) + q
    v0_qs[idx::n_qs] = these_kparts
    vlattice = np.diag(np.full(2*k_rad, -ver/4), k=1) + np.diag(np.full(2*k_rad, -ver/4), k=-1)
    qlattice = np.diag(np.square((2*np.arange(-k_rad, k_rad+1)) + q) )
    vals, vecs = np.linalg.eigh(vlattice + qlattice)
    order = np.argsort(vals)
    v0_vals[idx] = vals[order]
    v0_vecs[idx] = vecs[:, order]
    v0_wp[idx::n_qs] = v0_vecs[idx][:,0]
#n_xs = 2048+1
#x_range = 20.5
#x_s = np.linspace(-x_range-1, x_range+1, n_xs, endpoint=True).reshape(-1,1)
#kxs = np.exp(np.complex(0, np.pi)*(v0_qs.ravel().reshape(1,-1))*x_s)
#print(x_s.shape, v0_vecs[:,:,0].shape)
#wannier_funct = np.sum((((np.sign(v0_vecs[:,50,0][:,np.newaxis])*v0_vecs[:,:,0]).ravel().reshape(1,-1))*kxs), axis=1)/np.sqrt(n_qs)
#plt.plot(x_s, wannier_funct, 'k')
#plt.plot(x_s-1, wannier_funct, ':k')
#plt.plot(x_s+1, wannier_funct, ':k')
#plt.xlim(-x_range, x_range)
direct_wannier_p = v0_wp/np.sqrt((v0_qs[1]-v0_qs[0])*np.sum(np.square(np.abs(v0_wp))))
subrange = np.where(np.abs(v0_qs)< 10)
print(subrange[0].size)
sub_ks = np.pi/a*v0_qs[subrange]
sub_wp = direct_wannier_p[subrange]
plt.plot(v0_qs[subrange], np.square(np.abs(direct_wannier_p))[subrange], ',', label='direct')
xs = np.linspace(-4.0, 4.0, 513, endpoint=True)
direct_wannier_x = np.sum(direct_wannier_p[np.newaxis,:] * np.exp(complex(0, np.pi)*xs[:,np.newaxis]*v0_qs[np.newaxis,:]), axis=1)


# In[965]:


plt.plot(xs, direct_wannier_x)


# In[966]:


shift_one = np.exp(complex(0, np.pi)*-1.0*v0_qs[np.newaxis,:])
shift_two = np.exp(complex(0, np.pi)*2.0*v0_qs[np.newaxis,:])
shift_mone = np.exp(complex(0, np.pi)*1.0*v0_qs[np.newaxis,:])
direct_wannier_s1_x = np.sum(direct_wannier_p[np.newaxis,:] * shift_one * np.exp(complex(0, np.pi)*xs[:,np.newaxis]*v0_qs[np.newaxis,:]), axis=1)
direct_wannier_sm1_x = np.sum(direct_wannier_p[np.newaxis,:] * shift_mone * np.exp(complex(0, np.pi)*xs[:,np.newaxis]*v0_qs[np.newaxis,:]), axis=1)
plt.plot(xs, direct_wannier_s1_x)
plt.plot(xs, direct_wannier_sm1_x)
plt.plot(xs, direct_wannier_x)
plt.axvline(-1)
plt.axvline(0)
plt.axvline(1)


# In[967]:


wp_all = sub_wp[np.newaxis,:] * np.exp(complex(0, 1)*sub_ks[np.newaxis,:]*np.arange(-(n_sites//2),n_sites//2+1)[:,np.newaxis])


# In[968]:


plt.imshow(wp_all.real, aspect=100, cmap='RdBu')


# In[969]:


plt.imshow(wp_all.imag, aspect=100, cmap='RdBu')


# In[970]:


plt.imshow(np.square(np.abs(wp_all)), aspect=100)


# In[971]:


for i in np.arange(n_sites):
    plt.plot(sub_ks, ordered_quasi[i] + np.square(np.abs(np.sum(mags[:,quasi_order][:,i]*wp_all.T, axis=1))))
    plt.annotate(i, (4 + 0.5*(i%4), ordered_quasi[i]))
plt.xlim(-6, 6)
#plt.ylim(-20,20)


# In[972]:


s_list = np.arange(-(n_sites//2),n_sites//2+1)
smooth_set = set([0,1,2,5,10,22,27,30,31,32])
wiggle_set = set([3,4,6,7,11,12,20,21,25,26,28,29])
other_set = set([8,9,16,23,24])
rem_set = set(np.arange(n_sites)) - (smooth_set | wiggle_set | other_set)


# In[973]:


fig, (ax_p, ax_s, ax_sq) = plt.subplots(1,3, sharey=True, figsize=(8,3))
for i in smooth_set:
    ax_p.plot(sub_ks, ordered_quasi[i] + np.square(np.abs(np.sum(mags[:,quasi_order][:,i]*wp_all.T, axis=1))))
    ax_p.annotate(i, (4 + 0.4*(i%4), ordered_quasi[i]))
    ax_s.plot(s_list, ordered_quasi[i] + 4*mags[:,quasi_order][:,i])
    ax_s.annotate(i, (n_sites//2 - 4 + (i%2), ordered_quasi[i]))
    ax_sq.plot(s_list, ordered_quasi[i] + 10*np.square(np.abs(mags[:,quasi_order][:,i])))
    ax_sq.annotate(i, (n_sites//2 - 4 + (i%2), ordered_quasi[i]))
ax_p.set_xlim(-6,6)
ax_s.set_xlim(s_list[0],s_list[-1])
ax_p.set_xlabel('p-ish')
ax_s.set_xlabel('site index')
ax_sq.set_xlabel('site index')
ax_sq.set_xlim(s_list[0],s_list[-1])
fig.tight_layout()


# In[974]:


fig, (ax_p, ax_s, ax_sq) = plt.subplots(1,3, sharey=True, figsize=(8,3))
for i in wiggle_set:
    ax_p.plot(sub_ks, ordered_quasi[i] + np.square(np.abs(np.sum(mags[:,quasi_order][:,i]*wp_all.T, axis=1))))
    ax_p.annotate(i, (4 + 0.4*(i%4), ordered_quasi[i]))
    ax_s.plot(s_list, ordered_quasi[i] + 4*mags[:,quasi_order][:,i])
    ax_s.annotate(i, (n_sites//2 - 4 + (i%2), ordered_quasi[i]))
    ax_sq.plot(s_list, ordered_quasi[i] + 10*np.square(np.abs(mags[:,quasi_order][:,i])))
    ax_sq.annotate(i, (n_sites//2 - 4 + (i%2), ordered_quasi[i]))
ax_p.set_xlim(-6,6)
ax_s.set_xlim(s_list[0],s_list[-1])
ax_p.set_xlabel('p-ish')
ax_s.set_xlabel('site index')
ax_sq.set_xlabel('site index')
ax_sq.set_xlim(s_list[0],s_list[-1])
fig.tight_layout()


# In[975]:


fig, (ax_p, ax_s, ax_sq) = plt.subplots(1,3, sharey=True, figsize=(8,3))
for i in other_set:
    ax_p.plot(sub_ks, ordered_quasi[i] + np.square(np.abs(np.sum(mags[:,quasi_order][:,i]*wp_all.T, axis=1))))
    ax_p.annotate(i, (4 + 0.4*(i%4), ordered_quasi[i]))
    ax_s.plot(s_list, ordered_quasi[i] + 4*mags[:,quasi_order][:,i])
    ax_s.annotate(i, (n_sites//2 - 4 + (i%2), ordered_quasi[i]))
    ax_sq.plot(s_list, ordered_quasi[i] + 10*np.square(np.abs(mags[:,quasi_order][:,i])))
    ax_sq.annotate(i, (n_sites//2 - 4 + (i%2), ordered_quasi[i]))
ax_p.set_xlim(-6,6)
ax_s.set_xlim(s_list[0],s_list[-1])
ax_p.set_xlabel('p-ish')
ax_s.set_xlabel('site index')
ax_sq.set_xlabel('site index')
ax_sq.set_xlim(s_list[0],s_list[-1])
fig.tight_layout()


# In[976]:


fig, (ax_p, ax_s, ax_sq) = plt.subplots(1,3, sharey=True, figsize=(8,3))
for i in rem_set:
    ax_p.plot(sub_ks, ordered_quasi[i] + np.square(np.abs(np.sum(mags[:,quasi_order][:,i]*wp_all.T, axis=1))))
    ax_p.annotate(i, (4 + 0.4*(i%4), ordered_quasi[i]))
    ax_s.plot(s_list, ordered_quasi[i] + 4*mags[:,quasi_order][:,i])
    ax_s.annotate(i, (n_sites//2 - 4 + (i%2), ordered_quasi[i]))
    ax_sq.plot(s_list, ordered_quasi[i] + 10*np.square(np.abs(mags[:,quasi_order][:,i])))
    ax_sq.annotate(i, (n_sites//2 - 4 + (i%2), ordered_quasi[i]))
ax_p.set_xlim(-6,6)
ax_s.set_xlim(s_list[0],s_list[-1])
ax_p.set_xlabel('p-ish')
ax_s.set_xlabel('site index')
ax_sq.set_xlabel('site index')
ax_sq.set_xlim(s_list[0],s_list[-1])
fig.tight_layout()


# In[977]:


heff = np.matmul(mags[:,quasi_order], np.matmul(np.diag(ordered_quasi), mags[:,quasi_order].T))
maxmag = np.amax(np.abs(heff))
plt.imshow(heff, cmap="PuOr", vmin=-maxmag, vmax=maxmag)
plt.colorbar()


# In[979]:


from scipy.special import jv, jn_zeros
scale = jn_zeros(0,3)[-1]/21
plt.plot((s_list[1:]-0.5),-np.diag(heff, k=1), label="Tunneling, +1")
plt.plot((s_list[1:]-0.5),-np.diag(heff, k=-1), label="Tunneling, -1")
plt.plot((s_list[1:]-0.5),25*jv(0, (s_list[1:]-0.5)*scale), label="Bessel J_0")
plt.axvline(0)
plt.legend()


# In[984]:


def j0(x, x_scale, y_scale):
    return jv(0, scale*x)
from scipy import optimize
(fit_scale, fit_amp), _ =optimize.curve_fit(j0, (s_list[1:]-0.5), -np.diag(heff, k=1), (jn_zeros(0,3)[-1]/21, np.abs(np.diag(heff, k=1)).max()))
plt.plot((s_list[1:]-0.5),-np.diag(heff, k=1), label="Tunneling, +1")
plt.plot((s_list[1:]-0.5),-np.diag(heff, k=-1), label="Tunneling, -1")
plt.plot((s_list[1:]-0.5),fit_amp*jv(0, (s_list[1:]-0.5)*fit_scale), label="Bessel J_0")
plt.axvline(0)
plt.legend()
print(fit_scale, fit_amp)


# ### n=45

# In[985]:


idx_choice = np.empty(45, dtype=np.int)
for state_idx in np.arange(45):
    zz = np.nonzero(groups[0] == state_idx)[0]
    minchoice = np.argmin(refinedMore[0, zz])
    idx_choice[state_idx] = zz[minchoice]
refinedMore[0, idx_choice]


# In[986]:


get_ipython().run_cell_magic('time', '', "which_set = 32\nsorted_choices = np.sort(idx_choice)\nprint(sorted_choices)\nwith h5py.File(sourceMore, 'r', libver='latest') as f:\n    print(os.path.basename(sourceMore))\n    n_sites = f.attrs['n_sites']\n    m_rad = f.attrs['floquet_m_radius']\n    n_ms = 2*m_rad + 1\n    n_states = n_sites * n_ms\n    chosen_states = f['floquet_state'][which_set, :, sorted_choices]\n    chosen_quasi = f['floquet_energy'][which_set, sorted_choices]\n    chosen_gamma = f['scan_values'][which_set]\n    try:\n        chosen_drive = f['critical_drive'][which_set]\n    except KeyError:\n        chosen_drive = f['critcal_drive'][which_set]\n#plt.semilogx(gammas[::], fbz_mean[:,:], 'r', lw=1);\nsub_mean_quasi = np.sort(fbz_mean[which_set])\nsorted_wrapped_quasi = np.sort(np.mod(chosen_quasi + 0.5*chosen_drive, chosen_drive)-0.5*chosen_drive)\nplt.plot(sub_mean_quasi-sorted_wrapped_quasi, '.')")


# In[987]:


fig, (ax0, ax1) = plt.subplots(1,2)
quasi_order = np.argsort(np.mod(chosen_quasi + 0.5*chosen_drive, chosen_drive)-0.5*chosen_drive)
mags = np.real_if_close(np.sum(chosen_states.reshape(-1,45,45), axis=0))
max_mag = np.amax(np.abs(mags))
mag = ax0.imshow(mags[:,quasi_order].T, cmap='RdBu', vmin=-max_mag, vmax=max_mag, origin='bottom')
abssq = ax1.imshow(np.square(mags)[:,quasi_order].T, cmap='inferno', origin='bottom')
cbmag = fig.add_axes([0.05, 0.05, 0.42, 0.05])
cbabssq = fig.add_axes([0.55, 0.05, 0.42, 0.05])
plt.colorbar(abssq, cbabssq, orientation='horizontal')
plt.colorbar(mag, cbmag, orientation='horizontal')
fig.tight_layout()


# In[988]:


plt.imshow(np.log(np.matmul(mags.T, mags)), vmin=-16)
plt.colorbar()


# In[989]:


ordered_quasi = (np.mod(chosen_quasi + 0.5*chosen_drive, chosen_drive)-0.5*chosen_drive)[quasi_order]
for i in np.arange(n_sites):
    plt.plot(ordered_quasi[i] + 4*mags[:,quasi_order][:,i])


# # Wannier States

# In[990]:


er_scale = 573.320792 # Hz*um^2, needs spacing^2 divided through
def get_er_and_spacing(depth, gamma, spacing):
    v0 = depth*(1+0.5*gamma*gamma)
    a_eff = spacing/(1+0.5*gamma*gamma)
    return a_eff*a_eff*v0/er_scale, a_eff
    
def wannier_state(depth, gamma, spacing):
    v0 = depth*(1+0.5*gamma*gamma)
    a_eff = spacing/(1+0.5*gamma*gamma)
    k_l_eff = (1+0.5*gamma*gamma)*np.pi/spacing
    print(v0, k_l_eff, (spacing, a_eff), spacing*spacing*v0/er_scale)


# In[991]:


with h5py.File(sourceMore, 'r', libver='latest') as f:
    ver, a = get_er_and_spacing(f.attrs['lattice_depth'],
                  f['scan_values'][which_set],
                  f.attrs['site_spacing_um'])


# In[992]:


n_qs = 1024 + 1
k_rad =  50
n_ks = 2*k_rad + 1
all_qs = np.linspace(-1, 1, n_qs, endpoint=True)
v0_vals = np.empty((n_qs, n_ks))
v0_vecs = np.empty((n_qs, n_ks, n_ks))
v0_qs = np.empty((n_qs*n_ks))
v0_wp = np.empty_like(v0_qs, dtype=np.complex)
for idx, q in enumerate(all_qs):
    these_kparts = (2*np.arange(-k_rad, k_rad+1)) + q
    v0_qs[idx::n_qs] = these_kparts
    vlattice = np.diag(np.full(2*k_rad, -ver/4), k=1) + np.diag(np.full(2*k_rad, -ver/4), k=-1)
    qlattice = np.diag(np.square((2*np.arange(-k_rad, k_rad+1)) + q) )
    vals, vecs = np.linalg.eigh(vlattice + qlattice)
    order = np.argsort(vals)
    v0_vals[idx] = vals[order]
    v0_vecs[idx] = vecs[:, order]
    v0_wp[idx::n_qs] = v0_vecs[idx][:,0]
#n_xs = 2048+1
#x_range = 20.5
#x_s = np.linspace(-x_range-1, x_range+1, n_xs, endpoint=True).reshape(-1,1)
#kxs = np.exp(np.complex(0, np.pi)*(v0_qs.ravel().reshape(1,-1))*x_s)
#print(x_s.shape, v0_vecs[:,:,0].shape)
#wannier_funct = np.sum((((np.sign(v0_vecs[:,50,0][:,np.newaxis])*v0_vecs[:,:,0]).ravel().reshape(1,-1))*kxs), axis=1)/np.sqrt(n_qs)
#plt.plot(x_s, wannier_funct, 'k')
#plt.plot(x_s-1, wannier_funct, ':k')
#plt.plot(x_s+1, wannier_funct, ':k')
#plt.xlim(-x_range, x_range)
direct_wannier_p = v0_wp/np.sqrt((v0_qs[1]-v0_qs[0])*np.sum(np.square(np.abs(v0_wp))))
subrange = np.where(np.abs(v0_qs)< 10)
print(subrange[0].size)
sub_ks = np.pi/a*v0_qs[subrange]
sub_wp = direct_wannier_p[subrange]
plt.plot(v0_qs[subrange], np.square(np.abs(direct_wannier_p))[subrange], ',', label='direct')
xs = np.linspace(-4.0, 4.0, 513, endpoint=True)
direct_wannier_x = np.sum(direct_wannier_p[np.newaxis,:] * np.exp(complex(0, np.pi)*xs[:,np.newaxis]*v0_qs[np.newaxis,:]), axis=1)


# In[993]:


plt.plot(xs, direct_wannier_x)


# In[994]:


shift_one = np.exp(complex(0, np.pi)*-1.0*v0_qs[np.newaxis,:])
shift_two = np.exp(complex(0, np.pi)*2.0*v0_qs[np.newaxis,:])
shift_mone = np.exp(complex(0, np.pi)*1.0*v0_qs[np.newaxis,:])
direct_wannier_s1_x = np.sum(direct_wannier_p[np.newaxis,:] * shift_one * np.exp(complex(0, np.pi)*xs[:,np.newaxis]*v0_qs[np.newaxis,:]), axis=1)
direct_wannier_sm1_x = np.sum(direct_wannier_p[np.newaxis,:] * shift_mone * np.exp(complex(0, np.pi)*xs[:,np.newaxis]*v0_qs[np.newaxis,:]), axis=1)
plt.plot(xs, direct_wannier_s1_x)
plt.plot(xs, direct_wannier_sm1_x)
plt.plot(xs, direct_wannier_x)
plt.axvline(-1)
plt.axvline(0)
plt.axvline(1)


# In[995]:


wp_all = sub_wp[np.newaxis,:] * np.exp(complex(0, 1)*sub_ks[np.newaxis,:]*np.arange(-(n_sites//2),n_sites//2+1)[:,np.newaxis])


# In[996]:


plt.imshow(wp_all.real, aspect=100, cmap='RdBu')


# In[997]:


plt.imshow(wp_all.imag, aspect=100, cmap='RdBu')


# In[998]:


plt.imshow(np.square(np.abs(wp_all)), aspect=100)


# In[999]:


plt.plot(mags[:,quasi_order].T[0])


# In[1000]:


for i in np.arange(n_sites):
    plt.plot(sub_ks, ordered_quasi[i] + np.square(np.abs(np.sum(mags[:,quasi_order][:,i]*wp_all.T, axis=1))))
    plt.annotate(i, (4 + 0.5*(i%4), ordered_quasi[i]))
plt.xlim(-6, 6)
#plt.ylim(-20,20)


# In[1001]:


s_list = np.arange(-(n_sites//2),n_sites//2+1)
smooth_set = set([0,1,2,5,12,32,39,42,43,44])
wiggle_set = set([3,4,8,9,13,14,30,31,35,36,40,41])
other_set = set([6,7,10,11,15,16,28,29,33,34,37,38])
resplit_set = set([17,20,24,27])
again_set = set([18,19,22,25,26])
rem_set = set(np.arange(n_sites)) - (smooth_set | wiggle_set | other_set | resplit_set | again_set)


# In[1002]:


fig, (ax_p, ax_s, ax_sq) = plt.subplots(1,3, sharey=True, figsize=(8,3))
for i in smooth_set:
    ax_p.plot(sub_ks, ordered_quasi[i] + np.square(np.abs(np.sum(mags[:,quasi_order][:,i]*wp_all.T, axis=1))))
    ax_p.annotate(i, (4 + 0.4*(i%4), ordered_quasi[i]))
    ax_s.plot(s_list, ordered_quasi[i] + 4*mags[:,quasi_order][:,i])
    ax_s.annotate(i, (n_sites//2 - 4 + (i%2), ordered_quasi[i]))
    ax_sq.plot(s_list, ordered_quasi[i] + 10*np.square(np.abs(mags[:,quasi_order][:,i])))
    ax_sq.annotate(i, (n_sites//2 - 4 + (i%2), ordered_quasi[i]))
ax_p.set_xlim(-6,6)
ax_s.set_xlim(s_list[0],s_list[-1])
ax_p.set_xlabel('p-ish')
ax_s.set_xlabel('site index')
ax_sq.set_xlim(s_list[0],s_list[-1])
ax_sq.set_xlabel('site index')
fig.tight_layout()


# In[1003]:


fig, (ax_p, ax_s, ax_sq) = plt.subplots(1,3, sharey=True, figsize=(8,3))
for i in wiggle_set:
    ax_p.plot(sub_ks, ordered_quasi[i] + np.square(np.abs(np.sum(mags[:,quasi_order][:,i]*wp_all.T, axis=1))))
    ax_p.annotate(i, (4 + 0.4*(i%4), ordered_quasi[i]))
    ax_s.plot(s_list, ordered_quasi[i] + 4*mags[:,quasi_order][:,i])
    ax_s.annotate(i, (n_sites//2 - 4 + (i%2), ordered_quasi[i]))
    ax_sq.plot(s_list, ordered_quasi[i] + 10*np.square(np.abs(mags[:,quasi_order][:,i])))
    ax_sq.annotate(i, (n_sites//2 - 4 + (i%2), ordered_quasi[i]))
ax_p.set_xlim(-6,6)
ax_s.set_xlim(s_list[0],s_list[-1])
ax_p.set_xlabel('p-ish')
ax_s.set_xlabel('site index')
ax_sq.set_xlim(s_list[0],s_list[-1])
ax_sq.set_xlabel('site index')
fig.tight_layout()


# In[1004]:


fig, (ax_p, ax_s, ax_sq) = plt.subplots(1,3, sharey=True, figsize=(8,3))
for i in other_set:
    ax_p.plot(sub_ks, ordered_quasi[i] + np.square(np.abs(np.sum(mags[:,quasi_order][:,i]*wp_all.T, axis=1))))
    ax_p.annotate(i, (4 + 0.4*(i%4), ordered_quasi[i]))
    ax_s.plot(s_list, ordered_quasi[i] + 4*mags[:,quasi_order][:,i])
    ax_s.annotate(i, (n_sites//2 - 4 + (i%2), ordered_quasi[i]))
    ax_sq.plot(s_list, ordered_quasi[i] + 10*np.square(np.abs(mags[:,quasi_order][:,i])))
    ax_sq.annotate(i, (n_sites//2 - 4 + (i%2), ordered_quasi[i]))
ax_p.set_xlim(-6,6)
ax_s.set_xlim(s_list[0],s_list[-1])
ax_p.set_xlabel('p-ish')
ax_s.set_xlabel('site index')
ax_sq.set_xlim(s_list[0],s_list[-1])
ax_sq.set_xlabel('site index')
fig.tight_layout()


# In[ ]:





# In[ ]:





# In[1005]:


fig, (ax_p, ax_s, ax_sq) = plt.subplots(1,3, sharey=True, figsize=(8,3))
for i in resplit_set:
    ax_p.plot(sub_ks, ordered_quasi[i] + np.square(np.abs(np.sum(mags[:,quasi_order][:,i]*wp_all.T, axis=1))))
    ax_p.annotate(i, (4 + 0.4*(i%4), ordered_quasi[i]))
    ax_s.plot(s_list, ordered_quasi[i] + 4*mags[:,quasi_order][:,i])
    ax_s.annotate(i, (n_sites//2 - 4 + (i%2), ordered_quasi[i]))
    ax_sq.plot(s_list, ordered_quasi[i] + 10*np.square(np.abs(mags[:,quasi_order][:,i])))
    ax_sq.annotate(i, (n_sites//2 - 4 + (i%2), ordered_quasi[i]))
ax_p.set_xlim(-6,6)
ax_s.set_xlim(s_list[0],s_list[-1])
ax_p.set_xlabel('p-ish')
ax_s.set_xlabel('site index')
ax_sq.set_xlim(s_list[0],s_list[-1])
ax_sq.set_xlabel('site index')
fig.tight_layout()


# In[1006]:


fig, (ax_p, ax_s, ax_sq) = plt.subplots(1,3, sharey=True, figsize=(8,3))
for i in again_set:
    ax_p.plot(sub_ks, ordered_quasi[i] + np.square(np.abs(np.sum(mags[:,quasi_order][:,i]*wp_all.T, axis=1))))
    ax_p.annotate(i, (4 + 0.4*(i%4), ordered_quasi[i]))
    ax_s.plot(s_list, ordered_quasi[i] + 4*mags[:,quasi_order][:,i])
    ax_s.annotate(i, (n_sites//2 - 4 + (i%2), ordered_quasi[i]))
    ax_sq.plot(s_list, ordered_quasi[i] + 10*np.square(np.abs(mags[:,quasi_order][:,i])))
    ax_sq.annotate(i, (n_sites//2 - 4 + (i%2), ordered_quasi[i]))
ax_p.set_xlim(-6,6)
ax_s.set_xlim(s_list[0],s_list[-1])
ax_p.set_xlabel('p-ish')
ax_s.set_xlabel('site index')
ax_sq.set_xlim(s_list[0],s_list[-1])
ax_sq.set_xlabel('site index')
fig.tight_layout()


# In[1007]:


fig, (ax_p, ax_s, ax_sq) = plt.subplots(1,3, sharey=True, figsize=(8,3))
for i in rem_set:
    ax_p.plot(sub_ks, ordered_quasi[i] + np.square(np.abs(np.sum(mags[:,quasi_order][:,i]*wp_all.T, axis=1))))
    ax_p.annotate(i, (4 + 0.4*(i%4), ordered_quasi[i]))
    ax_s.plot(s_list, ordered_quasi[i] + 4*mags[:,quasi_order][:,i])
    ax_s.annotate(i, (n_sites//2 - 4 + (i%2), ordered_quasi[i]))
    ax_sq.plot(s_list, ordered_quasi[i] + 10*np.square(np.abs(mags[:,quasi_order][:,i])))
    ax_sq.annotate(i, (n_sites//2 - 4 + (i%2), ordered_quasi[i]))
ax_p.set_xlim(-6,6)
ax_s.set_xlim(s_list[0],s_list[-1])
ax_p.set_xlabel('p-ish')
ax_s.set_xlabel('site index')
ax_sq.set_xlim(s_list[0],s_list[-1])
ax_sq.set_xlabel('site index')
fig.tight_layout()


# In[1008]:


heff = np.matmul(mags[:,quasi_order], np.matmul(np.diag(ordered_quasi), mags[:,quasi_order].T))
maxmag = np.amax(np.abs(heff))
plt.imshow(heff, cmap="PuOr", vmin=-maxmag, vmax=maxmag)
plt.colorbar()


# In[1011]:


from scipy.special import jv, jn_zeros
scale = jn_zeros(0,3)[-1]/21
plt.plot((s_list[1:]-0.5),-np.diag(heff, k=1), label="Tunneling, +1")
plt.plot((s_list[1:]-0.5),-np.diag(heff, k=-1), label="Tunneling, -1")
#plt.plot(25*jv(0, (s_list[1:]-0.5)*scale), label="Bessel J_0")
plt.plot((s_list[1:]-0.5),fit_amp*jv(0, (s_list[1:]-0.5)*fit_scale), label="Bessel J_0")

plt.legend()


# In[1010]:


plt.plot(-np.diag(heff))


# In[ ]:




