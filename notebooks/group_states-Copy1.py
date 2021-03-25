#!/usr/bin/env python
# coding: utf-8

# In[1344]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import h5py
from matplotlib import pyplot as plt
import drive_coefficients as dc


# In[2]:


from zss_progbar import log_progress as progbar


# In[3]:


import os
import glob
import re


# In[1052]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
mpl.rcParams['axes.unicode_minus'] = False
basic_style_rc = {
    'lines.linewidth': 1,
    'savefig.format': 'pdf',
    'pdf.compression': 9,
    'axes.linewidth': 0.5,
    'grid.linewidth': 0.5,
}
umd_small = {
    'lines.linewidth': 1,
    'savefig.format': 'pdf',
    'pdf.compression': 9,
    'axes.linewidth': 0.5,
    'grid.linewidth': 0.5,
    'font.family': 'serif',
    'font.serif': 'Computer Modern Roman',
    'text.usetex': True,
    'axes.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.unicode_minus':False,
    'text.latex.preamble':[
        r"\usepackage[utf8]{inputenc}",
        r"\DeclareUnicodeCharacter{2212}{\ensuremath{-}}"
    ]
}
umd_small_cxt = mpl.rc_context(rc=umd_small)


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


# In[1050]:


from cycler import cycler
kelly_color_cycler = (cycler(color=['#FFB300', '#803E75', '#FF6800',
              '#A6BDD7', '#C10020', '#CEA262',
              '#817066', '#007D34']))
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
roughMore, refinedMore, sourceMore = load_resid_file(data_paths[3])
print(sourceMore)


# In[1202]:


with umd_small_cxt:
    fig, ax_arr = plt.subplots(1,2, sharex=True,sharey=True,figsize=(4,2))
    for i in np.arange(roughMore.shape[0]):
        ax_arr[1].loglog(refinedMore[i], roughMore[i], ',r')
    for i in np.arange(roughLess.shape[0]):
        ax_arr[0].loglog(refinedLess[i], roughLess[i], ',r')
    for ax in ax_arr:
        ax.plot([1e-15,1.5e-8], [1e-15,1.5e-8], ':k')
        ax.axhline(1.5e-8, ls=':', c='k')
        ax.axvline(1.5e-8, ls=':', c='k')
        ax.set_xlabel(r'$\widetilde{\Delta}_\mathrm{rms}$')
    ax_arr[0].annotate(r'$N_\mathrm{sites}=33$', (1e-15,1e-2))
    ax_arr[1].annotate(r'$N_\mathrm{sites}=45$', (1e-15,1e-2))
    ax_arr[0].set_ylabel(r'$\overline{\Delta}_\mathrm{rms}$')
    fig.tight_layout()
    #fig.savefig('deltadelta.pdf', dpi=600, bbox_inches='tight', pad_inches=1/72)
    fig.savefig('deltadelta.png', dpi=600, bbox_inches='tight', pad_inches=1/72)


# In[10]:


print(roughMore.shape, roughLess.shape)


# In[11]:


plt.plot(np.sum(refinedMore < 1e-13, axis=1)/45)
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


# In[1047]:


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
        try:
            all_drive_freqs = f['critical_drive'][:]
        except KeyError:
            all_drive_freqs = f['critcal_drive'][:]
        n_sets = all_drive_freqs[offset::stride].size
        group_idx = np.full((n_sets, n_states), -1, dtype=np.int8)
        all_fqe_means = np.empty((n_sets, n_sites))
        all_fqe_stdev = np.empty_like(all_fqe_means)
        all_fqe_count = np.empty_like(all_fqe_means)
        all_fqe_minmax = np.empty((n_sets, n_sites, 2))
        for set_idx in progbar(np.arange(0,n_sets)):
            below_thresh = np.where(resids[set_idx] < resid_thresh)[0]
            #stable_states = state_matrix[stride*set_idx+offset, :, below_thresh]
            stable_quasi_en = energy_matrix[stride*set_idx+offset, below_thresh]
            this_drive = all_drive_freqs[stride*set_idx+offset]
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
                    raise
            print("Found {} states with {} pairs of degenerate states".format(out_idx+1, n_deg))
        plt.semilogy(np.diff(sorted_en), '.k')
    return all_fqe_means, all_fqe_stdev, all_fqe_minmax, all_fqe_count, group_idx


# In[1391]:


fbz_mean, fbz_dev, fbz_minmax, fbz_count, groups = find_fbz_qes(refinedMore, sourceMore, resid_thresh=1e-13, stride=2)
plt.ylim(None,1e-4)


# In[1392]:


fbz_n33_mean, fbz_n33_dev, fbz_n33_minmax, fbz_n33_count, n33_groups = find_fbz_qes(refinedLess, sourceLess, resid_thresh=5e-14, diff_split=1e-6, stride=1)
plt.ylim(None,1e-4)


# In[1393]:


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


# In[294]:


plt.semilogy(fbz_dev[:,:])
#plt.semilogy(np.abs(fbz_minmax[:,:,0]-fbz_mean[:,:]), ls="--")
#plt.semilogy(np.abs(fbz_minmax[:,:,1]-fbz_mean[:,:]), ls="--")
plt.plot(np.diff(fbz_mean[:,:],axis=1), ls='--')
plt.axhline(2e-6, c='k')
plt.ylim(None,None)


# In[295]:


plt.semilogy(fbz_n33_dev[:,:])
#plt.semilogy(np.abs(fbz_minmax[:,:,0]-fbz_mean[:,:]), ls="--")
#plt.semilogy(np.abs(fbz_minmax[:,:,1]-fbz_mean[:,:]), ls="--")
plt.plot(np.diff(fbz_n33_mean[:,:],axis=1), ls='--')
plt.axhline(2e-6, c='k')
plt.ylim(None,None)


# In[296]:


plt.imshow(groups.T[5500:8500].T, aspect=20, vmin=0, vmax=45)
#plt.colorbar()


# In[297]:


plt.imshow(n33_groups.T, aspect=1/25, vmin=0, vmax=33)
plt.colorbar()


# In[298]:


np.all([np.equal(np.unique(groups[i]), np.arange(-1, 45)) for i in np.arange(64)]), np.all([np.equal(np.unique(n33_groups[i]), np.arange(-1, 33)) for i in np.arange(256)])


# In[299]:


plt.plot(fbz_count[:,:], 'k', lw=1);
plt.plot(fbz_n33_count[:,:], 'r', lw=1);


# In[1401]:


with umd_small_cxt:
    fig, ax_arr = plt.subplots(2,2, sharey=True, figsize=(5.5,4))
    ax_arr[0,0].set_ylabel(r"Energy $E/h$ (Hz)")
    #ax_arr[0,0].set_xlabel(r'$N_\mathrm{sites}=33$')
    #ax_arr[0,1].set_xlabel(r'$N_\mathrm{sites}=45$')
    this_pathh0 = r'/media/simulationData/BreathingLattice/floquet/small_lattice/h0_only_n33/h0_only_n33_out.h5'
    with h5py.File(this_pathh0, 'r', libver="latest") as f:
        print(os.path.basename(this_pathh0))
        loaded_energiesh0 = f['floquet_energy'][:]
        gammash0 = f['scan_values'][:]
    ax_arr[0,0].semilogx(gammash0, loaded_energiesh0, 'k', lw=1);
    this_pathh0 = '/media/simulationData/BreathingLattice/floquet/small_lattice/h0_only_n45/h0_only_n45_out.h5'
    ax_arr[0,0].set_xticklabels([])
    with h5py.File(this_pathh0, 'r', libver="latest") as f:
        print(os.path.basename(this_pathh0))
        loaded_energiesh0 = f['floquet_energy'][:]
        gammash0 = f['scan_values'][:]
    ax_arr[0,1].semilogx(gammash0, loaded_energiesh0, 'k', lw=1);
    ax_arr[0,1].set_xticklabels([])
    ax_arr[0,0].set_xticklabels([])
    ax_arr[0,1].set_yticklabels([])
    ax_arr[0,0].set_xlim(1e-5,1e-1)
    ax_arr[0,1].set_xlim(1e-5,1e-1)
    # Now the interesting ones
    ax_arr[1,1].semilogx(gammas[::], fbz_mean[:,:], 'k', lw=1);
    ax_arr[1,1].set_xlabel(r"$\gamma$")
    ax_arr[1,0].semilogx(n33_gammas[::], fbz_n33_mean[:,:], 'k', lw=1);
    ax_arr[1,1].set_xlim(1e-5,1e-1)
    ax_arr[1,0].set_xlim(1e-5,1e-1)
    ax_arr[1,0].set_xlabel(r"$\gamma$")
    ax_arr[1,0].set_ylabel(r"quasienergy $\epsilon/h$ (Hz)")
    ax_arr[0,0].set_title(r'$N_\mathrm{sites}=33$')
    ax_arr[0,1].set_title(r'$N_\mathrm{sites}=45$')
    plt.tight_layout()
    fig.savefig('quasien_lattice.pdf', dpi=600, bbox_inches='tight', pad_inches=1/72)


# In[301]:


plt.semilogx(gammas[::], fbz_mean[:,:], 'b', lw=1);
plt.semilogx(n33_gammas[::], fbz_n33_mean[:,:], 'k', lw=1);


# In[302]:


fig, ax_arr = plt.subplots(1,2, sharey=True)
ax_arr[1].semilogx(omegas[::], fbz_mean[:,:], 'k', lw=1);
ax_arr[0].semilogx(n33_omegas[::], fbz_n33_mean[:,:], 'k', lw=1);
plt.ylim(-20,20)


# In[303]:


plt.semilogx(omegas[::], fbz_mean[:,:], 'b', lw=1);
plt.semilogx(n33_omegas[::], fbz_n33_mean[:,:], 'k', lw=1);


# In[304]:


fig, ax_arr = plt.subplots(1,2, sharey=True)
ax_arr[1].semilogx(gammas[::], fbz_mean[:,:], 'k', lw=1);
ax_arr[0].semilogx(n33_gammas[::], fbz_n33_mean[:,:], 'k', lw=1);
ax_arr[0].set_ylim(-20, 20)


# In[305]:


plt.semilogx(gammas[::], fbz_mean[:,:], 'b', lw=1);
plt.semilogx(n33_gammas[::], fbz_n33_mean[:,:], 'k', lw=1);
plt.ylim(-20, 20)


# In[306]:


plt.semilogx(gammas[::], fbz_mean[:,:], 'b', lw=1);
plt.semilogx(n33_gammas[::], fbz_n33_mean[:,:], 'k', lw=1);
plt.ylim(-8, 8)


# ## End of energy-level diagrams
# 
# Now we'll try to do some states stuff.
# Let's do stroboscopic first, 'cause it is easier

# # n=33

# In[1210]:


idx_choice = np.empty(33, dtype=np.int)
for state_idx in np.arange(33):
    zz = np.nonzero(n33_groups[0] == state_idx)[0]
    minchoice = np.argmin(refinedLess[0, zz])
    idx_choice[state_idx] = zz[minchoice]
refinedLess[0, idx_choice]


# In[1211]:


#%%time
which_set = 220
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


# In[1212]:


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


# In[1213]:


plt.imshow(np.log(np.matmul(mags.T, mags)), vmin=-16)
plt.colorbar()


# In[1214]:


ordered_quasi = (np.mod(chosen_quasi + 0.5*chosen_drive, chosen_drive)-0.5*chosen_drive)[quasi_order]
for i in np.arange(33):
    plt.plot(ordered_quasi[i] + 4*mags[:,quasi_order][:,i])


# # Wannier States

# In[1215]:


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


# In[1216]:


with h5py.File(sourceLess, 'r', libver='latest') as f:
    ver, a = get_er_and_spacing(f.attrs['lattice_depth'],
                  f['scan_values'][which_set],
                  f.attrs['site_spacing_um'])


# In[1217]:


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


# In[1218]:


plt.plot(xs, direct_wannier_x)


# In[1219]:


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


# In[1220]:


wp_all = sub_wp[np.newaxis,:] * np.exp(complex(0, 1)*sub_ks[np.newaxis,:]*np.arange(-(n_sites//2),n_sites//2+1)[:,np.newaxis])


# In[1221]:


plt.imshow(wp_all.real, aspect=100, cmap='RdBu')


# In[1222]:


plt.imshow(wp_all.imag, aspect=100, cmap='RdBu')


# In[1223]:


plt.imshow(np.square(np.abs(wp_all)), aspect=100)


# In[1224]:


for i in np.arange(n_sites):
    plt.plot(sub_ks, ordered_quasi[i] + np.square(np.abs(np.sum(mags[:,quasi_order][:,i]*wp_all.T, axis=1))))
    plt.annotate(i, (4 + 0.5*(i%4), ordered_quasi[i]))
plt.xlim(-6, 6)
#plt.ylim(-20,20)


# In[1225]:


s_list = np.arange(-(n_sites//2),n_sites//2+1)
smooth_set = set([0,1,2,5,10,22,27,30,31,32])
wiggle_set = set([3,4,6,7,11,12,20,21,25,26,28,29])
other_set = set([8,9,16,23,24])
rem_set = set(np.arange(n_sites)) - (smooth_set | wiggle_set | other_set)


# In[1226]:


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


# In[1227]:


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


# In[1228]:


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


# In[1229]:


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


# In[1230]:


vert_stretch = 7
with umd_small_cxt:
    fig, ax_arr = plt.subplots(1,4,figsize=(5.8,3.0), sharey=True)
    for ax in ax_arr:
        ax.set_xlabel("site index $j$")
        ax.set_ylim(-50,50)
        ax.set_prop_cycle(kelly_color_cycler)
    for i in np.sort(list(smooth_set)):
        ax_arr[0].plot(s_list, ordered_quasi[i] + vert_stretch*mags[:,quasi_order][:,i])
    for i in np.sort(list(wiggle_set)):
        ax_arr[1].plot(s_list, ordered_quasi[i] + vert_stretch*mags[:,quasi_order][:,i])
    for i in np.sort(list(other_set)):
        ax_arr[2].plot(s_list, ordered_quasi[i] + vert_stretch*mags[:,quasi_order][:,i])
    for i in np.sort(list(rem_set)):
        ax_arr[3].plot(s_list, ordered_quasi[i] + vert_stretch*mags[:,quasi_order][:,i])
    ax_arr[0].set_ylabel(r'quasenergy $\epsilon$')
    fig.tight_layout()
    fig.savefig('x_states.pdf', dpi=600, bbox_inches='tight', pad_inches=1/72)


# In[1231]:


heff33 = np.matmul(mags[:,quasi_order], np.matmul(np.diag(ordered_quasi), mags[:,quasi_order].T))
maxmag33 = np.amax(np.abs(heff))
plt.imshow(heff33, cmap="PuOr", vmin=-maxmag, vmax=maxmag)
plt.colorbar()


# In[1232]:


from scipy.special import jv, jn_zeros
scale = jn_zeros(0,3)[-1]/21
plt.plot((s_list[1:]-0.5),-np.diag(heff33, k=1), label="Tunneling, +1")
plt.plot((s_list[1:]-0.5),-np.diag(heff33, k=-1), label="Tunneling, -1")
plt.plot((s_list[1:]-0.5),25*jv(0, (s_list[1:]-0.5)*scale), label="Bessel J0")
plt.axvline(0)
plt.legend()


# In[1233]:


def j0(x, x_scale, y_scale):
    return jv(0, scale*x)
from scipy import optimize
plt.plot((s_list[1:]-0.5),-np.diag(heff33, k=1), label="Tunneling, +1")
plt.plot((s_list[1:]-0.5),-np.diag(heff33, k=-1), label="Tunneling, -1")
plt.axvline(0)
plt.legend()
j33_list = s_list[1:]-0.5
print(fit_scale, fit_amp)


# ### n=45

# In[1234]:


idx_choice = np.empty(45, dtype=np.int)
for state_idx in np.arange(45):
    zz = np.nonzero(groups[0] == state_idx)[0]
    minchoice = np.argmin(refinedMore[0, zz])
    idx_choice[state_idx] = zz[minchoice]
refinedMore[0, idx_choice]


# In[1235]:


get_ipython().run_cell_magic('time', '', "which_set = 32\nsorted_choices = np.sort(idx_choice)\nprint(sorted_choices)\nwith h5py.File(sourceMore, 'r', libver='latest') as f:\n    print(os.path.basename(sourceMore))\n    n_sites = f.attrs['n_sites']\n    m_rad = f.attrs['floquet_m_radius']\n    n_ms = 2*m_rad + 1\n    n_states = n_sites * n_ms\n    chosen_states = f['floquet_state'][which_set, :, sorted_choices]\n    chosen_quasi = f['floquet_energy'][which_set, sorted_choices]\n    chosen_gamma = f['scan_values'][which_set]\n    try:\n        chosen_drive = f['critical_drive'][which_set]\n    except KeyError:\n        chosen_drive = f['critcal_drive'][which_set]\n#plt.semilogx(gammas[::], fbz_mean[:,:], 'r', lw=1);\nsub_mean_quasi = np.sort(fbz_mean[which_set])\nsorted_wrapped_quasi = np.sort(np.mod(chosen_quasi + 0.5*chosen_drive, chosen_drive)-0.5*chosen_drive)\nplt.plot(sub_mean_quasi-sorted_wrapped_quasi, '.')")


# In[1236]:


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


# In[1237]:


plt.imshow(np.log(np.matmul(mags.T, mags)), vmin=-16)
plt.colorbar()


# In[1238]:


ordered_quasi = (np.mod(chosen_quasi + 0.5*chosen_drive, chosen_drive)-0.5*chosen_drive)[quasi_order]
for i in np.arange(n_sites):
    plt.plot(ordered_quasi[i] + 4*mags[:,quasi_order][:,i])


# # Wannier States

# In[1239]:


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


# In[1240]:


with h5py.File(sourceMore, 'r', libver='latest') as f:
    ver, a = get_er_and_spacing(f.attrs['lattice_depth'],
                  f['scan_values'][which_set],
                  f.attrs['site_spacing_um'])


# In[1241]:


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


# In[1242]:


plt.plot(xs, direct_wannier_x)


# In[1243]:


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


# In[1244]:


wp_all = sub_wp[np.newaxis,:] * np.exp(complex(0, 1)*sub_ks[np.newaxis,:]*np.arange(-(n_sites//2),n_sites//2+1)[:,np.newaxis])


# In[1245]:


plt.imshow(wp_all.real, aspect=100, cmap='RdBu')


# In[1246]:


plt.imshow(wp_all.imag, aspect=100, cmap='RdBu')


# In[1247]:


plt.imshow(np.square(np.abs(wp_all)), aspect=100)


# In[1248]:


plt.plot(mags[:,quasi_order].T[0])


# In[1249]:


for i in np.arange(n_sites):
    plt.plot(sub_ks, ordered_quasi[i] + np.square(np.abs(np.sum(mags[:,quasi_order][:,i]*wp_all.T, axis=1))))
    plt.annotate(i, (4 + 0.5*(i%4), ordered_quasi[i]))
plt.xlim(-6, 6)
#plt.ylim(-20,20)


# In[1250]:


s_list = np.arange(-(n_sites//2),n_sites//2+1)
smooth_set = set([0,1,2,5,12,32,39,42,43,44])
wiggle_set = set([3,4,8,9,13,14,30,31,35,36,40,41])
other_set = set([6,7,10,11,15,16,28,29,33,34,37,38])
resplit_set = set([17,20,24,27])
again_set = set([18,19,22,25,26])
rem_set = set(np.arange(n_sites)) - (smooth_set | wiggle_set | other_set | resplit_set | again_set)


# In[1251]:


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


# In[1252]:


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


# In[1298]:


fig, (ax_p, ax_s, ax_sq) = plt.subplots(1,3, sharey=True, figsize=(8,3))
for i in wiggle_set:
    ax_p.plot(sub_ks, ordered_quasi[i] + np.square(np.abs(np.sum(mags[:,quasi_order][:,i]*wp_all.T, axis=1))))
    ax_p.plot(sub_ks, ordered_quasi[i] + np.square(np.abs(np.sum(mags[:,quasi_order][:,i]*wp_all.T, axis=1))))
    ax_p.annotate(i, (4 + 0.4*(i%4), ordered_quasi[i]))
    ax_s.plot(s_list, ordered_quasi[i] + 4*mags[:,quasi_order][:,i])
    ax_s.annotate(i, (n_sites//2 - 4 + (i%2), ordered_quasi[i]))
    ax_sq.plot(s_list, ordered_quasi[i] + 10*np.square(np.abs(mags[:,quasi_order][:,i])))
    ax_sq.annotate(i, (n_sites//2 - 4 + (i%2), ordered_quasi[i]))
ax_p.set_xlim(-6,6)
ax_s.set_xlim(s_list[0],s_list[-1])
ax_p.set_xlabel(r'$ka/\pi$')
ax_s.set_xlabel('site index')
ax_sq.set_xlim(s_list[0],s_list[-1])
ax_sq.set_xlabel('site index $j$')
fig.tight_layout()


# In[1253]:


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





# In[1254]:


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


# In[1255]:


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


# In[1256]:


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


# In[1257]:


heff45 = np.matmul(mags[:,quasi_order], np.matmul(np.diag(ordered_quasi), mags[:,quasi_order].T))
maxmag = np.amax(np.abs(heff))
plt.imshow(heff, cmap="PuOr", vmin=-maxmag, vmax=maxmag)
plt.colorbar()


# In[1258]:


from scipy.special import jv, jn_zeros
scale = jn_zeros(0,3)[-1]/21
plt.plot((s_list[1:]-0.5),-np.diag(heff, k=1), label="Tunneling, +1")
plt.plot((s_list[1:]-0.5),-np.diag(heff, k=-1), label="Tunneling, -1")
#plt.plot(25*jv(0, (s_list[1:]-0.5)*scale), label="Bessel J_0")
plt.plot((s_list[1:]-0.5),fit_amp*jv(0, (s_list[1:]-0.5)*fit_scale), label="Bessel J_0")

plt.legend()


# In[1259]:


plt.plot(-np.diag(heff))


# In[1275]:


with umd_small_cxt:
    fig, ax = plt.subplots(1,1, figsize=(4,2))
    (fit_scale, fit_amp), _ =optimize.curve_fit(j0, j33_list, -np.diag(heff33, k=1), (jn_zeros(0,3)[-1]/21, np.abs(np.diag(heff33, k=1)).max()))
    ax.set_prop_cycle(kelly_color_cycler)
    
    ax.plot(np.linspace(-22,22,256,endpoint=True), fit_amp*jv(0, np.linspace(-22,22,256,endpoint=True)*fit_scale), label="Bessel J0")
    ax.plot((s_list[1:]-0.5), -np.diag(heff45, k=1), '.')
    ax.plot(j33_list, -np.diag(heff33, k=1), '.')
    ax.set_xlabel('site index $j$')
    ax.set_ylabel('$-J_{j,j-1}/h$ (Hz)')
    fig.tight_layout()
    fig.savefig('bessel_tun.pdf', dpi=600, bbox_inches='tight', pad_inches=1/72)
print(fit_scale, fit_amp)


# In[1201]:


with umd_small_cxt:
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(4,2))
    maxmag = max(np.abs(heff33).max(), np.abs(heff45).max())
    axl.imshow(heff33, vmin=-maxmag, vmax=maxmag, cmap='PuOr')
    axl.tick_params(which='both', bottom=False, left=False, labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    mesh = axr.imshow(heff45, vmin=-maxmag, vmax=maxmag, cmap='PuOr')
    axr.tick_params(which='both', bottom=False, left=False, labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    fig.tight_layout()
    fig.savefig('heff.pdf', dpi=600, bbox_inches='tight', pad_inches=1/72)


# In[1279]:


with umd_small_cxt:
    fig, ax = plt.subplots(1,1, figsize=(1,2))
    ax.axis('off')
    plt.colorbar(mesh)
    fig.savefig('heff_cb.pdf', dpi=600, bbox_inches='tight', pad_inches=1/72)


# In[1342]:


import matplotlib.gridspec as gridspec
s_list = np.arange(-(n_sites//2),n_sites//2+1)
smooth_set = set([0,1,2,5,12,32,39,42,43,44])
wiggle_set = set([3,4,8,9,13,14,30,31,35,36,40,41])
other_set = set([6,7,10,11,15,16,28,29,33,34,37,38])
resplit_set = set([17,20,24,27])
again_set = set([18,19,22,25,26])
rem_set = set(np.arange(n_sites)) - (smooth_set | wiggle_set | other_set | resplit_set | again_set)
with umd_small_cxt:
    fig = plt.figure(figsize=(5.8,6))
    group_panes = gridspec.GridSpec(6,1)
    for idx, this_set in enumerate([smooth_set, wiggle_set, other_set,
                                    resplit_set, again_set, rem_set]):
        top, bottom = gridspec.GridSpecFromSubplotSpec(1,2,
                                                      subplot_spec=group_panes[idx],
                                                      wspace=0.1, hspace=0.0, width_ratios=[3,1])
        axt = fig.add_subplot(top)
        axt.set_prop_cycle(kelly_color_cycler)
        axt.set_xlim(-6,6)
        axt.set_ylabel('$\\epsilon/h$ [Hz]\n$|\psi|^2$')
        #axt.set_ylim(-55,55)
        
        axb = fig.add_subplot(bottom)
        axb.set_xlim(s_list[0],s_list[-1])
        
        for state_idx in this_set:
            line, = axt.plot(sub_ks, ordered_quasi[state_idx] + np.square(np.abs(np.sum(mags[:,quasi_order][:,state_idx]*wp_all.T, axis=1))), lw=1)
            #ax_p.annotate(i, (4 + 0.4*(i%4), ordered_quasi[i]))
            #ax_s.plot(s_list, ordered_quasi[i] + 4*mags[:,quasi_order][:,i])
            #ax_s.annotate(i, (n_sites//2 - 4 + (i%2), ordered_quasi[i]))
            
            axb.plot(s_list, ordered_quasi[state_idx] + 10*np.square(np.abs(mags[:,quasi_order][:,state_idx])), c=line.get_color(), lw=1)
            axb.tick_params(left=False, labelleft=False)
            #ax_sq.annotate(i, (n_sites//2 - 4 + (i%2), ordered_quasi[i]))
        axb.set_ylim(axt.get_ylim())
    axt.set_xlabel(r'$ka/\pi$')
    axb.set_xlabel(r'site index $j$')
    fig.tight_layout()
    fig.align_ylabels()
    fig.savefig('modsqp_and_x.pdf')


# In[1380]:


gammas = np.geomspace(1e-5, 0.99, 1024)
Omegacrit = np.sqrt(2*(1-np.square(gammas)))/gammas
big, small = dc.n_both_xis_from_gamma_list(10, gammas)

with umd_small_cxt:
    fig = plt.figure(figsize=(5.8,3))
    ax = fig.add_subplot(111)
    ax.loglog(gammas,Omegacrit,'k')
    ax.set_ylabel('$\\Omega_c/\\omega_T$')
    ax.set_xlabel('$\\gamma$')
    plt.xlim(1e-5, 0.99)
    fig.tight_layout()
    fig.savefig('omega_crit_loglog.pdf', dpi=600)


# In[1381]:


gammas = np.geomspace(1e-5, 0.99, 1024)
Omegacrit = np.sqrt(2*(1-np.square(gammas)))/gammas
big, small = dc.n_both_xis_from_gamma_list(10, gammas)

with umd_small_cxt:
    fig = plt.figure(figsize=(5.8,3))
    # small
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(kelly_cycler)
    for idx in np.arange(11):
        line, = plt.loglog(gammas, np.abs(small[:,idx] - big[:,idx]*np.square(Omegacrit)), label=idx)
    ax.set_ylabel('$|(\\xi_n - \\Xi_n \\Omega_c^2)/\\omega_T^2|$')
    ax.set_xlabel('$\\gamma$')
    plt.xlim(1e-5,1)
    plt.ylim(1e-6, 1e6)
    fig.tight_layout()
    fig.savefig('omega_drivemag_loglog.pdf', dpi=600)


# In[1378]:


gammas = np.linspace(1e-5, 0.99, 1024)
big, small = dc.n_both_xis_from_gamma_list(10, gammas)
n_terms = 8
with umd_small_cxt:
    fig = plt.figure(figsize=(5.8,2.5))
    ax = fig.add_subplot(132)
    ax.set_prop_cycle(kelly_cycler)
    for idx in np.arange(n_terms):
        line, = plt.loglog(gammas, np.abs(big[:,idx]), label=idx)
        if idx == 0:
            ax.annotate(idx,
                xy=(gammas[110], np.abs(big[110, idx])), xycoords='data',
                xytext=(5.e-5, 10.**(-0.5*(idx+1))),
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="angle,angleA=0,angleB=135,rad=10"))
        if idx == 1:
            ax.annotate(idx,
                xy=(gammas[18], np.abs(big[18, idx])), xycoords='data',
                xytext=(5.e-5, 10.**(-0.5*(idx+1))),
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="angle,angleA=0,angleB=135,rad=10"))
        if idx == 2:
            ax.annotate(idx,
                xy=(gammas[60 - 15*idx], np.abs(big[60 - 15*idx, idx])), xycoords='data',
                xytext=(5.e-5, 10.**(-0.5*(idx+1))),
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="angle,angleA=0,angleB=135,rad=10"))
    ax.set_ylabel('$|\\Xi_n| = |(\\kappa\\ddot\\kappa/\\Omega^2)_n|$')
    ax.set_xlabel('$\\gamma$')
    plt.xlim(1e-5,1)
    plt.ylim(1e-6, 10)

    # small
    ax = fig.add_subplot(133)
    ax.set_prop_cycle(kelly_cycler)
    for idx in np.arange(n_terms):
        line, = plt.loglog(gammas, np.abs(small[:,idx]), label=idx)
    ax.set_ylabel('$|\\xi_n| = |(\\kappa^{2})_n|$')
    ax.set_xlabel('$\\gamma$')
    plt.xlim(1e-5,1)
    plt.ylim(1e-6, 10)

    # Chis
    chis = np.empty((3, len(gammas)))
    for idx, gamma in enumerate(gammas):
        chis[:, idx] = dc.all_chis(dc.y(gamma))
    ax = fig.add_subplot(131)
    ax.set_prop_cycle(kelly_cycler)
    plt.loglog(gammas, chis.T)
    ax.set_ylabel('$\\chi_n = (\\kappa^{-2})_n$')
    ax.set_xlabel('$\\gamma$')
    plt.xlim(1e-5,1)
    plt.ylim(1e-6, 10)
    fig.tight_layout()
    fig.savefig('fourier_loglog.pdf', dpi=600)


# In[ ]:




