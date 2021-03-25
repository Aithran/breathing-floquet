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


# In[3]:


base_dir = os.path.join('/media', 'simulationData', 'BreathingLattice', 'floquet', 'small_lattice')
#done_flag_path = os.path.join(base_dir, 'done_flag.lck')
#while not os.path.exists(done_flag_path):
#    time.sleep(60)


# In[4]:


m_re = re.compile(r'.*_n(\d+)_out.h5')
def get_m(file_name):
    return int(m_re.search(file_name).group(1))
base_path = base_dir
data_dirs = [os.path.basename(dirpath[:-1]) for dirpath in glob.glob(base_path + "/h0_*/")]
print(data_dirs)
#data_paths = [os.path.join(base_path, this_dir, this_dir + '_out.h5') for this_dir in data_dirs]
data_paths = sorted([os.path.join(base_path, this_dir, this_dir + '_out.h5') for this_dir in data_dirs], key=get_m)
for idx, path in enumerate(data_paths):
    print('{}:\t{}'.format(idx, os.path.basename(path)))
largest_m_path = data_paths[-1]


# In[5]:


from cycler import cycler
kelly_cycler = (cycler(color=['#FFB300', '#803E75', '#FF6800',
              '#A6BDD7', '#C10020', '#CEA262',
              '#817066', '#007D34']) + cycler(linestyle=['-', '--', ':', '-.', '-', '--', ':', '-.']))


# In[7]:


with umd_small_cxt:
    fig, ax_arr = plt.subplots(1,2, sharey=True, figsize=(5.5,2))
    ax_arr[0].set_ylabel(r"quasienergy $\epsilon$")
    ax_arr[0].set_xlabel(r'$N_\mathrm{sites}=33$')
    ax_arr[1].set_xlabel(r'$N_\mathrm{sites}=45$')
    this_path = data_paths[0]
    print(this_path)
    with h5py.File(this_path, 'r', libver="latest") as f:
        print(os.path.basename(this_path))
        chosen_states = f['floquet_state'][0]
        chosen_quasi = f['floquet_energy'][0]
        chosen_drive = f['critical_drive'][0]
        loaded_energies = f['floquet_energy'][:]
        gammas = f['scan_values'][:]
        omegas = f['critical_drive'][:]

    ax_arr[0].plot(loaded_energies, 'k', lw=1);
    this_path = data_paths[1]
    print(this_path)
    ax_arr[0].set_xticklabels([])
    with h5py.File(this_path, 'r', libver="latest") as f:
        print(os.path.basename(this_path))
        loaded_energies = f['floquet_energy'][:]
        gammas = f['scan_values'][:]
        omegas = f['critical_drive'][:]
    ax_arr[1].set_xticklabels([])
    ax_arr[1].set_yticklabels([])
    ax_arr[1].plot(loaded_energies, 'k', lw=1);
    ax_arr[0].set_xlim(0,10)
    ax_arr[1].set_xlim(0,10)

    fig.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()
    fig.savefig('quasien_h0_boring.pdf', dpi=600, bbox_inches='tight', pad_inches=1/72)


# In[23]:


loaded_energies.min() * np.cos(np.linspace(0,np.pi,45,endpoint=True)) - np.sort(loaded_energies[0])


# In[6]:


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


# In[7]:


this_path = data_paths[1]
with h5py.File(this_path, 'r', libver="latest") as f:
    print(os.path.basename(this_path))
    chosen_states = f['floquet_state'][0]
    chosen_quasi = f['floquet_energy'][0]
    chosen_drive = f['critical_drive'][0]
    loaded_energies = f['floquet_energy'][:]
    gammas = f['scan_values'][:]
    omegas = f['critical_drive'][:]
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


# In[ ]:




