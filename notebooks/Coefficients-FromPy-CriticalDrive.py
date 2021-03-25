#!/usr/bin/env python
# coding: utf-8

# In[1]:


import breathing_critical_floquet as critical
import numpy as np

import copy
import os
import h5py

from matplotlib import pyplot as plt

from zss_progbar import log_progress as progbar


# In[2]:


import logging
logger = logging.getLogger('floquet')


# In[3]:


from uuid import uuid4
import git_tools
from datetime import datetime, timezone
# Test metadata crap
test=uuid4()
print(test.int, str(test))
print(git_tools.git_version(), git_tools.git_describe())
nowutc = datetime.now(timezone.utc)
print(int(nowutc.timestamp()), nowutc.astimezone().isoformat(sep=' ', timespec='seconds'))


# In[4]:


required_params = frozenset({'trap_freq', 'lattice_depth', 'site_spacing_um',
                             'gamma', 'n_sites', 'floquet_m_radius', 'sort'})

def do_1D_scan(default_dict, scan_param, scan_array, filename_stem, output_directory, save_h0=True):
    if not (required_params <= set(default_dict)):
        missing_params = list(required_params - set(default_dict))
        raise KeyError('Missing required entries from default_dict: {}'.format(missing_params))
    if scan_param not in default_dict:
        raise KeyError('Unknown scan_param provided: {}'.format(scan_param))
    # Build paths and create directories if necessary
    odir = os.path.join(os.path.abspath(os.path.expanduser(output_directory)), filename_stem)
    try:
        os.makedirs(odir)
    except FileExistsError:
        pass # No warning
    save_h0_file = os.path.join(odir, filename_stem + '_h0.h5')
    save_floquet_file = os.path.join(odir, filename_stem + '_out.h5')
    run_log_file = os.path.join(odir, filename_stem + '.log')
    print('Logfile: {}'.format(run_log_file))
    # Setup logging
    logger.setLevel(logging.INFO)
    fhandler = logging.FileHandler(filename=run_log_file, mode='w')
    logger.addHandler(fhandler)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    # Make a copy so defaults can't get changed out from under us
    default_copy = copy.deepcopy(default_dict)
    n_total_states = default_copy['n_sites']*(2*default_copy['floquet_m_radius'] + 1)
    logger.info('Starting with code from git {}'.format(git_tools.git_describe()))
    start_time = datetime.now()
    try:
        with h5py.File(save_floquet_file, 'w', libver="latest") as f:
            # Setup main file
            logger.info('Saving to files {} and {}'.format(save_floquet_file, save_h0_file))
            scan_dset = f.create_dataset('scan_values', data=scan_array, compression='gzip')
            scan_dset.attrs['scan_param'] = str(scan_param)
            energy_dset = f.create_dataset('floquet_energy', (len(scan_array), n_total_states), dtype=np.float, compression='gzip', chunks=(1, n_total_states))
            states_dset = f.create_dataset('floquet_state',  (len(scan_array), n_total_states, n_total_states), dtype=np.complex, compression='gzip', chunks=(1, 256, 256))
            drive_dset  = f.create_dataset('critcal_drive', (len(scan_array),), dtype=np.float)
            for result_idx, scan_value in enumerate(scan_array):
                these_settings = {**default_copy, scan_param:scan_value}
                logger.info('Starting iteration {}/{} with parameters {}'.format(result_idx + 1, len(scan_array), these_settings))
                c_drive, f_en, f_states = critical.diagonalize_critical_floquet_system(**these_settings)
                try:
                    drive_dset[result_idx] = c_drive
                except KeyboardInterrupt:
                    raise
                except:
                    logger.exception('Saving critical drive failed')
                try:
                    energy_dset[result_idx, :] = f_en
                except KeyboardInterrupt:
                    raise
                except:
                    logger.exception('Saving energies failed')
                try:
                    states_dset[result_idx, ...] = f_states.astype(np.complex)
                except KeyboardInterrupt:
                    raise
                except:
                    logger.exception('Saving states failed')
        logger.info('Finished with entire scan')
        stop_time = datetime.now()
        logger.info('Scan took {} for {} runs (avg. {} per run)'.format(stop_time-start_time,
                                                                        len(scan_array),
                                                                        (stop_time-start_time)/len(scan_array)))
    except KeyboardInterrupt:
        logger.error('User interrupted scan, aborting')
        raise
    except:
        logger.exception('Failed to finish scan due to an exception.')
        raise
    finally:
        fhandler.flush()
        fhandler.close()
        logger.removeHandler(fhandler)


# In[5]:


get_ipython().run_cell_magic('time', '', "base_dir = os.path.join('/media', 'simulationData', 'BreathingLattice', 'floquet')\ngamma_scan_defaults = {'trap_freq':20.0, 'lattice_depth':3.0e3, 'site_spacing_um':1.3,\n                             'gamma':0.005,\n                             'n_sites':257, 'floquet_m_radius':4, 'sort':False}\ngamma_list = np.linspace(5e-5, 0.1, 200, endpoint=True)\ndo_1D_scan(gamma_scan_defaults, 'gamma', gamma_list, 'critical_gamma_corrected_scan', base_dir)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "base_dir = os.path.join('/media', 'simulationData', 'BreathingLattice', 'floquet')\ngamma_scan_defaults = {'trap_freq':20.0, 'lattice_depth':3.0e3, 'site_spacing_um':1.3,\n                             'gamma':0.005,\n                             'n_sites':257, 'floquet_m_radius':4, 'sort':False}\ngamma_list = np.linspace(5e-5, 0.1, 2000, endpoint=True)\ndo_1D_scan(gamma_scan_defaults, 'gamma', gamma_list, 'critical_gamma_scan', base_dir)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "base_dir = os.path.join('/media', 'simulationData', 'BreathingLattice', 'floquet')\ngamma_scan_defaults = {'trap_freq':20.0, 'lattice_depth':3.0e3, 'site_spacing_um':1.3,\n                             'gamma':0.005,\n                             'n_sites':513, 'floquet_m_radius':4, 'sort':False}\ngamma_list = np.linspace(2e-4, 0.1, 500, endpoint=True)\ndo_1D_scan(gamma_scan_defaults, 'gamma', gamma_list, 'critical_gamma_wider_scan', base_dir)")


# In[15]:


def gen_energy_plot(ax, x_array, y_matrix, e_min, e_max, modulus=None, mod_offset=0, **kwargs):
    """ Plot into ax points from rows of y_matrix whose x-coordinate comes from matching x_array, e_min<y<e_max
       
    """
    for idx, this_x in enumerate(x_array):
        if modulus is None:
            y_row = y_matrix[idx]
        elif hasattr(modulus, "__len__"):
            y_row = np.mod(y_matrix[idx] + modulus[idx]/2, modulus[idx]) - modulus[idx]/2 + mod_offset*modulus[idx]
        else:
            y_row = np.mod(y_matrix[idx] + modulus/2, modulus) - modulus/2 + mod_offset*modulus
        y_in_range = y_row[np.logical_and(y_row >= e_min, y_row <= e_max)]
        ax.plot(np.repeat(this_x, len(y_in_range)), y_in_range, **kwargs)
    ax.set_xlim(2*x_array[0] - x_array[1], 2*x_array[-1] - x_array[-2])
    ax.set_ylim(e_min, e_max)


# In[1]:


with h5py.File(os.path.join(base_dir, 'critical_gamma_scan', 'critical_gamma_scan_out.h5'), 'r', libver="latest") as f:
    fig = plt.figure(figsize=(8, 10.5))
    ax = fig.add_subplot(111)
    gen_energy_plot(ax, f['scan_values'][:], f['floquet_energy'][:], -50, 50, marker=',', ls='None', c='k', modulus=f['critcal_drive'][:])
fig.tight_layout()


# In[16]:


with h5py.File(os.path.join(base_dir, 'critical_gamma_scan', 'critical_gamma_scan_out.h5'), 'r', libver="latest") as f:
    critical_drive_freqs=f['critcal_drive'][:]
    matching_gammas = f['scan_values'][:]
plt.loglog(matching_gammas, critical_drive_freqs)


# In[12]:


with h5py.File(os.path.join(base_dir, 'critical_gamma_wider_scan', 'critical_gamma_wider_scan_out.h5'), 'r', libver="latest") as f:
    fig = plt.figure(figsize=(8,10.5))
    ax = fig.add_subplot(111)
    gen_energy_plot(ax, f['scan_values'][:], f['floquet_energy'][:], -50, 50, marker=',', ls='None', c='k', modulus=f['critcal_drive'][:])
    fig.tight_layout()


# ## Scan Ms around

# In[10]:


base_dir = os.path.join('/media', 'simulationData', 'BreathingLattice', 'floquet', 'mscan')
gamma_list = np.geomspace(1e-5, 1e-1, 256, endpoint=True)
gamma_scan_defaults = {'trap_freq':20.0, 'lattice_depth':1.5e3, 'site_spacing_um':1.3,
                             'gamma':0.005,
                             'n_sites':129, 'floquet_m_radius':4, 'sort':False}
scan_name_template = 'critical_shallow_gamma_scan_m{:d}'
min_m = 14;
max_m = 17;


# In[11]:


get_ipython().run_cell_magic('time', '', "for this_rad in progbar(range(min_m, max_m+1)):\n    gamma_scan_defaults['floquet_m_radius'] = this_rad\n    this_stem = scan_name_template.format(this_rad)\n    %time do_1D_scan(gamma_scan_defaults, 'gamma', gamma_list, this_stem, base_dir)")


# In[ ]:


base_dir = os.path.join('/media', 'simulationData', 'BreathingLattice', 'floquet', 'mscan')
gamma_list = np.linspace(0.1002, 0.5, 256, endpoint=True)
gamma_scan_defaults = {'trap_freq':20.0, 'lattice_depth':1.5e3, 'site_spacing_um':1.3,
                             'gamma':0.005,
                             'n_sites':129, 'floquet_m_radius':4, 'sort':False}
scan_name_template = 'critical_long_shallow_gamma_scan_m{:d}'
min_m = 14;
max_m = 17;


# In[9]:


get_ipython().run_cell_magic('time', '', "for this_rad in progbar(range(min_m, max_m+1)):\n    gamma_scan_defaults['floquet_m_radius'] = this_rad\n    this_stem = scan_name_template.format(this_rad)\n    %time do_1D_scan(gamma_scan_defaults, 'gamma', gamma_list, this_stem, base_dir)")


# ## End M's Block

# In[28]:


with h5py.File(os.path.join(base_dir, 'critical_shallow_gamma_scan', 'critical_shallow_gamma_scan_out.h5'), 'r', libver="latest") as f:
    fig = plt.figure(figsize=(8,10.5))
    ax = fig.add_subplot(111)
    gen_energy_plot(ax, f['scan_values'][::], f['floquet_energy'][::], -100, 100, marker=',', ls='None', c='k', modulus=f['critcal_drive'][::])
fig.tight_layout()
fig.savefig('/home/zachsmith/Desktop/test_small_param.png',dpi=300)


# In[93]:


subset = slice(0, 500, 1)
with h5py.File(os.path.join(base_dir, 'critical_long_shallow_gamma_scan', 'critical_long_shallow_gamma_scan_out.h5'), 'r', libver="latest") as f:
    fig = plt.figure(figsize=(8,10.5))
    ax = fig.add_subplot(111)
    gen_energy_plot(ax, f['scan_values'][subset], f['floquet_energy'][subset], -50, 50, marker=',', ls='None', c='k', modulus=f['critcal_drive'][subset])
    fig.tight_layout()


# In[18]:


base_dir = os.path.join('/media', 'simulationData', 'BreathingLattice', 'floquet')
fig = plt.figure(figsize=(8,10.5))
ax = fig.add_subplot(111)
with h5py.File(os.path.join(base_dir, 'critical_shallow_gamma_scan', 'critical_shallow_gamma_scan_out.h5'), 'r', libver="latest") as f:
    gen_energy_plot(ax, f['scan_values'][:], f['floquet_energy'][:], -100, 100, marker=',', ls='None', c='k', modulus=None)
with h5py.File(os.path.join(base_dir, 'critical_long_shallow_gamma_scan', 'critical_long_shallow_gamma_scan_out.h5'), 'r', libver="latest") as f:
    gen_energy_plot(ax, f['scan_values'][:], f['floquet_energy'][:], -100, 100, marker=',', ls='None', c='k',
                    modulus=None)
plt.xlim(0,0.5)


# In[24]:


subset = slice(0, -1, 1)
with h5py.File(os.path.join(base_dir, 'critical_long_shallow_gamma_scan', 'critical_long_shallow_gamma_scan_out.h5'), 'r', libver="latest") as f:
    fig = plt.figure(figsize=(8,10.5))
    ax = fig.add_subplot(111)
    gen_energy_plot(ax, f['scan_values'][subset], f['floquet_energy'][subset], -50, -25, marker=',', ls='None', c='k', modulus=f['critcal_drive'][subset])
    gen_energy_plot(ax, f['scan_values'][subset], f['floquet_energy'][subset], -50, -25, marker=',', ls='None', c='k', modulus=f['critcal_drive'][subset], mod_offset=1)
    gen_energy_plot(ax, f['scan_values'][subset], f['floquet_energy'][subset], -50, -25, marker=',', ls='None', c='k', modulus=f['critcal_drive'][subset], mod_offset=-1)
fig.tight_layout()
fig.savefig('/home/zachsmith/Desktop/test_wide.png',dpi=300)
#fig.set_size_inches(4, 10.5)
plt.xlim(.25,.35)
fig.tight_layout()
fig.savefig('/home/zachsmith/Desktop/test_zoom.png',dpi=100)


# In[25]:


subset = slice(0, -1, 1)
with h5py.File(os.path.join(base_dir, 'critical_long_shallow_gamma_scan', 'critical_long_shallow_gamma_scan_out.h5'), 'r', libver="latest") as f:
    fig = plt.figure(figsize=(8,10.5))
    ax = fig.add_subplot(111)
    gen_energy_plot(ax, f['scan_values'][subset], f['floquet_energy'][subset], 25, 50, marker=',', ls='None', c='k', modulus=f['critcal_drive'][subset])
    gen_energy_plot(ax, f['scan_values'][subset], f['floquet_energy'][subset], 25, 50, marker=',', ls='None', c='k', modulus=f['critcal_drive'][subset], mod_offset=1)
    gen_energy_plot(ax, f['scan_values'][subset], f['floquet_energy'][subset], 25, 50, marker=',', ls='None', c='k', modulus=f['critcal_drive'][subset], mod_offset=-1)
fig.tight_layout()
#fig.savefig('/home/zachsmith/Desktop/test_wide.png',dpi=300)
#fig.set_size_inches(4, 10.5)
plt.xlim(.25,.35)
fig.tight_layout()
fig.savefig('/home/zachsmith/Desktop/test_top_zoom.png',dpi=100)


# In[ ]:




