#!/usr/bin/env python
# coding: utf-8

# In[1]:


import breathing_direct_floquet as direct
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


try:
    comp_en, comp_pop = lattice_plus_trap_sitespace(2, 256, 0.0357, 0.0009)
except Exception as e:
    logger.exception("Failed to build sitespace basis")
    #raise bury the exception, because we want to continue


# In[5]:


required_params = frozenset({'trap_freq', 'lattice_depth', 'site_spacing_um',
                             'drive_freq', 'gamma',
                             'n_sites', 'floquet_m_radius', 'sort'})

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
            for result_idx, scan_value in enumerate(scan_array):
                these_settings = {**default_copy, scan_param:scan_value}
                logger.info('Starting iteration {}/{} with parameters {}'.format(result_idx + 1, len(scan_array), these_settings))
                f_en, f_states = direct.diagonalize_floquet_system(**these_settings)
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


# In[6]:


get_ipython().run_cell_magic('time', '', "base_dir = os.path.join('/media', 'simulationData', 'BreathingLattice', 'floquet')\nomega_scan_defaults = {'trap_freq':20.0, 'lattice_depth':3.0e3, 'site_spacing_um':1.3,\n                             'drive_freq':1e3, 'gamma':0.005,\n                             'n_sites':257, 'floquet_m_radius':4, 'sort':False}\nomega_list = np.linspace(100.0, 5000.0, 490*2+1, endpoint=True)\ndo_1D_scan(omega_scan_defaults, 'drive_freq', omega_list, 'direct_drive_scan', base_dir)")


# In[12]:


get_ipython().run_cell_magic('time', '', "base_dir = os.path.join('/media', 'simulationData', 'BreathingLattice', 'floquet')\ngamma_scan_defaults = {'trap_freq':20.0, 'lattice_depth':3.0e3, 'site_spacing_um':1.3,\n                             'drive_freq':1e3, 'gamma':0.005,\n                             'n_sites':257, 'floquet_m_radius':4, 'sort':False}\ngamma_list = np.linspace(0.001, 0.01, 901, endpoint=True)\ndo_1D_scan(gamma_scan_defaults, 'gamma', gamma_list, 'direct_gamma_scan', base_dir)")


# In[13]:


get_ipython().run_cell_magic('time', '', "base_dir = os.path.join('/media', 'simulationData', 'BreathingLattice', 'floquet')\ngamma_scan_defaults = {'trap_freq':20.0, 'lattice_depth':5.0e3, 'site_spacing_um':1.3,\n                             'drive_freq':1e3, 'gamma':0.005,\n                             'n_sites':257, 'floquet_m_radius':4, 'sort':False}\ngamma_list = np.linspace(0.001, 0.01, 451, endpoint=True)\ndo_1D_scan(gamma_scan_defaults, 'gamma', gamma_list, 'direct_gamma_deep_scan', base_dir)")


# In[8]:


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


# In[14]:


with h5py.File(os.path.join(base_dir, 'direct_gamma_scan', 'direct_gamma_scan_out.h5'), 'r', libver="latest") as f:
    fig = plt.figure(figsize=(8,10.5))
    ax = fig.add_subplot(111)
    gen_energy_plot(ax, f['scan_values'][:], f['floquet_energy'][:], -50, 50, marker=',', ls='None', c='k', modulus=1e3)
    fig.tight_layout()


# In[15]:


with h5py.File(os.path.join(base_dir, 'direct_gamma_deep_scan', 'direct_gamma_deep_scan_out.h5'), 'r', libver="latest") as f:
    fig = plt.figure(figsize=(8,10.5))
    ax = fig.add_subplot(111)
    gen_energy_plot(ax, f['scan_values'][:], f['floquet_energy'][:], -100, 100, marker=',', ls='None', c='k', modulus=1e3)
    fig.tight_layout()


# In[16]:


with h5py.File(os.path.join(base_dir, 'direct_drive_scan', 'direct_drive_scan_out.h5'), 'r', libver="latest") as f:
    fig = plt.figure(figsize=(8,10.5))
    ax = fig.add_subplot(111)
    gen_energy_plot(ax, f['scan_values'], f['floquet_energy'], -50, 50, marker=',', ls='None', c='k', modulus=f['scan_values'])
    fig.tight_layout()


# In[ ]:




