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


base_dir = os.path.join('/media', 'simulationData', 'BreathingLattice', 'floquet', 'small_lattice', 'critical')


# In[4]:


from uuid import uuid4
import git_tools
from datetime import datetime, timezone
# Test metadata crap
test=uuid4()
print(test.int, str(test))
print(git_tools.git_version(), git_tools.git_describe())
nowutc = datetime.now(timezone.utc)
print(int(nowutc.timestamp()), nowutc.astimezone().isoformat(sep=' ', timespec='seconds'))


# In[5]:


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


# In[6]:


done_flag_path = os.path.join(base_dir, 'done_flag.lck')
if os.path.exists(done_flag_path):
    os.remove(done_flag_path)
else:
    print("No done flag to clear")


# In[7]:


gamma_list = np.geomspace(1e-5, 1e-1, 256, endpoint=True)
gamma_scan_defaults = {'trap_freq':20.0, 'lattice_depth':1.5e3, 'site_spacing_um':1.3,
                             'gamma':0.005,
                             'n_sites':65, 'floquet_m_radius':4, 'sort':False}
scan_name_template = 'critical_scan_n65_m{:d}'
min_m = 33;
max_m = 34;


# In[8]:


get_ipython().run_cell_magic('time', '', "for this_rad in progbar(range(min_m, max_m+1)):\n    gamma_scan_defaults['floquet_m_radius'] = this_rad\n    this_stem = scan_name_template.format(this_rad)\n    %time do_1D_scan(gamma_scan_defaults, 'gamma', gamma_list, this_stem, base_dir)")


# In[9]:


base_dir = os.path.join('/media', 'simulationData', 'BreathingLattice', 'floquet', 'small_lattice', 'critical')
gamma_list = np.geomspace(1e-5, 1e-1, 256, endpoint=True)
gamma_scan_defaults = {'trap_freq':20.0, 'lattice_depth':1.5e3, 'site_spacing_um':1.3,
                             'gamma':0.005,
                             'n_sites':33, 'floquet_m_radius':4, 'sort':False}
scan_name_template = 'critical_scan_n33_m{:d}'
min_m = 67;
max_m = 68;


# In[10]:


get_ipython().run_cell_magic('time', '', "for this_rad in progbar(range(min_m, max_m+1)):\n    gamma_scan_defaults['floquet_m_radius'] = this_rad\n    this_stem = scan_name_template.format(this_rad)\n    %time do_1D_scan(gamma_scan_defaults, 'gamma', gamma_list, this_stem, base_dir)")


# In[11]:


open(done_flag_path, 'a').close()


# In[ ]:




