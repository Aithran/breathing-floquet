#!/usr/bin/env python
# coding: utf-8

# In[1085]:


import numpy as np
from matplotlib import pyplot as plt
import h5py
import os


# In[1408]:


base_path = '/home/zachsmith/gitlab/breathing-floquet/notebooks/data'
run_stem = 'drive_scan'
run_stem = 'critical_gamma_scan'
run_stem = 'direct_gamma_deep_scan'
data_file = os.path.join(base_path, run_stem, run_stem + '_out.h5')
print(os.path.basename(data_file))


# In[1409]:


with h5py.File(data_file, 'r', libver="latest") as f:
    states_matrix = f['floquet_state']
    quasi_energies = f['floquet_energy'][:]
    #drives = f['critcal_drive'][:]
    #scaled_quasi = f['floquet_energy'][:]/f['critcal_drive'][:][:,np.newaxis]
    print(states_matrix.shape)


# In[1410]:


print(quasi_energies.shape)


# In[1412]:


plt.plot(quasi_energies)
plt.ylim(-100,900)


# In[1414]:


plt.plot(quasi_energies, ',k')
plt.ylim(-100, 900)
plt.xlim(0,500)


# In[1415]:


def get_cycle_proj_op(n_states, n_floquet):
    return np.tile(np.diag(np.ones(n_states)), (1, n_floquet))
plt.pcolormesh(get_cycle_proj_op(16,3))


# In[1416]:


with h5py.File(data_file, 'r', libver="latest") as f:
    states_matrix = f['floquet_state']
    #central_site
    #col_slice = states_matrix[:,:,2556:2564+1]
    static_center_pops = states_matrix[:,257*4+128,:]
    one_slice = states_matrix[-1]


# In[1417]:


states_above_thresh = np.unique(np.argwhere(np.square(np.abs(static_center_pops)) > 1e-5)[:,1])
print(m_j_from_idx(states_above_thresh).T)
print(get_cycle_proj_op(257,9).shape)
compress = get_cycle_proj_op(257, 9)
print(one_slice.shape)
def to_site_pop(full_pop):
    state = one_slice @ full_pop
    return np.square(np.abs(compress @ state))
print(np.sum(np.square(np.abs(start_state))))


# In[1605]:


which_scan = 0
twopii = np.complex(0,2.0*np.pi)
with h5py.File(data_file, 'r', libver="latest") as f:
    states_matrix = f['floquet_state']
    one_slice = states_matrix[which_scan,...]
    quasi_energies = f['floquet_energy'][:]
    scaled_quasi=quasi_energies
    #drives = f['critcal_drive'][:]
    drives = np.full(len(scaled_quasi[:,0]), 1e3)
    #print(drives[:,np.newaxis].shape, quasi_energies.shape)
    #scaled_quasi = f['floquet_energy'][:]/(drives[:,np.newaxis])
print(scaled_quasi.shape)
compress = get_cycle_proj_op(257, 9)
thing = np.linalg.inv(one_slice) @ one_slice
thing = np.conjugate(one_slice.T) @ one_slice
def to_site_pop(full_pop):
    state = one_slice @ full_pop
    return np.square(np.abs(compress @ state))
def pop_at_period(num_periods, initial_state):
    after_one = np.exp(-num_periods * twopii * scaled_quasi[which_scan]) * initial_state
    return to_site_pop(after_one[:, np.newaxis])
print("1 cycle = {} s".format(1.0/drives[which_scan]))


# In[1540]:


plt.plot(np.sum(np.square(np.abs(one_slice)), axis=0))


# In[1541]:


plt.plot(np.diag(thing))


# In[1542]:


site_offset = 2
n_sites=257
center=n_sites//2
#start_state = np.conjugate(one_slice[n_sites*4+center+site_offset])
#start_state = (compress @ one_slice)[128+site_offset]/9
start_state = one_slice[:,464]
print(start_state.shape, scaled_quasi.shape)
print(np.sum(np.square(np.abs(start_state))))
periods = np.arange(1024)*2
results = np.empty((1024,n_sites))
for idx, per in enumerate(periods):
    results[idx] = pop_at_period(per, start_state).flatten()
plt.plot(np.abs(start_state),'.')
for mult in np.arange(10):
    plt.axvline(mult*n_sites)
print(periods[-1]/drives[which_scan])


# In[1608]:


good_idx = np.where(np.logical_and(quasi_energies[which_scan] > -0.1*drives[which_scan], quasi_energies[which_scan] < drives[which_scan]))[0]
print(drives[which_scan])
for idx in good_idx:
    print("{}:\t{:f}\t{:f}".format(idx, quasi_energies[which_scan, idx], np.mod(quasi_energies[which_scan, idx] + 0.5*drives[which_scan], drives[which_scan])-0.5*drives[which_scan]))


# In[1600]:


which_state = 1018
site_fourier_state = one_slice[:, which_state]
#site_fourier_state = (one_slice[:, 1031] - one_slice[:, 1032])/np.sqrt(2.0)
print(np.sum(np.square(np.abs(site_fourier_state))), site_fourier_state.shape, site_fourier_state[:, np.newaxis].shape)
#plt.plot(np.abs(site_fourier_state),'.')
def time_phasor(t_over_T, scaled_quasi_energy=1000):
    return np.exp(np.complex(0,2.0*np.pi) * t_over_T * (np.repeat(np.arange(-4,5), n_sites) - scaled_quasi_energy))
def site_pop_at_time(t_over_T, floquet_state):
    return compress @ np.diag(time_phasor(t_over_T)) @ floquet_state
#for mult in np.arange(10):
#    plt.axvline(mult*n_sites, ls='dotted', c='k')
#plt.plot(np.square(np.abs(site_pop_at_time(0, site_fourier_state))), '.k')
one_cycle_trace = np.empty((128,257), dtype=np.complex)
ts = np.linspace(0.0,1.0, 128)
for idx, t in enumerate(ts):
    one_cycle_trace[idx] = site_pop_at_time(t, site_fourier_state[:, np.newaxis]).flatten()
#plt.pcolormesh(np.square(np.abs(one_cycle_trace))[:, 128-20:128+20+1])
#plt.colorbar()
#print(quasi_energies[which_scan, which_state])
fig, ax_array = plt.subplots(nrows=2, ncols=2, sharex=False, figsize=(10.5,8))
ax_array[0,0].plot(ts, np.square(np.abs(one_cycle_trace)))
ax_array[0,0].set_xlim(0,1)
ax_array[1,0].axhline(1.0, c='r', ls=':')
ax_array[1,0].plot(ts, np.sum(np.square(np.abs(one_cycle_trace)),axis=1), 'k', label="Total")
ax_array[1,0].set_xlim(0,1)
mesh = ax_array[0,1].pcolormesh(np.square(np.abs(one_cycle_trace))[:, 128-40:128+40+1], cmap='inferno')
plt.colorbar(mesh, ax=ax_array[0,1])

ax_array[1,1].plot(np.imag(site_fourier_state),'.b', ms=3, label='Im(psi)')
ax_array[1,1].plot(np.real(site_fourier_state),'.r', ms=3, label='Re(psi)')
for mult in np.arange(10):
    plt.axvline(mult*n_sites, ls=':', c='k')
#ax_array[1,1].legend()
ax_array[1,1].set_xlim(-1, len(site_fourier_state))

plt.tight_layout()
print(np.sum(np.square(np.abs(one_cycle_trace)))/512.)
fig.savefig('/home/zachsmith/Desktop/multiplot.png', dpi=300)


# In[1559]:


tzero_slice = (compress @ one_slice)


# In[1578]:


center_site = np.zeros(257)
center_site[128] = 1.0
print(np.sum(np.square(np.abs(center_site))))
floquet_pops = (np.conjugate(tzero_slice).T @ center_site)
plt.plot(np.real(floquet_pops), ',')
np.sum(np.square(np.abs(compress @ floquet_pops)))
floquet_pops = floquet_pops/np.sum(np.square(np.abs(floquet_pops)))


# In[1549]:


plt.plot(np.imag(site_fourier_state),'.b', ms=3, label='Im(psi)')
plt.plot(np.real(site_fourier_state),'.r', ms=3, label='Re(psi)')
for mult in np.arange(10):
    plt.axvline(mult*n_sites, ls=':', c='k')
plt.legend()
plt.xlim(-1, len(site_fourier_state))


# In[1348]:


plot_rad=40
plt.pcolormesh(results[:,center+site_offset-plot_rad:center+site_offset+plot_rad+1], vmin=0, vmax=1)
plt.colorbar()


# In[1396]:


plt.plot(np.sum(results[:,:],axis=1))


# In[99]:


above_thresh_floquetstates = np.unique(np.argwhere(np.square(np.abs(col_slice)) > )[:,1])
print(above_thresh_floquetstates)


# In[109]:


def m_j_from_idx(idx):
    return np.divmod(idx, 513) - (np.array((4, 256)) if not hasattr(idx, '__len__') else np.array((4, 256))[:,np.newaxis])
def m_j_label_from_idx(idx):
    return "|{}, {}\u300B".format(*m_j_from_idx(idx))


# In[77]:


m_j_from_idx(above_thresh_floquetstates).T


# In[91]:


for idx in above_thresh_floquetstates:
    plt.plot(np.square(np.abs(col_slice[:, idx, 0])), label=m_j_label_from_idx(idx))
plt.legend()


# In[31]:


fig = plt.figure(figsize=(8,10.5))
plt.pcolormesh(np.square(np.abs(col_slice.T[513*4:513*5])), cmap='inferno')
plt.tight_layout()
fig.savefig('test.png',dpi=500)


# In[38]:


4617/9


# In[795]:


floquet_radius = 4
drive_freq = 1
n_states = 5
m_couplings = np.kron(np.arange(-floquet_radius, floquet_radius+1), np.tile(drive_freq, n_states))
print(m_couplings)


# In[820]:


257*9


# In[1219]:


np.repeat(np.arange(-4,5), 5)


# In[1614]:


which_state = 603
site_fourier_state = one_slice[:, which_state]
#site_fourier_state = (one_slice[:, 1031] - one_slice[:, 1032])/np.sqrt(2.0)
print(np.sum(np.square(np.abs(site_fourier_state))), site_fourier_state.shape, site_fourier_state[:, np.newaxis].shape)
#plt.plot(np.abs(site_fourier_state),'.')
def time_phasor(t_over_T, scaled_quasi_energy=1000):
    return np.exp(np.complex(0,2.0*np.pi) * t_over_T * (np.repeat(np.arange(-4,5), n_sites) - scaled_quasi_energy))
def site_pop_at_time(t_over_T, floquet_state):
    return compress @ np.diag(time_phasor(t_over_T)) @ floquet_state
#for mult in np.arange(10):
#    plt.axvline(mult*n_sites, ls='dotted', c='k')
#plt.plot(np.square(np.abs(site_pop_at_time(0, site_fourier_state))), '.k')
one_cycle_trace = np.empty((128,257), dtype=np.complex)
ts = np.linspace(0.0,1.0, 128)
for idx, t in enumerate(ts):
    one_cycle_trace[idx] = site_pop_at_time(t, site_fourier_state[:, np.newaxis]).flatten()
#plt.pcolormesh(np.square(np.abs(one_cycle_trace))[:, 128-20:128+20+1])
#plt.colorbar()
#print(quasi_energies[which_scan, which_state])
fig, ax_array = plt.subplots(nrows=2, ncols=2, sharex=False, figsize=(15,8))
ax_array[0,0].plot(ts, np.square(np.abs(one_cycle_trace)))
ax_array[0,0].set_xlim(0,1)
ax_array[1,0].axhline(1.0, c='r', ls=':')
ax_array[1,0].plot(ts, np.sum(np.square(np.abs(one_cycle_trace)),axis=1), 'k', label="Total")
ax_array[1,0].set_xlim(0,1)
mesh = ax_array[0,1].pcolormesh(np.square(np.abs(one_cycle_trace))[:, 128-40:128+40+1], cmap='inferno')
plt.colorbar(mesh, ax=ax_array[0,1])

ax_array[1,1].plot(np.imag(site_fourier_state),'.b', ms=3, label='Im(psi)')
ax_array[1,1].plot(np.real(site_fourier_state),'.r', ms=3, label='Re(psi)')
for mult in np.arange(10):
    plt.axvline(mult*n_sites, ls=':', c='k')
#ax_array[1,1].legend()
ax_array[1,1].set_xlim(-1, len(site_fourier_state))

plt.tight_layout()
print(np.sum(np.square(np.abs(one_cycle_trace)))/512.)
fig.savefig('/home/zachsmith/Desktop/groupPlots/badMode.png', dpi=150)


# In[1615]:


which_state = 808
site_fourier_state = one_slice[:, which_state]
#site_fourier_state = (one_slice[:, 1031] - one_slice[:, 1032])/np.sqrt(2.0)
print(np.sum(np.square(np.abs(site_fourier_state))), site_fourier_state.shape, site_fourier_state[:, np.newaxis].shape)
#plt.plot(np.abs(site_fourier_state),'.')
def time_phasor(t_over_T, scaled_quasi_energy=1000):
    return np.exp(np.complex(0,2.0*np.pi) * t_over_T * (np.repeat(np.arange(-4,5), n_sites) - scaled_quasi_energy))
def site_pop_at_time(t_over_T, floquet_state):
    return compress @ np.diag(time_phasor(t_over_T)) @ floquet_state
#for mult in np.arange(10):
#    plt.axvline(mult*n_sites, ls='dotted', c='k')
#plt.plot(np.square(np.abs(site_pop_at_time(0, site_fourier_state))), '.k')
one_cycle_trace = np.empty((128,257), dtype=np.complex)
ts = np.linspace(0.0,1.0, 128)
for idx, t in enumerate(ts):
    one_cycle_trace[idx] = site_pop_at_time(t, site_fourier_state[:, np.newaxis]).flatten()
#plt.pcolormesh(np.square(np.abs(one_cycle_trace))[:, 128-20:128+20+1])
#plt.colorbar()
#print(quasi_energies[which_scan, which_state])
fig, ax_array = plt.subplots(nrows=2, ncols=2, sharex=False, figsize=(15,8))
ax_array[0,0].plot(ts, np.square(np.abs(one_cycle_trace)))
ax_array[0,0].set_xlim(0,1)
ax_array[1,0].axhline(1.0, c='r', ls=':')
ax_array[1,0].plot(ts, np.sum(np.square(np.abs(one_cycle_trace)),axis=1), 'k', label="Total")
ax_array[1,0].set_xlim(0,1)
mesh = ax_array[0,1].pcolormesh(np.square(np.abs(one_cycle_trace))[:, 128-40:128+40+1], cmap='inferno')
plt.colorbar(mesh, ax=ax_array[0,1])

ax_array[1,1].plot(np.imag(site_fourier_state),'.b', ms=3, label='Im(psi)')
ax_array[1,1].plot(np.real(site_fourier_state),'.r', ms=3, label='Re(psi)')
for mult in np.arange(10):
    plt.axvline(mult*n_sites, ls=':', c='k')
#ax_array[1,1].legend()
ax_array[1,1].set_xlim(-1, len(site_fourier_state))

plt.tight_layout()
print(np.sum(np.square(np.abs(one_cycle_trace)))/512.)
fig.savefig('/home/zachsmith/Desktop/groupPlots/okayMode.png', dpi=150)


# In[ ]:




