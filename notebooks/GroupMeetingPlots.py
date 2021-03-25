#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import drive_coefficients as dc
from labellines import labelLine, labelLines


# In[64]:


font = {'size'   : 10}
matplotlib.rc('font', **font,)
matplotlib.rc('text', usetex=True)
outdir = '/home/zachsmith/Desktop/groupPlots/'
kelly_colors = ['#F3C300', '#875692', '#F38400', '#A1CAF1', '#BE0032', '#C2B280', '#848482', '#008856', '#E68FAC', '#0067A5', '#F99379', '#604E97', '#F6A600', '#B3446C', '#DCD300', '#882D17', '#8DB600', '#654522', '#E25822', '#2B3D26']
kelly_also = ['#FFB300', '#803E75', '#FF6800',
              '#A6BDD7', '#C10020', '#CEA262',
              '#817066', '#007D34', '#F6768E',
              '#00538A', '#FF7A5C', '#53377A',
              '#FF8E00']
from cycler import cycler
kelly_cycler = (cycler(color=['#FFB300', '#803E75', '#FF6800',
              '#A6BDD7', '#C10020', '#CEA262',
              '#817066', '#007D34']) + cycler(linestyle=['-', '--', ':', '-.', '-', '--', ':', '-.']))
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)


# In[70]:


locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8, 1.0),numticks=12)
locmaj = matplotlib.ticker.LogLocator(base=10.0,numticks=12)
gammas = np.geomspace(1e-5, 0.5,1024)
big, small = dc.n_both_xis_from_gamma_list(10, gammas)
fig = plt.figure(figsize=(6,3))
ax1 = fig.add_subplot(131)
ax = fig.add_subplot(132, sharey=ax1)
all_lines = [None]*12
for idx in np.arange(11):
    all_lines[idx], = plt.loglog(gammas, np.abs(big[:,idx]), label=idx,
                                 ls=(':' if big[1,idx] < 0 else 'solid'), c=kelly_also[idx])
    if idx < 3:
        ax.annotate(idx,
            xy=(gammas[700 - 50*idx], np.abs(big[700 - 50*idx, idx])), xycoords='data',
            xytext=(2e-5, 10.**(-(1+idx))),
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="angle,angleA=0,angleB=135,rad=10"))
labelLines(all_lines, align=False)
ax.set_title('$|\\Xi_n| = |(\\kappa\\ddot\\kappa/\\Omega^2)_n|$')
ax.set_xlabel('$\\gamma$')
plt.xlim(1e-5,0.5)
plt.ylim(1e-12, 10)
plt.setp(ax.get_yticklabels(), visible=False)
ax.tick_params(axis='both', which='both', direction='in', bottom=True, left=True, top=True, right=True)
ax.xaxis.set_minor_locator(locmin)
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

# small
ax = fig.add_subplot(133)
ax.set_prop_cycle(kelly_cycler)
for idx in np.arange(8):
    all_lines[idx], = plt.loglog(gammas, np.abs(small[:,idx]), label=idx)
                       #ls=(':' if big[1,idx] < 0 else 'solid'), c=kelly_colors[idx])
labelLines(all_lines, align=False)
for idx in np.arange(4):
    plt.loglog(gammas, np.flipud(np.abs(small[:,idx+1])), label=idx, c='k')
                       #ls=(':' if big[1,idx] < 0 else 'solid'), c=kelly_colors[idx])

ax.set_title('$|\\xi_n| = |(\\kappa^{2})_n|$')
ax.set_xlabel('$\\gamma$')
plt.xlim(1e-5,0.5)
plt.ylim(1e-12, 10)
plt.setp(ax.get_yticklabels(), visible=False)
ax.tick_params(axis='both', which='both', direction='in', bottom=True, left=True, top=True, right=True)

# Chis
chis = np.empty((3, len(gammas)))
for idx, gamma in enumerate(gammas):
    chis[:, idx] = dc.all_chis(dc.y(gamma))
ax1.loglog(gammas, chis.T)
ax1.set_title('$\\chi_n = (\\kappa^{-2})_n$')
ax1.set_xlabel('$\\gamma$')
plt.xlim(1e-5,0.5)
plt.ylim(1e-12, 10)
ax.xaxis.set_minor_locator(locmin)
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax1.xaxis.set_minor_locator(locmin)
ax1.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax1.tick_params(axis='both', which='both', direction='in', bottom=True, left=True, top=True, right=True)
fig.tight_layout()
plt.subplots_adjust(wspace=0.0)
fig.savefig(outdir + 'fourier_loglog.pdf', dpi=300, pad_inches=0)


# In[7]:


get_ipython().run_line_magic('pinfo', 'labelLines')


# In[4]:


gammas = np.linspace(1e-5, 0.99, 1024)
big, small = dc.n_both_xis_from_gamma_list(10, gammas)
fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(132)
for idx in np.arange(11):
    line, = plt.semilogy(gammas, np.abs(big[:,idx]), label=idx)
    if idx < 3:
        ax.annotate(idx,
            xy=(gammas[500 - 75*idx], np.abs(big[500 - 75*idx, idx])), xycoords='data',
            xytext=(.1, 10.**(-0.5*(idx-1))),
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="angle,angleA=0,angleB=135,rad=10"))
ax.set_ylabel('$|\\Xi_n| = |(\\kappa\\ddot\\kappa/\\Omega^2)_n|$')
ax.set_xlabel('$\\gamma$')
plt.xlim(1e-5,1)
plt.ylim(1e-6, 10)

# small
ax = fig.add_subplot(133)
for idx in np.arange(11):
    line, = plt.semilogy(gammas, np.abs(small[:,idx]), label=idx)
ax.set_ylabel('$|\\xi_n| = |(\\kappa^{2})_n|$')
ax.set_xlabel('$\\gamma$')
plt.xlim(1e-5,1)
plt.ylim(1e-6, 10)

# Chis
chis = np.empty((3, len(gammas)))
for idx, gamma in enumerate(gammas):
    chis[:, idx] = dc.all_chis(dc.y(gamma))
ax = fig.add_subplot(131)
plt.semilogy(gammas, chis.T)
ax.set_ylabel('$\\chi_n = (\\kappa^{-2})_n$')
ax.set_xlabel('$\\gamma$')
plt.xlim(1e-5,1)
plt.ylim(1e-6, 10)
fig.tight_layout()
fig.savefig(outdir + 'fourier_semilog.png', dpi=150)


# In[5]:


from mpl_toolkits.mplot3d import Axes3D  

def lattice(x, t):
    return -np.cos(2.0*np.pi*(1.0 + 0.1*np.sin(2*np.pi*t))*x)

fig = plt.figure(figsize=(14,7))
ax = fig.add_subplot(121, projection='3d')
x = np.arange(-3., 3., 0.01)
y = np.arange(0.00, 1.0, 0.01)
X, Y = np.meshgrid(x, y)
zs = np.array(lattice(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)

xl = x.copy()
yl = np.zeros_like(x)
zl = lattice(xl, yl)
#yl -= 0.05

ax.plot(xl, yl, zl, c='k', lw=2, zorder=10)
ax.plot_surface(X, Y, Z, cmap='RdBu', rcount=200, ccount=300, vmin=-1.25, vmax=1.25, alpha=1.0)

ax.view_init(elev=35., azim=-75)
ax.set_xlabel('$x/a$')
ax.set_ylabel('$t/T$')
ax.set_zlabel('$V/V_0$')

ax = fig.add_subplot(122)
quad = ax.pcolormesh(X, Y, Z, cmap='RdBu', vmin=-1., vmax=1.)

ax.set_xlabel('$x/a$')
ax.set_ylabel('$t/T$')
plt.colorbar(quad, ax=ax)
fig.tight_layout()
fig.savefig(outdir + 'potential.png', dpi=150)


# In[6]:


gammas = np.geomspace(1e-5, 0.99, 1024)
Omegacrit = np.sqrt(2*(1-np.square(gammas)))/gammas
big, small = dc.n_both_xis_from_gamma_list(10, gammas)

fig = plt.figure(figsize=(14,7))
ax = fig.add_subplot(121)
ax.semilogy(gammas,Omegacrit)
ax.set_ylabel('$\\Omega_c/\\omega_T$')
ax.set_xlabel('$\\gamma$')
plt.xlim(1e-5, 0.99)

# small
ax = fig.add_subplot(122)
for idx in np.arange(11):
    line, = plt.semilogy(gammas, np.abs(small[:,idx] - big[:,idx]*np.square(Omegacrit)), label=idx)
ax.set_ylabel('$|(\\xi_n - \\Xi_n \\Omega_c^2)/\\omega_T^2|$')
ax.set_xlabel('$\\gamma$')
plt.xlim(1e-5,1)
plt.ylim(1e-6, 100)

fig.tight_layout()
fig.tight_layout()
fig.savefig(outdir + 'omega_crit_semilog.png', dpi=150)


# In[7]:


gammas = np.geomspace(1e-5, 0.99, 1024)
Omegacrit = np.sqrt(2*(1-np.square(gammas)))/gammas
big, small = dc.n_both_xis_from_gamma_list(10, gammas)

fig = plt.figure(figsize=(14,7))
ax = fig.add_subplot(121)
ax.loglog(gammas,Omegacrit)
ax.set_ylabel('$\\Omega_c/\\omega_T$')
ax.set_xlabel('$\\gamma$')
plt.xlim(1e-5, 0.99)

# small
ax = fig.add_subplot(122)
for idx in np.arange(11):
    line, = plt.loglog(gammas, np.abs(small[:,idx] - big[:,idx]*np.square(Omegacrit)), label=idx)
ax.set_ylabel('$|(\\xi_n - \\Xi_n \\Omega_c^2)/\\omega_T^2|$')
ax.set_xlabel('$\\gamma$')
plt.xlim(1e-5,1)
plt.ylim(1e-6, 1e6)
fig.tight_layout()
fig.savefig(outdir + 'omega_crit_loglog.png', dpi=150)


# In[24]:


gammas = np.geomspace(1e-5, 0.99, 1024)
Omegadc = np.sqrt(2*(1-np.square(gammas)))
Omegadc2 = np.sqrt(2*(1-np.square(gammas))*(4.0+np.square(gammas))/(2.0+3.0*np.square(gammas)))
big, small = dc.n_both_xis_from_gamma_list(10, gammas)
chis = np.zeros((11, len(gammas)))
for idx, gamma in enumerate(gammas):
    chis[:3, idx] = dc.all_chis(dc.y(gamma))

fig = plt.figure(figsize=(14,7))
ax = fig.add_subplot(121)
ax.plot(gammas,Omegadc, 'k')
ax.plot(gammas,Omegadc2, ':k')
ax.set_ylabel('$\\Omega_{DC}/\\omega_T$')
ax.set_xlabel('$\\gamma$')
plt.xlim(1e-5, 0.99)

# small
ax = fig.add_subplot(122)
for idx in np.arange(11):
    line, = plt.semilogy(gammas, np.abs(small[:,idx] - big[:,idx]*np.square(Omegadc)), label=idx)
    plt.semilogy(gammas, np.abs(small[:,idx] - big[:,idx]*np.square(Omegadc2)
                            - chis[idx]*(small[:,0] - big[:,0]*np.square(Omegadc2))), ls=':', c=line.get_color())
ax.set_ylabel('$|(\\xi_n - \\Xi_n \\Omega_{DC}^2)/\\omega_T^2|$')
ax.set_xlabel('$\\gamma$')
plt.xlim(1e-5, 1)
plt.ylim(1e-6, 100)
fig.tight_layout()
fig.savefig(outdir + 'omegadc_semilog.png', dpi=150)


# In[26]:


gammas = np.geomspace(1e-5, 0.99, 1024)
Omegadc = np.sqrt(2*(1-np.square(gammas)))
Omegadc2 = np.sqrt(2*(1-np.square(gammas))*(4.0+np.square(gammas))/(2.0+3.0*np.square(gammas)))
big, small = dc.n_both_xis_from_gamma_list(10, gammas)
chis = np.zeros((11, len(gammas)))
for idx, gamma in enumerate(gammas):
    chis[:3, idx] = dc.all_chis(dc.y(gamma))

fig = plt.figure(figsize=(14,7))
ax = fig.add_subplot(121)
ax.plot(gammas,Omegadc, 'k')
ax.plot(gammas,Omegadc2, ':k')
ax.set_ylabel('$\\Omega_{DC}/\\omega_T$')
ax.set_xlabel('$\\gamma$')
plt.xlim(1e-5, 0.99)

# small
ax = fig.add_subplot(122)
for idx in np.arange(11):
    line, = plt.loglog(gammas, np.abs(small[:,idx] - big[:,idx]*np.square(Omegadc)), label=idx)
    plt.loglog(gammas, np.abs(small[:,idx] - big[:,idx]*np.square(Omegadc2)
                            - chis[idx]*(small[:,0] - big[:,0]*np.square(Omegadc2))), ls=':', c=line.get_color())

ax.set_ylabel('$|(\\xi_n - \\Xi_n \\Omega_{DC}^2)/\\omega_T^2|$')
ax.set_ylabel('$|(\\xi_n - \\Xi_n \\Omega_{DC}^2)/\\omega_T^2|$')
ax.set_xlabel('$\\gamma$')
plt.xlim(1e-5, 1)
plt.ylim(1e-6, 100)
fig.tight_layout()
fig.savefig(outdir + 'omegadc_loglog.png', dpi=150)


# In[ ]:




