#!/usr/bin/env python
# coding: utf-8
import harm_lattice as h0
import numpy as np
from matplotlib import pyplot as plt

R_ref = 3.47e3
hbar_omega = 60
V0_ref = 7.4*ER_ref
f_trap_ref = 60
J_ref = 0.0357*ER_ref
harm_param = 0.0009*ER_ref
comp_en, comp_pop = h0.lattice_plus_trap_sitespace(1024, 256, J_ref, harm_param)
scaled_en = comp_en[:35]/(0.0009*np.sqrt(4.0*0.0357/0.0009)) #Their plot is in units of hbar omega*
scaled_en -= scaled_en[0]

plt.plot(scaled_en[:35],'.')
plt.annotate('$n_c$', (16, scaled_en[16]), (-6,-24), arrowprops={'arrowstyle':'-|>'}, textcoords='offset points')

center = comp_pop.shape[1]//2
plt.plot(np.arange(-30,30),comp_pop[5, center-30:center+30], '.', ls='dotted', label='5')
plt.plot(np.arange(-30,30),comp_pop[2, center-30:center+30], 'x', ls='solid', label='2')
plt.ylim(-0.5,0.5)
plt.legend()

center = comp_pop.shape[1]//2
plt.plot(np.arange(-30,30),-comp_pop[42, center-30:center+30], '.', ls='dotted', label='42')
plt.plot(np.arange(-30,30),comp_pop[34, center-30:center+30], 'x', ls='solid', label='34')
plt.ylim(-0.5,0.5)
plt.legend()

