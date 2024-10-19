import numpy as np
import matplotlib.pyplot as plt
from PyDRT import DebyeDRT

# Setup model
frequency_model_Hz  = np.geomspace(1000, 0.001, 70)
impedance_model_Ohm = lambda omega: 1 + 2/(1 + 1j * 2 * np.pi * omega * 0.1) ** 0.99 + 4/(1 + 1j * 2 * np.pi * omega * 1) * 0.99
impedance_model_Ohm = impedance_model_Ohm(frequency_model_Hz)

# Setup DRT
drt = DebyeDRT(frequency_model_Hz, impedance_model_Ohm, epsilon = 0.001)

# Setup plot
fig, [eis_ax, drt_ax] = plt.subplots(ncols = 2, nrows = 1, figsize = (10,3), constrained_layout = True)

# Plot impedance
eis_ax.axhline(0, color = 'k', lw = 0.5)
eis_ax.plot(impedance_model_Ohm.real, -impedance_model_Ohm.imag, 'o', color = 'k', markerfacecolor = '#ffffff', label = 'Data')
eis_ax.plot(drt.z_back_Ohm.real, -drt.z_back_Ohm.imag, '-', color = 'k', label = 'DRT reconstruction')

# Plot DRT
drt_ax.axhline(0, color = 'k', lw = 0.5)
drt_ax.plot(drt.tau_s, drt.gamma_hat_Ohm, '-', color = 'k')

# Config plot
for ax in [eis_ax]:
    ax.set_aspect('equal', 'datalim')
    ax.minorticks_on()
    
    ax.legend(loc = 'upper left')
    
    ax.set_xlabel(r"$Z'\;/\;\mathrm{m\Omega}$")
    ax.set_ylabel(r"$Z''\;/\;\mathrm{m\Omega}$") 
    
for ax in [drt_ax]:
    ax.set_xscale('log')
    ax.minorticks_on()
    
    ax.set_xlim([1e-4, 1e2])
    
    ax.set_xlabel(r'$\tau\;/\;\mathrm{s}$')
    ax.set_ylabel(r'$R_p\,\gamma\left(\ln{\left(\tau/\tau_c\right)}\right)\;/\;\mathrm{m\Omega}$')

plt.show()