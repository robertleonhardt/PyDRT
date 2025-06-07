import numpy as np
import matplotlib.pyplot as plt
from pydrt import ColeColeDRT, DRTPeak

# Setup model
frequency_model_Hz  = np.geomspace(1000, 0.001, 70)
impedance_model_Ohm = lambda omega: 1 + 2/(1 + (1j * 2 * np.pi * omega * 0.2) ** 0.86) + 4/(1 + (1j * 2 * np.pi * omega * 1) ** 0.86)
impedance_model_Ohm = impedance_model_Ohm(frequency_model_Hz)

# Setup DRT and get peak list
drt             = ColeColeDRT.optimize_shape_parameters(ColeColeDRT(frequency_model_Hz, impedance_model_Ohm, tau_max_s = 1e1))
drt_peak_list   = drt.get_separated_peak_list()

# Setup plot
fig, [eis_ax, drt_ax] = plt.subplots(ncols = 2, nrows = 1, figsize = (10,3), constrained_layout = True)

# Plot impedance
eis_ax.axhline(0, color = 'k', lw = 0.5)
eis_ax.plot(impedance_model_Ohm.real, -impedance_model_Ohm.imag, 'o', color = 'k', markerfacecolor = '#ffffff', label = 'Data')
eis_ax.plot(drt.z_back_Ohm.real, -drt.z_back_Ohm.imag, '-', color = 'k', label = 'DRT reconstruction')

# Plot model for the identified peaks
R_offset_Ohm = drt.R_offset_Ohm
for peak in drt_peak_list:
    # Setup process model
    impedance_process_model_Ohm = lambda omega: R_offset_Ohm + peak.R_Ohm / (1 + (1j * 2 * np.pi * omega * peak.tau_s) ** drt.alpha)
    impedance_process_model_Ohm = impedance_process_model_Ohm(frequency_model_Hz)
    
    # Plot process
    eis_ax.fill_between(impedance_process_model_Ohm.real, -impedance_process_model_Ohm.imag, color = 'k', alpha = 0.2)
    
    # Count-up resistance offset
    R_offset_Ohm += peak.R_Ohm

# Plot DRT
drt_ax.axhline(0, color = 'k', lw = 0.5)
for peak in drt_peak_list:
    drt_ax.fill_between(drt.tau_s, peak.gamma_hat_Ohm, '-', color = 'k', alpha = 0.2)
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