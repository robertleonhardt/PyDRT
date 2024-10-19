# PyDRT
PyDRT is a lightweight Python implementation of the regularization-regression DRT with basis functions.
The present code is published as part of the following work:
> Leonhardt, et al. (2024). "Reconstructing the distribution of relaxation times with analytical basis functions" Journal of Y, Submitted, DOI: Y

If this code helps you with your research, please consider citing the reference above - this would be very helpful. :)

In case you want a more convenient DRT experience with a more user-friencly GUI, check out Polarographica:
https://github.com/Polarographica/Polarographica_program

## Basic usage
In the simplest case, the DRT can be set up as follows.
```python
from PyDRT import DebyeDRT

# Setup arbitrary model
frequency_model_Hz  = np.geomspace(1000, 0.001, 70)
impedance_model_Ohm = lambda omega: 1 + 2/(1 + 1j * 2 * np.pi * omega * 0.1) ** 0.99 + 4/(1 + 1j * 2 * np.pi * omega * 1) * 0.99
impedance_model_Ohm = impedance_model_Ohm(frequency_model_Hz)

# Determine DRT
drt = DebyeDRT(frequency_model_Hz, impedance_model_Ohm, epsilon = 0.001)
```

The DRT primarily takes two inputs, a frequency vector and a complex-valued impedance vector of the same length.
The latter can be constructed in python by `impedance_model_Ohm = real_part + 1j * imag_part`.
