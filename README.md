# PyDRT
PyDRT is a lightweight Python implementation of the regularization-regression DRT with basis functions.
The present code is published as part of the following work:
> Leonhardt, et al. (2024). "Reconstructing the distribution of relaxation times with analytical basis functions" Journal of Y, Submitted, DOI: Y

If this code helps you with your research, please consider citing the reference above - this would be very helpful. :)

In case you want a more convenient DRT experience with a more user-friencly GUI, check out Polarographica:
https://github.com/Polarographica/Polarographica_program

## Usage
In the simplest case, the DRT can be set up as follows.
```python
from PyDRT import DebyeDRT

drt = DebyeDRT(frequency_data_Hz, impedance_data_Ohm, epsilon = 0.001)
```
