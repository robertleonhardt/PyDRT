# PyDRT
PyDRT is a lightweight Python implementation of the regularization-regression DRT with basis functions.

## Usage
In the simplest case, the DRT can be set up as follows.
```
from PyDRT import DebyeDRT

drt = DebyeDRT(frequency_data_Hz, impedance_data_Ohm, epsilon = 0.001)
```
