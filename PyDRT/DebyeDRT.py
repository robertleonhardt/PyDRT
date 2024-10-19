from __future__ import annotations
import numpy as np
import numpy.typing as npt
from .DRT import DRT

class DebyeDRT(DRT):
    """
    Native DRT with a Dirac-distribution basis
    
    @version:   AA-20240712(20240712)
    @code:      Robert Leonhardt - mail@robertleonhardt.de
    """
    
    def _get_basis_matrix(self) -> npt.ArrayLike:
        return np.identity(len(self.tau_s))
    
    def get_separated_peak_list(self) -> None:
        raise NotImplementedError('Separation of peaks is not implemented in DebyeDRT. Please one one of the other bases.')